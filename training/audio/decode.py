"""Audio decoding shared by the cache build (`training.cache.rows`), the noisy re-encode
(`training.cache.noisy`), the evaluation sets (`giga-heldout`) and the tranche's TTS (`training.tranche.voice`),
so a re-encoded clip has the samples, and so the frames, the cache was built from; and the Ogg Opus the tranche
writes its clips in (`encode_opus`). Each codec runs on one thread, as ffmpeg's default is a thread per host CPU
(224 on a box). `decode` does not resample: PyAV's AudioResampler holds a thread per host CPU in its ffmpeg filter
graph while it lives (they pile up), so callers resample with soxr or scipy. `encode_opus` cannot avoid it (PyAV
cuts libopus's 20 ms frames through one), so voice.py runs a few at a time.
"""

from __future__ import annotations

import io

import av
import numpy as np

OPUS_BITRATE = 32_000  # real speech, noise included; Mimi keeps far less than this
OPUS_COMPLEXITY = 5


class Rejected(Exception):
    """A data fault in one row (bad audio, too long), which the cache builder rejects; any other
    error stops its run. The noisy re-encode, the evaluation sets and voice.py stop on either."""


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Opus or MP3 bytes -> mono float32 PCM at its own rate (Opus 48 kHz, MP3 32/48 kHz)."""
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            chunks, rate = [], 0
            for frame in container.decode(stream):
                if frame.format.name != "fltp" or frame.layout.nb_channels != 1:
                    raise Rejected(
                        f"expected mono fltp audio, got {frame.format.name} {frame.layout.name}"
                    )
                chunks.append(frame.to_ndarray()[0])
                rate = frame.sample_rate
    except av.error.MemoryError:
        raise
    except av.FFmpegError as exc:
        raise Rejected(f"undecodable audio: {exc}") from None
    if not chunks:
        raise Rejected("no audio frames")
    return np.concatenate(chunks), rate


def encode_opus(pcm: np.ndarray, rate: int) -> bytes:
    """Mono float32 -> Ogg Opus; libopus takes 8, 12, 16, 24 or 48 kHz input as it is."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=rate, layout="mono")
        stream.codec_context.thread_count = 1
        stream.bit_rate = OPUS_BITRATE
        stream.options = {"compression_level": str(OPUS_COMPLEXITY)}
        frame = av.AudioFrame.from_ndarray(np.clip(pcm, -1, 1)[None].astype(np.float32), format="flt",
                                           layout="mono")
        frame.sample_rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue()
