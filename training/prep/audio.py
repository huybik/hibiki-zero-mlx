"""Audio decoding shared by the cache builder (`cache_codes`) and the noisy re-encode
(`noisy_cache`), so a re-encoded clip has the samples, and so the frames, the cache was built from.
"""

from __future__ import annotations

import io

import av
import numpy as np


class Rejected(Exception):
    """A data fault in one row (bad audio, too long), which the cache builder rejects; any other
    error stops its run. The noisy re-encode stops on either."""


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Opus or MP3 bytes -> mono float32 PCM at its own rate (Opus 48 kHz, MP3 32/48 kHz)."""
    try:
        with av.open(io.BytesIO(data)) as container:
            chunks, rate = [], 0
            for frame in container.decode(audio=0):
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
