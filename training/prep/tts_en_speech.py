#!/usr/bin/env python
"""Voice the Vietnamese text of the EN-VI dataset's en_speech rows in each row's own English voice.

VieNeu-TTS v3 Turbo in MLX (training/prep/vieneu_mlx.py) clones the speaker of the row's own
English clip: its first 8 s, edge silence trimmed, give the x-vector and the in-context reference
codes. The text is cut into upstream's chunks of at most 256 characters, spoken, and the chunks are
joined with upstream's pauses.

One stream runs across shards. CPU workers prepare rows ahead of the GPU (decoding, fbank,
phonemes) and encode the finished audio; the GPU takes the x-vectors, encodes references, generates
every chunk with continuous batching and decodes finished rows in groups. Once its last row is
encoded, a dataset shard is rewritten with its vi_audio{bytes,path} (48 kHz mono Ogg Opus,
~32 kbps) and vi_duration_s filled. A run skips the shards already voiced, so it resumes; the
publisher's `upload --follow` sends them to the Hub as they come. Shards go top down in path order
while another machine voices them bottom up; the upload brings in the shards voiced there, and the
run skips them, so the two meet in between.

Run from the repo root on Apple Silicon with the conda base Python:
  python -m training.prep.tts_en_speech [--shards N]
"""
from __future__ import annotations

import argparse
import io
import os
import signal
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# As in the publisher: macOS's default Arrow pool keeps freed shard buffers.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import av  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks  # noqa: E402
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions  # noqa: E402

from training.prep import vieneu_mlx  # noqa: E402
from training.prep.publish_en_vi_speech import AUDIO, lock, voiced, write_shard  # noqa: E402
from training.prep.translate_en_speech import log  # noqa: E402

MAX_CHARS = 256  # upstream's chunk length
# libopus at complexity 3 and a 24 kbps target gives ~32 kbps on this speech and encodes at 250x
# real time a core. libsndfile's Opus (complexity 10) managed 41x at the same size, which left the
# CPU workers behind the GPU; its log-mel distance to the source was 1.24 dB, this one's 1.41 dB.
OPUS_COMPLEXITY = 3
OPUS_BITRATE = 24_000
PREFETCH = 256  # rows prepared ahead of the GPU; each holds an 8 s 48 kHz fp16 reference (0.8 MB)
ENROLL = 128  # prepared rows whose x-vectors and references are taken together, grouped by length
DECODE_BATCH = 8192  # frames of finished rows decoded together (11 minutes of audio)


def prepare(audio: bytes, text: str) -> tuple[vieneu_mlx.Voice, list[str], list[float]]:
    """CPU work for one row: its voice, its chunks' phonemes and the pauses between them."""
    wav, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=MAX_CHARS)
    if not chunks:
        raise ValueError(f"nothing to speak in {text!r}")
    return vieneu_mlx.enroll(wav.mean(axis=1), sr), [phonemize_text_with_emotions(c) for c in chunks], gaps_to_silence(gaps)


def encode(chunks: list[np.ndarray], pauses: list[float]) -> tuple[bytes, float]:
    wav = np.clip(join_audio_chunks(chunks, vieneu_mlx.SAMPLE_RATE, silence_ps=pauses), -1.0, 1.0)
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=vieneu_mlx.SAMPLE_RATE, layout="mono")
        stream.bit_rate = OPUS_BITRATE
        stream.options = {"compression_level": str(OPUS_COMPLEXITY)}
        frame = av.AudioFrame.from_ndarray(wav[None].astype(np.float32), format="flt", layout="mono")
        frame.sample_rate = vieneu_mlx.SAMPLE_RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue(), len(wav) / vieneu_mlx.SAMPLE_RATE


def shards(root: Path) -> list[Path]:
    """Top down: data/train/cv27-*, data/train/libriheavy-*, then data/validation/*."""
    return sorted(root.glob("data/*/*.parquet"))


@dataclass
class Shard:
    path: Path
    ids: list[str]
    clips: list[tuple[bytes, float] | None]
    left: int


@dataclass
class Row:
    shard: Shard
    index: int
    pauses: list[float]
    codes: list[np.ndarray | None]  # per chunk, once generated


def read(paths: list[Path]) -> Iterator[tuple[Shard, int, bytes, str]]:
    """Each row of each shard, one shard in memory at a time."""
    for path in paths:
        if voiced(path):  # pulled from the Hub, voiced bottom up, since the run began
            continue
        t = pq.read_table(path, columns=["id", "en_audio", "text_vi"])
        ids, texts = t["id"].to_pylist(), t["text_vi"].to_pylist()
        audio = [a["bytes"] for a in t["en_audio"].to_pylist()]
        del t
        shard = Shard(path, ids, [None] * len(ids), len(ids))
        for i in range(len(ids)):
            yield shard, i, audio[i], texts[i]


def fill(path: Path, ids: list[str], clips: list[tuple[bytes, float]]) -> bool:
    """Rewrite a dataset shard with the Vietnamese clips of its rows, unless the upload pulled it
    in, voiced on the Hub, meanwhile: that voicing stays."""
    t = pq.read_table(path)
    if t["vi_duration_s"].null_count == 0:
        return False
    if t["id"].to_pylist() != ids:
        raise RuntimeError(f"{path} changed while it was voiced")
    audio = pa.array([{"bytes": b, "path": f"{i.rsplit('/', 1)[-1]}.opus"} for i, (b, _) in zip(ids, clips)], AUDIO)
    t = t.set_column(t.schema.get_field_index("vi_audio"), "vi_audio", audio)
    t = t.set_column(t.schema.get_field_index("vi_duration_s"), "vi_duration_s", pa.array([d for _, d in clips], pa.float32()))
    write_shard(path, t)
    return True


class Voicer:
    def __init__(self, synth: vieneu_mlx.Synthesizer, pool: ProcessPoolExecutor, batch: int):
        self.synth, self.pool, self.batch = synth, pool, batch
        self.gen = vieneu_mlx.Generator(synth, batch)
        self.prepared: deque = deque()  # (shard, index, future of prepare)
        self.generated: list[Row] = []  # every chunk has codes; waiting to be decoded
        self.encoding: deque = deque()  # (row, future of encode)
        self.start, self.hours = time.perf_counter(), 0.0

    def run(self, paths: list[Path]) -> None:
        source = read(paths)
        exhausted = False
        while True:
            while not exhausted and len(self.prepared) < PREFETCH:
                item = next(source, None)
                if item is None:
                    exhausted = True
                else:
                    shard, i, audio, text = item
                    self.prepared.append((shard, i, self.pool.submit(prepare, audio, text)))
            # Keep a batch of chunks queued; wait for the CPU only when the GPU would idle.
            if len(self.gen.queue) < self.batch and self.prepared and (self.prepared[0][2].done() or not self.gen.busy):
                self.enroll()
            for (row, j), codes in self.gen.run():
                row.codes[j] = codes
                if all(c is not None for c in row.codes):
                    self.generated.append(row)
            idle = not (self.gen.busy or self.prepared)
            if sum(len(c) for r in self.generated for c in r.codes) >= DECODE_BATCH or (idle and self.generated):
                self.decode()
            while self.encoding and (self.encoding[0][1].done() or (idle and not self.generated)):
                row, f = self.encoding.popleft()
                self.store(row, *f.result())
            if exhausted and not (self.gen.busy or self.prepared or self.generated or self.encoding):
                return

    def enroll(self) -> None:
        """Take the x-vectors and reference codes of the next prepared rows and queue their chunks."""
        take = [self.prepared.popleft()]
        while self.prepared and len(take) < ENROLL and self.prepared[0][2].done():
            take.append(self.prepared.popleft())
        rows = [(shard, i, f.result()) for shard, i, f in take]
        ref_codes = self.synth.encode_refs([voice.ref for _, _, (voice, _, _) in rows])
        try:
            anchors = self.synth.speaker_anchors([voice.fbank for _, _, (voice, _, _) in rows])
        except FloatingPointError as e:
            bad = [f"{rows[k][0].ids[rows[k][1]]} ({len(rows[k][2][0].fbank)} fbank frames)" for k in e.args[1]]
            raise FloatingPointError(f"non-finite x-vectors: {', '.join(bad)}") from e
        jobs = []
        for k, ((shard, i, (_, phonemes, pauses)), codes) in enumerate(zip(rows, ref_codes)):
            row = Row(shard, i, pauses, [None] * len(phonemes))
            jobs += [((row, j), self.synth.prompt_rows(ph, codes), anchors[k], ph) for j, ph in enumerate(phonemes)]
        for job in sorted(jobs, key=lambda job: len(job[1])):  # prefill groups then pad less
            self.gen.submit(*job)

    def decode(self) -> None:
        rows, self.generated = self.generated, []
        wavs = iter(self.synth.decode([c for row in rows for c in row.codes]))
        for row in rows:
            chunks = [next(wavs) for _ in row.codes]
            self.encoding.append((row, self.pool.submit(encode, chunks, row.pauses)))

    def store(self, row: Row, audio: bytes, seconds: float) -> None:
        shard = row.shard
        shard.clips[row.index] = (audio, seconds)
        shard.left -= 1
        if shard.left:
            return
        if not fill(shard.path, shard.ids, shard.clips):
            log(f"{shard.path.parent.name}/{shard.path.name}: voiced on the Hub meanwhile, so that voicing stays")
            return
        hours = sum(d for _, d in shard.clips) / 3600
        self.hours += hours
        elapsed = time.perf_counter() - self.start
        log(f"{shard.path.parent.name}/{shard.path.name}: {len(shard.ids)} rows, {hours:.2f} h | "
            f"{self.hours:.1f} h in {elapsed / 3600:.2f} h, {self.hours * 3600 / elapsed:.0f}x real time")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--shards", type=int, help="voice at most this many shards, then stop")
    p.add_argument("--batch", type=int, default=128, help="chunks generated together")
    p.add_argument("--workers", type=int, default=6, help="CPU processes enrolling and encoding")
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # so a `kill` shuts the workers down too
    with lock(args.root, "tts"):
        todo = [s for s in shards(args.root) if not voiced(s)][: args.shards]
        log(f"{len(todo)} shards to voice")
        synth = vieneu_mlx.Synthesizer()
        with ProcessPoolExecutor(args.workers) as pool:
            try:
                Voicer(synth, pool, args.batch).run(todo)
            finally:
                pool.shutdown(cancel_futures=True)  # without running the rows queued ahead


if __name__ == "__main__":
    sys.exit(main())
