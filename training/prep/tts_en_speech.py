#!/usr/bin/env python
"""Voice the Vietnamese text of the EN-VI dataset's en_speech rows in each row's own English voice.

VieNeu-TTS v3 Turbo in MLX (training/prep/vieneu_mlx.py) clones the speaker of the row's own
English clip: its first 8 s, edge silence trimmed, give the x-vector and the in-context reference
codes. The text is cut into upstream's chunks of at most 256 characters, spoken, and the chunks are
joined with upstream's pauses.

One stream runs across shards. CPU workers enroll rows ahead of the GPU and encode the finished
audio; the GPU encodes references, generates every chunk with continuous batching and decodes
finished rows in groups. Each dataset shard gets one output shard,
--out/{split}/{shard}.parquet with id, vi_audio{bytes,path} (48 kHz mono Ogg Opus, ~32 kbps) and
vi_duration_s, written whole once its last row is encoded. A run skips the shards already written,
so it resumes. Validation shards come first, then train shards in a fixed shuffled order, so a run
cut short has voiced a random sample.

Run from the repo root on Apple Silicon with the conda base Python:
  python -m training.prep.tts_en_speech [--shards N]
"""
from __future__ import annotations

import argparse
import io
import os
import random
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
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks  # noqa: E402
from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions  # noqa: E402

from training.prep import vieneu_mlx  # noqa: E402
from training.prep.publish_en_vi_speech import AUDIO, write_shard  # noqa: E402
from training.prep.translate_en_speech import exclusive, log  # noqa: E402

MAX_CHARS = 256  # upstream's chunk length
# libopus at complexity 3 and a 24 kbps target gives ~32 kbps on this speech and encodes at 250x
# real time a core. libsndfile's Opus (complexity 10) managed 41x at the same size, which left the
# CPU workers behind the GPU; its log-mel distance to the source was 1.24 dB, this one's 1.41 dB.
OPUS_COMPLEXITY = 3
OPUS_BITRATE = 24_000
PREFETCH = 256  # rows enrolled ahead of the GPU; each holds an 8 s 48 kHz reference (1.5 MB)
ENROLL = 128  # prepared rows whose references are encoded together, grouped by length
DECODE_BATCH = 8192  # frames of finished rows decoded together (11 minutes of audio)
SCHEMA = pa.schema([("id", pa.string()), ("vi_audio", AUDIO), ("vi_duration_s", pa.float32())])

_enroll: vieneu_mlx.Enroller | None = None


def _start_worker() -> None:
    global _enroll
    _enroll = vieneu_mlx.Enroller()


def prepare(audio: bytes, text: str) -> tuple[vieneu_mlx.Voice, list[str], list[float]]:
    """CPU work for one row: its voice, its chunks' phonemes and the pauses between them."""
    wav, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=MAX_CHARS)
    if not chunks:
        raise ValueError(f"nothing to speak in {text!r}")
    return _enroll(wav.mean(axis=1), sr), [phonemize_text_with_emotions(c) for c in chunks], gaps_to_silence(gaps)


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
    found = sorted(p for p in root.glob("data/*/*.parquet") if not p.name.startswith("phomt-"))
    train = [p for p in found if p.parent.name == "train"]
    random.Random(0).shuffle(train)
    return [p for p in found if p.parent.name == "validation"] + train


@dataclass
class Shard:
    target: Path
    ids: list[str]
    clips: list[tuple[bytes, float] | None]
    left: int


@dataclass
class Row:
    shard: Shard
    index: int
    pauses: list[float]
    codes: list[np.ndarray | None]  # per chunk, once generated


def read(paths: list[Path], out: Path) -> Iterator[tuple[Shard, int, bytes, str]]:
    """Each row of each shard, one shard in memory at a time."""
    for path in paths:
        t = pq.read_table(path, columns=["id", "en_audio", "text_vi"])
        ids, texts = t["id"].to_pylist(), t["text_vi"].to_pylist()
        audio = [a["bytes"] for a in t["en_audio"].to_pylist()]
        del t
        shard = Shard(out / path.parent.name / path.name, ids, [None] * len(ids), len(ids))
        for i in range(len(ids)):
            yield shard, i, audio[i], texts[i]


class Voicer:
    def __init__(self, synth: vieneu_mlx.Synthesizer, pool: ProcessPoolExecutor, batch: int):
        self.synth, self.pool, self.batch = synth, pool, batch
        self.gen = vieneu_mlx.Generator(synth, batch)
        self.prepared: deque = deque()  # (shard, index, future of prepare)
        self.generated: list[Row] = []  # every chunk has codes; waiting to be decoded
        self.encoding: deque = deque()  # (row, future of encode)
        self.start, self.hours = time.perf_counter(), 0.0

    def run(self, paths: list[Path], out: Path) -> None:
        source = read(paths, out)
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
        """Encode the references of the next prepared rows and queue their chunks."""
        take = [self.prepared.popleft()]
        while self.prepared and len(take) < ENROLL and self.prepared[0][2].done():
            take.append(self.prepared.popleft())
        rows = [(shard, i, f.result()) for shard, i, f in take]
        ref_codes = self.synth.encode_refs([voice.ref for _, _, (voice, _, _) in rows])
        anchors = self.synth.speaker_anchor(mx.array(np.stack([voice.speaker_emb for _, _, (voice, _, _) in rows])))
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
        table = pa.Table.from_pydict(
            {
                "id": shard.ids,
                "vi_audio": [{"bytes": b, "path": f"{i.rsplit('/', 1)[-1]}.opus"} for i, (b, _) in zip(shard.ids, shard.clips)],
                "vi_duration_s": [d for _, d in shard.clips],
            },
            schema=SCHEMA,
        )
        shard.target.parent.mkdir(parents=True, exist_ok=True)
        write_shard(shard.target, table)
        hours = sum(d for _, d in shard.clips) / 3600
        self.hours += hours
        elapsed = time.perf_counter() - self.start
        log(f"{shard.target.parent.name}/{shard.target.name}: {len(shard.ids)} rows, {hours:.2f} h | "
            f"{self.hours:.1f} h in {elapsed / 3600:.2f} h, {self.hours * 3600 / elapsed:.0f}x real time")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--out", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_tts"))
    p.add_argument("--shards", type=int, help="voice at most this many shards, then stop")
    p.add_argument("--batch", type=int, default=128, help="chunks generated together")
    p.add_argument("--workers", type=int, default=6, help="CPU processes enrolling and encoding")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # so a `kill` shuts the workers down too
    with exclusive(args.out / ".tts.lock", f"another TTS run is writing {args.out}"):
        todo = [s for s in shards(args.root) if not (args.out / s.parent.name / s.name).exists()][: args.shards]
        log(f"{len(todo)} shards to voice")
        synth = vieneu_mlx.Synthesizer()
        with ProcessPoolExecutor(args.workers, initializer=_start_worker) as pool:
            try:
                Voicer(synth, pool, args.batch).run(todo, args.out)
            finally:
                pool.shutdown(cancel_futures=True)  # without running the rows queued ahead


if __name__ == "__main__":
    sys.exit(main())
