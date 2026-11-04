#!/usr/bin/env python
"""Ingest: the tranche's real Vietnamese segments, one parquet per source shard, in the corpus's
own order ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)).

  segments <source>  download each source shard at its pinned commit, keep its usable segments,
                     re-encode their audio to Ogg Opus and write
                     <root>/<repo>/<source>/segments/<shard>.parquet; the download is deleted
  push <source>      upload <root>/<repo>/<source> to its tranche repo (resumable)

Sources go to the public repo, or to the private one where their terms forbid redistribution
(PhoAudiobook, GigaSpeech 2). Rows: id, source, doc (the reader, channel or recording whose
segments share a voice), seq (order within doc), speaker, start_s/end_s (within the recording,
where the corpus gives them), text (the corpus transcript), duration_s, source_rate, audio
{bytes, path}. A rerun skips shards already written. CPU only; from the repo root on the box:
  .venv/bin/python -m training.tranche.ingest segments pho --shards 0:40
"""
from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/workspace/tranche/repo")
REPOS = {"public": "huybik/hibiki-zero-vi-real", "restricted": "huybik/hibiki-zero-vi-real-restricted"}
OPUS_BITRATE = 32_000  # real speech, noise included; Mimi keeps far less than this
OPUS_COMPLEXITY = 5
ROW_S = (5.0, 22.4)  # the trainer's row lengths, for sources whose rows cannot be rejoined
# PhoAudiobook's train split is 940 h of clips of 10 s and more plus 554 h of "augmented short
# audio" (its card): only the originals are kept.
PHO_MIN_S = 10.0


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    pattern: str  # the source shards, a regex over the repo's file names
    dest: str  # "public" or "restricted"


SOURCES = {
    "pho": Source("thivux/phoaudiobook", "f19f241", r"data/train-\d+-of-\d+\.parquet", "restricted"),
    "vivoice": Source("capleaf/viVoice", "693e988", r"data/train-\d+-of-\d+\.parquet", "public"),
    "giga": Source("speechcolab/gigaspeech2", "8dc0d0e", r"data/vi/train/\d+\.tar\.gz", "restricted"),
}
SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("doc", pa.string()), ("seq", pa.int64()),
    ("speaker", pa.string()), ("start_s", pa.float64()), ("end_s", pa.float64()),
    ("text", pa.string()), ("duration_s", pa.float64()), ("source_rate", pa.int32()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
])


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Any container PyAV reads -> mono float32 at its own rate."""
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = [f.to_ndarray()[0] for frame in container.decode(stream) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray()[0] for f in resampler.resample(None)]
    return np.concatenate(chunks), rate


def encode_opus(pcm: np.ndarray, rate: int) -> bytes:
    """Mono float32 -> Ogg Opus; libopus takes 8, 12, 16, 24 or 48 kHz input as it is."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=rate, layout="mono")
        stream.bit_rate = OPUS_BITRATE
        stream.options = {"compression_level": str(OPUS_COMPLEXITY)}
        frame = av.AudioFrame.from_ndarray(np.clip(pcm, -1, 1)[None].astype(np.float32), format="flt",
                                           layout="mono")
        frame.sample_rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue()


def times(path: str) -> tuple[float, float]:
    """PhoAudiobook `chunk-2541_74-2554_64_…` and viVoice `audio_502.22_504.3.wav` -> start, end."""
    m = re.match(r"chunk-(\d+)_(\d+)-(\d+)_(\d+)", path)
    if m:
        return float(f"{m[1]}.{m[2]}"), float(f"{m[3]}.{m[4]}")
    m = re.match(r"audio_([\d.]+)_([\d.]+)\.wav", path)
    if m:
        return float(m[1]), float(m[2])
    raise ValueError(f"no times in {path!r}")


def rows_of(name: str, shard: int, local: Path) -> Iterator[dict]:
    """(row fields, audio bytes) of one downloaded source shard, in its own order."""
    if name in ("pho", "vivoice"):
        for k, r in enumerate(pq.read_table(local).to_pylist()):
            start, end = times(r["audio"]["path"])
            speaker = r["speaker"] if name == "pho" else r["channel"]
            yield {"id": f"{name}/{shard:05d}/{k:05d}", "doc": speaker, "seq": shard * 100_000 + k,
                   "speaker": speaker, "start_s": start, "end_s": end, "text": r["text"],
                   "data": r["audio"]["bytes"]}
    else:
        texts = giga_texts()
        with tarfile.open(local, "r|gz") as tar:
            for member in tar:
                key = Path(member.name).stem
                if member.isfile() and key in texts:
                    archive, recording, segment = key.split("-")
                    doc = f"{archive}-{recording}"
                    yield {"id": f"giga/{key}", "doc": doc, "seq": int(segment), "speaker": doc,
                           "start_s": None, "end_s": None, "text": texts[key],
                           "data": tar.extractfile(member).read()}


_GIGA_TEXTS: dict[str, str] = {}


def giga_texts() -> dict[str, str]:
    """GigaSpeech 2's refined transcripts by segment id, read once per worker."""
    from huggingface_hub import hf_hub_download

    if not _GIGA_TEXTS:
        src = SOURCES["giga"]
        path = hf_hub_download(src.repo, "data/vi/train_refined.tsv", repo_type="dataset", revision=src.revision)
        for line in open(path):
            key, text = line.rstrip("\n").split("\t", 1)
            _GIGA_TEXTS[key] = text
    return _GIGA_TEXTS


def ingest_shard(name: str, index: int, file: str, out: Path) -> str:
    from huggingface_hub import hf_hub_download

    src = SOURCES[name]
    stem = re.sub(r"\W+", "_", Path(file).name.split(".")[0])
    dst = out / f"{stem}.parquet"
    if dst.exists():
        return f"{file}: exists"
    scratch = out.parent / ".download"
    local = Path(hf_hub_download(src.repo, file, repo_type="dataset", revision=src.revision, local_dir=scratch))
    rows, dropped = [], 0
    try:
        for row in rows_of(name, index, local):
            pcm, rate = decode(row.pop("data"))
            duration = len(pcm) / rate
            if name == "vivoice" and not ROW_S[0] <= duration <= ROW_S[1]:  # shuffled: no neighbours
                dropped += 1
                continue
            if name == "pho" and duration < PHO_MIN_S:  # its augmented short copies
                dropped += 1
                continue
            rows.append(row | {"source": name, "duration_s": round(duration, 3), "source_rate": rate,
                               "audio": {"bytes": encode_opus(pcm, rate), "path": f"{row['id'].replace('/', '_')}.opus"}})
    finally:
        local.unlink()
    part = dst.with_name(dst.name + ".part")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), part, compression="zstd")
    os.replace(part, dst)
    hours = sum(r["duration_s"] for r in rows) / 3600
    return f"{file}: {len(rows)} rows, {hours:.2f} h, {dropped} dropped"


def shard_files(name: str, selection: str | None) -> list[str]:
    from huggingface_hub import HfApi

    src = SOURCES[name]
    files = sorted((f for f in HfApi().list_repo_files(src.repo, repo_type="dataset", revision=src.revision)
                    if re.fullmatch(src.pattern, f)), key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
    if selection:
        start, _, stop = selection.partition(":")
        files = files[int(start or 0): int(stop) if stop else None]
    return files


def segments(args: argparse.Namespace) -> None:
    name = args.source
    out = args.root / SOURCES[name].dest / name / "segments"
    out.mkdir(parents=True, exist_ok=True)
    files = shard_files(name, args.shards)
    print(f"{name}: {len(files)} shards -> {out}", flush=True)
    # Each shard's rows keep their order through its index in the whole list.
    everything = shard_files(name, None)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(ingest_shard, name, everything.index(f), f, out) for f in files]
        for job in jobs:
            print(job.result(), flush=True)


def push(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    dest = SOURCES[args.source].dest
    folder = args.root / dest
    HfApi().upload_large_folder(repo_id=REPOS[dest], repo_type="dataset", folder_path=folder,
                                allow_patterns=[f"{args.source}/segments/*.parquet"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("segments")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--shards", help="a slice of the sorted source shards, e.g. 0:40")
    p.add_argument("--workers", type=int, default=8)
    p = steps.add_parser("push")
    p.add_argument("source", choices=sorted(SOURCES))
    args = parser.parse_args()
    {"segments": segments, "push": push}[args.step](args)


if __name__ == "__main__":
    main()
