#!/usr/bin/env python
"""Ingest: the tranche's real Vietnamese segments, one parquet per source shard, in the corpus's
own order ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)).

  segments <source>  download each source shard at its pinned commit, keep its usable segments,
                     re-encode their audio to Ogg Opus and write
                     <root>/<source>/segments/<shard>.parquet; the download is deleted
  push <source>      upload <root>/<source>/segments, or with --part another stage's folder
                     (text_en, en_audio, asr_vi; cache: <root>/cache/<source>, to CACHE_REPO), to
                     the tranche repo (a rerun skips files already committed), only while it is gated

Every source goes to the public repos of the anquachdev account (`REPO`, `CACHE_REPO`), gated by the
sources' terms: GigaSpeech 2's own terms of access, which let a researcher share it with those who first
agree to them; PhoAudiobook with its author's permission (the user, 2026-11-06). Rows, sorted by
seq within a shard:
  id, source
  doc       the recording the row comes from, as far as the corpus tells; a doc never spans
            shards. PhoAudiobook: one reader's run of chunks in rising time (it names no book);
            GigaSpeech 2: the recording; viVoice: the row itself (its rows are shuffled).
  seq       the row's order within its shard, and so within its doc
  speaker   the corpus's reader or channel, which can be a platform rather than one voice;
            none for GigaSpeech 2, whose recordings may hold several voices
  start_s, end_s  the times the corpus names the clip by, within the recording; the trimmed
            audio can be seconds shorter than end - start, so seams come from alignment
  text      the corpus transcript; duration_s, source_rate and audio {bytes, path} of the clip
A rerun skips shards already written. CPU only; from the repo root on the box:
  .venv/bin/python -m training.tranche.ingest segments pho --shards 0:40
"""
from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/workspace/tranche/repo")
# Public, gated by the sources' terms (GigaSpeech 2's in full); the anquachdev account's, so pushes take
# HF_TOKEN_AN. REPO holds every part but the caches (beside PhoMT's synthetic pairs in data/), as
# <source>/<part>/<shard>; CACHE_REPO the caches, as <source>/shard_<shard>.pt.
REPO = "anquachdev/en-vi-speech-20k"
CACHE_REPO = "anquachdev/mimi-en-vi-speech-20k"
OPUS_BITRATE = 32_000  # real speech, noise included; Mimi keeps far less than this
OPUS_COMPLEXITY = 5
ROW_S = (5.0, 22.4)  # the trainer's row lengths, for sources whose rows cannot be rejoined
# PhoAudiobook's train split is 947 h of 10-20 s chunks plus 554 h of "augmented short audio" (its
# card), cut from the same readings: only the originals, which this suffix names, are kept.
PHO_ORIGINAL = "_trimmed_norm_float32.wav"


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    pattern: str  # the source shards, a regex over the repo's file names
    shards: int | None = None  # how many of the sorted shards hold usable rows; None: all


SOURCES = {
    # Shards 224-802 hold only augmented audio.
    "pho": Source("thivux/phoaudiobook", "f19f241", r"data/train-\d+-of-\d+\.parquet", 224),
    "vivoice": Source("capleaf/viVoice", "693e988", r"data/train-\d+-of-\d+\.parquet"),
    "giga": Source("speechcolab/gigaspeech2", "8dc0d0e", r"data/vi/train/\d+\.tar\.gz"),
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
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU
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
    """The usable rows of one downloaded source shard, each with its source audio as `data`."""
    if name == "pho":
        last = None  # (reader, end) of the previous original
        for k, r in enumerate(pq.read_table(local).to_pylist()):
            if not r["audio"]["path"].endswith(PHO_ORIGINAL):
                continue
            start, end = times(r["audio"]["path"])
            row_id = f"pho/{shard:05d}/{k:05d}"
            if last is None or last[0] != r["speaker"] or start < last[1]:  # another reading
                doc = row_id
            last = (r["speaker"], end)
            yield {"id": row_id, "doc": doc, "seq": k, "speaker": r["speaker"], "start_s": start,
                   "end_s": end, "text": r["text"], "data": r["audio"]["bytes"]}
    elif name == "vivoice":
        for k, r in enumerate(pq.read_table(local).to_pylist()):
            start, end = times(r["audio"]["path"])
            row_id = f"vivoice/{shard:05d}/{k:05d}"
            yield {"id": row_id, "doc": row_id, "seq": k, "speaker": r["channel"], "start_s": start,
                   "end_s": end, "text": r["text"], "data": r["audio"]["bytes"]}
    else:
        archive = local.name.split(".")[0]
        texts = giga_texts(archive)  # the refined transcripts: segments without one are dropped
        found = 0
        with tarfile.open(local, "r|gz") as tar:  # members come in file-system order
            for member in tar:
                key = Path(member.name).stem
                if member.isfile() and key in texts:
                    found += 1
                    _, recording, segment = key.split("-")
                    yield {"id": f"giga/{key}", "doc": f"giga/{archive}-{recording}",
                           "seq": int(recording) * 1_000_000 + int(segment), "speaker": None,
                           "start_s": None, "end_s": None, "text": texts[key],
                           "data": tar.extractfile(member).read()}
        if found != len(texts):
            raise ValueError(f"archive {archive}: {len(texts) - found} of {len(texts)} transcripts have no audio")


def giga_texts(archive: str) -> dict[str, str]:
    """GigaSpeech 2's refined transcripts of one archive's segments, by segment id."""
    from huggingface_hub import hf_hub_download

    src = SOURCES["giga"]
    path = hf_hub_download(src.repo, "data/vi/train_refined.tsv", repo_type="dataset", revision=src.revision)
    with open(path) as f:
        return dict(line.rstrip("\n").split("\t", 1) for line in f if line.startswith(f"{archive}-"))


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
            rows.append(row | {"source": name, "duration_s": round(duration, 3), "source_rate": rate,
                               "audio": {"bytes": encode_opus(pcm, rate), "path": f"{row['id'].replace('/', '_')}.opus"}})
    finally:
        local.unlink()
    rows.sort(key=lambda r: r["seq"])
    part = dst.with_name(dst.name + ".part")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), part, compression="zstd")
    os.replace(part, dst)
    hours = sum(r["duration_s"] for r in rows) / 3600
    return f"{file}: {len(rows)} rows, {hours:.2f} h, {dropped} dropped"


def shard_files(name: str) -> list[str]:
    """The source's usable shards in numeric order; a shard's index here names its rows."""
    from huggingface_hub import HfApi

    src = SOURCES[name]
    files = sorted((f for f in HfApi().list_repo_files(src.repo, repo_type="dataset", revision=src.revision)
                    if re.fullmatch(src.pattern, f)), key=lambda f: [int(n) for n in re.findall(r"\d+", f)])
    return files[:src.shards]


def segments(args: argparse.Namespace) -> None:
    name = args.source
    out = args.root / name / "segments"
    out.mkdir(parents=True, exist_ok=True)
    files = shard_files(name)
    picked = range(len(files))[args.shards]
    print(f"{name}: {len(picked)} of {len(files)} shards -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(ingest_shard, name, i, files[i], out) for i in picked]
        try:
            for job in as_completed(jobs):
                print(job.result(), flush=True)
        except BaseException:
            pool.shutdown(cancel_futures=True)  # stop at the first failure, not after the queue
            raise


def an_token() -> str:
    """The write token of the anquachdev account, which owns REPO and CACHE_REPO; the stages that
    push read it at their start, so a missing one stops them before any work."""
    return os.environ["HF_TOKEN_AN"]


def push(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=an_token())
    repo, folder, pattern = ((CACHE_REPO, args.root / "cache", f"{args.source}/*.pt") if args.part == "cache"
                             else (REPO, args.root, f"{args.source}/{args.part}/*.parquet"))
    # upload_folder never creates the repo, so a missing one fails instead of appearing ungated.
    if not api.dataset_info(repo).gated:
        raise SystemExit(f"{repo} is not gated: its sources may go only to those who accept their terms")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, allow_patterns=[pattern],
                      commit_message=f"{args.source} {args.part}")


def part_file(root: Path, source: str, part: str, name: str) -> Path:
    """A shard's part (segments, text_en, en_audio, asr_vi): the box's own copy under `root` where a
    stage wrote one (each writes whole, by a rename, and keeps it after pushing), else the Hub's (the
    HF cache keeps it). The box's downloads through Xet run ~5 MB/s (`HF_HUB_DISABLE_XET=1`: ~140 MB/s), and a second copy in
    the HF cache doubles a shard's disk."""
    local = root / source / part / name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, f"{source}/{part}/{name}", repo_type="dataset"))


def push_part(root: Path, source: str, part: str) -> None:
    """`push`, tried again after a transient Hub error, for the stages that push as they go."""
    import time

    an_token()  # a missing token is not a Hub error: fail now, not after the retries
    for attempt in range(4):
        try:
            push(argparse.Namespace(root=root, source=source, part=part))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"pushed {source} {part}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - the Hub's 5xx and timeouts come as many types
            if attempt == 3:
                raise
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"push {source} {part} failed "
                  f"({type(exc).__name__}: {exc}); again in {60 * 2 ** attempt} s", flush=True)
            time.sleep(60 * 2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("segments")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                   default=slice(None), help="a slice of the usable source shards, e.g. 0:40")
    p.add_argument("--workers", type=int, default=8)
    p = steps.add_parser("push")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--part", choices=["segments", "text_en", "en_audio", "asr_vi", "cache"], default="segments",
                   help="the folder to upload: the segments, their translations (translate.py), their "
                        "English speech (voice.py), their Vietnamese ASR (filters.py) or their "
                        "training cache (cache.py)")
    args = parser.parse_args()
    {"segments": segments, "push": push}[args.step](args)


if __name__ == "__main__":
    main()
