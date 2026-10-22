#!/usr/bin/env python
"""Build a pooled English speech+text corpus for Mimi training.

Sources, both read speech with cased, punctuated transcripts:
  - Common Voice Scripted Speech 27.0 English (Mozilla Data Collective):
    validated clips. dev becomes validation and train is every other validated
    clip (the official test included) whose speaker and sentence are not in dev;
    invalidated and unvalidated clips are dropped.
  - Libriheavy (mythicinfinity/libriheavy): LibriVox audiobooks aligned to the
    book text, mono rows only. dev becomes validation; train is small + medium
    plus seeded-random large files, keeping only their 5-15 s clips, until the
    pooled train split holds --target-hours of 5-15 s clips.

Every source keeps only clips of 5-22.4 s; 22.4 s is the trainer's 280-frame row cap.

Audio keeps each source's original mono codec bytes (CV MP3, Libriheavy 48 kHz
Opus). Both are already lossy, so a lossless 24 kHz FLAC copy would be ~3x the
size for no gain and a lossy re-encode would add loss; resample to Mimi's
24 kHz at load time instead.

Every shard under --root has one schema:
  train/{cv27,libriheavy-*}.parquet, validation/{cv27,libriheavy-dev}.parquet
  id, audio{bytes,path}, text, speaker, duration (seconds)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from training.paths import REPO_ROOT

MDC_DATASET = "cmu5jplf300nwmh07iqvk9leo"
LIBRIHEAVY = "mythicinfinity/libriheavy"
CLIP_SECONDS = (5.0, 280 / 12.5)  # clip range every source keeps; the top is the trainer's 280-frame cap at 12.5 Hz
TOP_UP_SECONDS = (5.0, 15.0)  # clip range that large files keep and --target-hours counts
LARGE_FILE_HOURS = 5.5  # mean hours of mono 5-15 s clips per Libriheavy large file; sizes top-up batches
SHARD_ROWS = 20_000
DOWNLOAD_CHUNK = 1 << 28  # 256 MiB ranges fetched in parallel

FEATURES = {
    "id": {"dtype": "string", "_type": "Value"},
    "audio": {"_type": "Audio"},
    "text": {"dtype": "string", "_type": "Value"},
    "speaker": {"dtype": "string", "_type": "Value"},
    "duration": {"dtype": "float32", "_type": "Value"},
}
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text", pa.string()),
        ("speaker", pa.string()),
        ("duration", pa.float32()),
    ],
    metadata={"huggingface": json.dumps({"info": {"features": FEATURES}})},
)
# Audio bytes are already compressed; zstd only the metadata columns.
COMPRESSION = {c: "zstd" for c in ("id", "audio.path", "text", "speaker", "duration")} | {
    "audio.bytes": "none"
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_shard(path: Path, columns: dict) -> None:
    table = pa.Table.from_pydict(columns, schema=SCHEMA)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, row_group_size=500, compression=COMPRESSION)
    tmp.replace(path)


def in_range(duration: pa.ChunkedArray, seconds: tuple[float, float]) -> pa.ChunkedArray:
    return pc.and_(pc.greater_equal(duration, seconds[0]), pc.less_equal(duration, seconds[1]))


def split_hours(root: Path, split: str, seconds: tuple[float, float] = CLIP_SECONDS) -> float:
    total = 0.0
    for p in (root / split).glob("*.parquet"):
        duration = pq.read_table(p, columns=["duration"])["duration"]
        total += pc.sum(duration.filter(in_range(duration, seconds))).as_py() or 0.0
    return total / 3600


# ---------------------------------------------------------------- Common Voice


def mdc_plan() -> dict:
    key = os.environ.get("MDC_API_KEY")
    if not key:
        env = REPO_ROOT / ".env"
        lines = env.read_text().splitlines() if env.is_file() else []
        key = next(
            (l.split("=", 1)[1].strip() for l in lines if l.startswith("MDC_API_KEY=")),
            None,
        )
    if not key:
        raise SystemExit("Set MDC_API_KEY (environment or the repo .env).")
    r = requests.post(
        f"https://mozilladatacollective.com/api/datasets/{MDC_DATASET}/download",
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def download_cv(raw: Path, workers: int) -> tuple[Path, str]:
    """Parallel, resumable range download of the CV archive; returns (path, expected sha256)."""
    plan = mdc_plan()
    size, checksum = int(plan["sizeBytes"]), plan["checksum"]
    target = raw / plan["filename"]
    if target.exists() and target.stat().st_size == size:
        return target, checksum
    raw.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    done_log = target.with_name(target.name + ".done")  # indices of finished chunks
    done = set(map(int, done_log.read_text().split())) if done_log.exists() else set()
    with part.open("ab") as f:
        f.truncate(size)
    todo = [i for i in range(-(-size // DOWNLOAD_CHUNK)) if i not in done]

    def fetch(i: int) -> int:
        start = i * DOWNLOAD_CHUNK
        end = min(size, start + DOWNLOAD_CHUNK)
        for attempt in range(5):
            try:
                headers = {"Range": f"bytes={start}-{end - 1}"}
                with requests.get(plan["downloadUrl"], headers=headers, stream=True, timeout=(10, 60)) as r:
                    r.raise_for_status()
                    if r.status_code != 206:
                        raise RuntimeError("server ignored the Range header")
                    with part.open("r+b") as f:
                        f.seek(start)
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                        if f.tell() != end:
                            raise requests.ConnectionError(f"chunk {i} ended at byte {f.tell()} of {end}")
                return i
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                if attempt == 4:
                    raise
                log(f"CV chunk {i} retry after {exc!r}")
                time.sleep(10)

    ex = ThreadPoolExecutor(workers)
    try:
        with done_log.open("a") as d:
            for n, fut in enumerate(as_completed([ex.submit(fetch, i) for i in todo]), 1):
                d.write(f"{fut.result()}\n")
                d.flush()
                if n % 20 == 0 or n == len(todo):
                    log(f"CV download {len(done) + n}/{len(done) + len(todo)} chunks")
    finally:
        ex.shutdown(cancel_futures=True)
    part.replace(target)
    done_log.unlink()
    return target, checksum


class _Sha256Reader(io.RawIOBase):
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        n = self.f.readinto(b)
        self.h.update(memoryview(b)[:n])
        return n


def read_cv_tsvs(archive: Path, checksum: str) -> dict[str, list[dict]]:
    """Full pass: verify the archive sha256 and load the TSVs the split needs."""
    want = {"validated.tsv", "dev.tsv", "clip_durations.tsv"}
    tsvs = {}
    with archive.open("rb") as f:
        reader = _Sha256Reader(f)
        with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 22), mode="r|gz") as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in want:
                    text = tar.extractfile(m).read().decode("utf-8")
                    tsvs[name] = list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
                    log(f"read {m.name}: {len(tsvs[name])} rows")
            while reader.read(1 << 22):  # hash any bytes after the tar end marker
                pass
    if reader.h.hexdigest() != checksum:
        raise RuntimeError(f"CV archive sha256 {reader.h.hexdigest()} != expected {checksum}")
    if missing := want - set(tsvs):
        raise RuntimeError(f"CV archive lacks {sorted(missing)}")
    return tsvs


def cv_splits(tsvs: dict[str, list[dict]]) -> dict[str, tuple[str, dict, float]]:
    durations = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in tsvs["clip_durations.tsv"]}
    dev = tsvs["dev.tsv"]
    dev_speakers = {r["client_id"] for r in dev}
    dev_sentences = {r["sentence"].strip() for r in dev}
    keep = {r["path"]: ("validation", r) for r in dev}
    for r in tsvs["validated.tsv"]:
        if r["client_id"] not in dev_speakers and r["sentence"].strip() not in dev_sentences:
            keep[r["path"]] = ("train", r)
    return {
        path: (split, r, durations[path])
        for path, (split, r) in keep.items()
        if r["sentence"].strip() and CLIP_SECONDS[0] <= durations[path] <= CLIP_SECONDS[1]
    }


def build_cv(root: Path, workers: int) -> None:
    raw = root / "_raw"
    archive, checksum = download_cv(raw, workers)
    log(f"verifying {archive.name} and reading TSVs")
    keep = cv_splits(read_cv_tsvs(archive, checksum))
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True, exist_ok=True)
        for old in (root / split).glob("cv27-*.parquet"):
            old.unlink()
    buffers = {s: {k: [] for k in FEATURES} for s in ("train", "validation")}
    counts = {"train": 0, "validation": 0}

    def flush(split: str) -> None:
        buf = buffers[split]
        if buf["id"]:
            write_shard(root / split / f"cv27-{counts[split]:05d}.parquet", buf)
            counts[split] += 1
            buffers[split] = {k: [] for k in FEATURES}

    found = 0
    with tarfile.open(archive, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or name not in keep:
                continue
            split, r, duration = keep[name]
            buf = buffers[split]
            buf["id"].append("cv27/" + Path(name).stem)
            buf["audio"].append({"bytes": tar.extractfile(m).read(), "path": name})
            buf["text"].append(r["sentence"].strip())
            buf["speaker"].append("cv27/" + r["client_id"])
            buf["duration"].append(duration)
            found += 1
            if len(buf["id"]) >= SHARD_ROWS:
                flush(split)
                log(f"CV {found}/{len(keep)} clips written")
    for split in buffers:
        flush(split)
    if found != len(keep):
        raise RuntimeError(f"CV archive had {found} of {len(keep)} selected clips")
    archive.unlink()
    raw.rmdir()
    log(f"CV done: {found} clips, train {split_hours(root, 'train'):.1f} h total so far")


# ------------------------------------------------------------------ Libriheavy


def lh_output(root: Path, src: str) -> Path:
    # default/medium/medium-00012-of-00287.parquet -> train/libriheavy-medium-00012.parquet
    # default/large/shard-00003-of-0020/large-00045-of-00147.parquet -> train/libriheavy-large-00003-00045.parquet
    parts = src.split("/")
    subset = parts[1]
    split = "validation" if subset == "dev" else "train"
    index = "-".join(p.split("-")[1] for p in parts[2:])
    return root / split / f"libriheavy-{subset}-{index}.parquet"


def convert_lh(root: Path, src: str, seconds: tuple[float, float]) -> None:
    """Write the mono rows of one Libriheavy file whose clip lasts within `seconds`."""
    dst = lh_output(root, src)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory() as tmp:
        local = hf_hub_download(LIBRIHEAVY, src, repo_type="dataset", local_dir=tmp)
        t = pq.read_table(local, columns=["audio", "id", "speaker_id", "text_original", "audio_duration"])
    # Keep mono rows only (~10% are stereo); the channel count is byte 37 of the Ogg Opus header.
    data = pc.struct_field(t["audio"], "bytes")
    if not pc.all(pc.equal(pc.binary_slice(data, 28, 36), pa.scalar(b"OpusHead"))).as_py():
        raise RuntimeError(f"{src}: audio is not single-stream Ogg Opus")
    total = t.num_rows
    t = t.filter(pc.and_(pc.equal(pc.binary_slice(data, 37, 38), pa.scalar(b"\x01")), in_range(t["audio_duration"], seconds)))
    write_shard(
        dst,
        {
            "id": pc.binary_join_element_wise("libriheavy/", t["id"], ""),
            "audio": t["audio"],
            "text": t["text_original"],
            "speaker": pc.binary_join_element_wise("libriheavy/", t["speaker_id"], ""),
            "duration": t["audio_duration"].cast(pa.float32()),
        },
    )
    log(f"wrote {dst.relative_to(root)} ({t.num_rows}/{total} mono rows of {seconds[0]:g}-{seconds[1]:g} s)")


def build_libriheavy(root: Path, target_hours: float, workers: int) -> None:
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(LIBRIHEAVY, repo_type="dataset") if f.endswith(".parquet"))
    subset = lambda f: f.split("/")[1]
    base = [f for f in files if subset(f) in ("dev", "small", "medium")]
    large = [f for f in files if subset(f) == "large"]
    random.Random(0).shuffle(large)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda f: convert_lh(root, f, CLIP_SECONDS), base))
        while (have := split_hours(root, "train", TOP_UP_SECONDS)) < target_hours:
            todo = [f for f in large if not lh_output(root, f).exists()]
            if not todo:
                raise RuntimeError(f"Libriheavy large exhausted at {have:.1f} train hours of 5-15 s clips")
            batch = todo[: max(1, int((target_hours - have) / LARGE_FILE_HOURS))]
            log(f"train {have:.1f}/{target_hours:.0f} h of 5-15 s clips; adding {len(batch)} large files")
            list(ex.map(lambda f: convert_lh(root, f, TOP_UP_SECONDS), batch))
    log(
        f"Libriheavy done: train {split_hours(root, 'train'):.1f} h, "
        f"{split_hours(root, 'train', TOP_UP_SECONDS):.1f} h of it in 5-15 s clips"
    )


# ------------------------------------------------------------------ dataset card

CARD = """---
language:
- en
license: other
task_categories:
- automatic-speech-recognition
- text-to-speech
pretty_name: English read speech (Common Voice 27 + Libriheavy)
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---

# en_speech

Pooled English read speech with cased, punctuated transcripts, prepared for Mimi
(24 kHz) tokenization: {{TRAIN_HOURS}} h train ({{TRAIN_TOP_UP_HOURS}} h of 5-15 s clips),
{{VALIDATION_HOURS}} h validation. Built by `training/prep/build_en_speech.py` in the
hibiki-zero repository.

| split | source | shards | rows | hours | 5-15 s hours | speakers | GB |
|---|---|---:|---:|---:|---:|---:|---:|
{{STATS}}

## Columns

| column | type | content |
|---|---|---|
| `id` | string | `cv27/<clip>` or `libriheavy/<Libri-Light path>/<speaker>/<book>/<segment>` |
| `audio` | struct{bytes, path} | original codec bytes; the `path` extension names the codec |
| `text` | string | cased, punctuated transcript |
| `speaker` | string | `cv27/<client_id>` or `libriheavy/<speaker_id>` |
| `duration` | float32 | seconds |

## Audio

Mono, in each source's original codec, never transcoded:

- Common Voice: MP3, 48 kHz, ~64 kbps; decodes ~4,000x realtime per core (libsndfile).
- Libriheavy: Ogg Opus, 48 kHz, ~70 kbps; decodes ~340x realtime per core.

Both sources are already lossy. A lossless 24 kHz FLAC copy measured ~3x the Opus
size (229 vs 73 kbps) for no information gain, and a smaller lossy re-encode would
add generation loss. Decode and resample to 24 kHz when feeding Mimi:

```python
import io

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF

for batch in pq.ParquetFile("train/cv27-00000.parquet").iter_batches(batch_size=64):
    for row in batch.to_pylist():
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        wav = AF.resample(torch.from_numpy(wav), sr, 24_000)  # mono at Mimi's rate
```

With Hugging Face `datasets` (needs `torchcodec`):
`load_dataset("/Volumes/data/datasets/en_speech", streaming=True)` and
`.cast_column("audio", Audio(sampling_rate=24_000))`.

## Selection

- Common Voice Scripted Speech 27.0 English: validated clips only. dev is the
  validation split; train is every other validated clip whose speaker and sentence
  are not in dev. The official test clips are folded into train, so CV test is not
  a clean benchmark for models trained here. Invalidated, unvalidated and reported
  clips, the sentence corpora, and the vote and demographic columns are not kept.
- Libriheavy (`mythicinfinity/libriheavy`, LibriVox audio taken from the best
  available source): mono rows only (the ~10% stereo Opus rows are dropped rather
  than re-encoded). dev is the validation split; train is small + medium plus
  seeded-random large files until the pooled train split holds the target hours
  of 5-15 s clips. Large files keep only their 5-15 s clips, except the first
  194, which were written before that rule and keep every clip of 5-22.4 s.
  `text` is the book text (`text_original`); the upper-case normalized copy, book
  ids and test sets are not kept.
- Both sources keep only clips of 5-22.4 s; 22.4 s is 280 Mimi frames at
  12.5 Hz, the training row cap.

## Licenses

- Common Voice: CC0-1.0 under the Mozilla Data Collective terms, which forbid
  re-hosting or re-sharing the data and attempting to identify speakers. Keep this
  corpus private.
- Libriheavy: Apache-2.0; the LibriVox recordings are public domain.
"""


def write_card(root: Path) -> None:
    rows = []
    for split in ("train", "validation"):
        for source in ("cv27", "libriheavy"):
            shards = sorted((root / split).glob(f"{source}-*.parquet"))
            if not shards:
                continue
            t = pa.concat_tables(pq.read_table(p, columns=["speaker", "duration"]) for p in shards)
            duration = t["duration"]
            rows.append(
                (
                    split,
                    source,
                    len(shards),
                    t.num_rows,
                    pc.sum(duration).as_py() / 3600,
                    pc.sum(duration.filter(in_range(duration, TOP_UP_SECONDS))).as_py() / 3600,
                    len(pc.unique(t["speaker"])),
                    sum(p.stat().st_size for p in shards) / 1e9,
                )
            )
    table = "\n".join(
        f"| {s} | {src} | {n} | {r:,} | {h:,.1f} | {h_top:,.1f} | {spk:,} | {gb:.1f} |"
        for s, src, n, r, h, h_top, spk, gb in rows
    )
    totals = {s: sum(r[4] for r in rows if r[0] == s) for s in ("train", "validation")}
    card = (
        CARD.replace("{{STATS}}", table)
        .replace("{{TRAIN_HOURS}}", f"{totals['train']:,.0f}")
        .replace("{{TRAIN_TOP_UP_HOURS}}", f"{sum(r[5] for r in rows if r[0] == 'train'):,.0f}")
        .replace("{{VALIDATION_HOURS}}", f"{totals['validation']:,.0f}")
    )
    (root / "README.md").write_text(card)
    log(f"wrote {root / 'README.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("cv", "libriheavy", "card"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    # With the PhoMT cache's ~0.8k hours, training gets ~10k hours of 5-15 s clips.
    p.add_argument("--target-hours", type=float, default=9_200, help="pooled train hours of 5-15 s clips (libriheavy step)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.step == "cv":
        build_cv(args.root, args.workers)
    elif args.step == "libriheavy":
        build_libriheavy(args.root, args.target_hours, args.workers)
    else:
        write_card(args.root)


if __name__ == "__main__":
    sys.exit(main())
