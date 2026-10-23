#!/usr/bin/env python
"""Build a pooled English speech+text corpus for Mimi training.

Sources, both read speech with cased, punctuated transcripts:
  - Common Voice Scripted Speech 27.0 English (Mozilla Data Collective):
    validated clips. dev becomes validation and train is every other validated
    clip (the official test included) whose speaker and sentence are not in dev;
    invalidated and unvalidated clips are dropped.
  - Libriheavy (mythicinfinity/libriheavy): LibriVox audiobooks aligned to the
    book text, mono rows only. dev becomes validation; train is medium + small
    plus seeded-random large files, keeping only their 5-15 s clips, until the
    pooled train split holds --target-hours of 5-15 s clips. A train file whose
    footer shows only capped speakers gets an empty shard without downloading.

Every source keeps only clips of 5-22.4 s; longer clips can never fit the trainer's
280-frame rows. Every recording (a CV clip, a Libriheavy chapter within one file)
must reach MIN_BANDWIDTH_HZ, and train keeps each speaker's clips, in build order,
only until the speaker holds SPEAKER_HOURS.

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
from collections import Counter, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
import soundfile as sf
from scipy.signal import welch

from training.paths import REPO_ROOT

MDC_DATASET = "cmu5jplf300nwmh07iqvk9leo"
LIBRIHEAVY = "mythicinfinity/libriheavy"
CLIP_SECONDS = (5.0, 280 / 12.5)  # kept from every source; longer clips never fit a 280-frame row
TOP_UP_SECONDS = (5.0, 15.0)  # kept from large files, and counted by --target-hours
# Mean TOP_UP_SECONDS hours a downloaded large file adds after the filters and the speaker cap;
# sizes top-up batches.
LARGE_FILE_HOURS = 3.7
# Mimi codes audio up to 12 kHz. Just below the ~11 kHz lowpass of Common Voice's 32 kHz MP3s,
# which pass whole.
MIN_BANDWIDTH_HZ = 10_500
BANDWIDTH_SECONDS = 30  # audio per recording the bandwidth is estimated from, as HiFiTTS-2 does
# Train hours kept per speaker, so a few prolific readers can't dominate. Libriheavy is too
# concentrated in a few readers for a lower cap to reach the target hours.
SPEAKER_HOURS = 30.0
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


def write_shard(path: Path, table: pa.Table) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, row_group_size=500, compression=COMPRESSION)
    tmp.replace(path)


def span(seconds: tuple[float, float]) -> str:
    return f"{seconds[0]:g}-{seconds[1]:g} s"


def in_range(duration: pa.ChunkedArray, seconds: tuple[float, float]) -> pa.ChunkedArray:
    return pc.and_(pc.greater_equal(duration, seconds[0]), pc.less_equal(duration, seconds[1]))


def bandwidth(clips: Iterable[bytes]) -> float:
    """HiFiTTS-2's bandwidth estimate: the highest frequency whose mean power is within 50 dB
    of the spectral peak, over the first BANDWIDTH_SECONDS of one recording's clips."""
    audio, rate = [], 0
    for data in clips:
        a, rate = sf.read(io.BytesIO(data), dtype="float32")
        audio.append(a)
        if sum(map(len, audio)) >= BANDWIDTH_SECONDS * rate:
            break
    f, power = welch(np.concatenate(audio), rate, nperseg=4096)
    return float(f[np.flatnonzero(power >= power.max() * 1e-5)[-1]])


def split_hours(root: Path, split: str, seconds: tuple[float, float] = CLIP_SECONDS) -> float:
    total = 0.0
    for p in (root / split).glob("*.parquet"):
        duration = pq.read_table(p, columns=["duration"])["duration"]
        total += pc.sum(duration.filter(in_range(duration, seconds))).as_py() or 0.0
    return total / 3600


def train_seconds(root: Path) -> Counter:
    """Seconds of train audio per speaker."""
    totals = Counter()
    for p in (root / "train").glob("*.parquet"):
        t = pq.read_table(p, columns=["speaker", "duration"]).group_by("speaker").aggregate([("duration", "sum")])
        totals.update(dict(zip(t["speaker"].to_pylist(), t["duration_sum"].to_pylist())))
    return totals


def keep_speakers(t: pa.Table, totals: Counter) -> pa.Table:
    """Keep each speaker's train rows, in order, until the speaker holds SPEAKER_HOURS."""
    keep = []
    for speaker, seconds in zip(t["speaker"].to_pylist(), t["duration"].to_pylist()):
        keep.append(totals[speaker] < SPEAKER_HOURS * 3600)
        if keep[-1]:
            totals[speaker] += seconds
    return t.filter(pa.array(keep, pa.bool_()))


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
    totals = Counter()  # train seconds per speaker

    def flush(split: str) -> None:
        t = pa.Table.from_pydict(buffers[split], schema=SCHEMA)
        if split == "train":
            t = keep_speakers(t, totals)
        if t.num_rows:
            write_shard(root / split / f"cv27-{counts[split]:05d}.parquet", t)
            counts[split] += 1
        buffers[split] = {k: [] for k in FEATURES}

    found = 0
    with tarfile.open(archive, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or name not in keep:
                continue
            found += 1
            data = tar.extractfile(m).read()
            if bandwidth([data]) < MIN_BANDWIDTH_HZ:
                continue
            split, r, duration = keep[name]
            buf = buffers[split]
            buf["id"].append("cv27/" + Path(name).stem)
            buf["audio"].append({"bytes": data, "path": name})
            buf["text"].append(r["sentence"].strip())
            buf["speaker"].append("cv27/" + r["client_id"])
            buf["duration"].append(duration)
            if len(buf["id"]) >= SHARD_ROWS:
                flush(split)
                log(f"CV {found}/{len(keep)} clips read")
    for split in buffers:
        flush(split)
    if found != len(keep):
        raise RuntimeError(f"CV archive had {found} of {len(keep)} selected clips")
    archive.unlink()
    raw.rmdir()
    log(f"CV done: {found} clips read, train {split_hours(root, 'train'):.1f} h total so far")


# ------------------------------------------------------------------ Libriheavy


def lh_output(root: Path, src: str) -> Path:
    # default/medium/medium-00012-of-00287.parquet -> train/libriheavy-medium-00012.parquet
    # default/large/shard-00003-of-0020/large-00045-of-00147.parquet -> train/libriheavy-large-00003-00045.parquet
    parts = src.split("/")
    subset = parts[1]
    split = "validation" if subset == "dev" else "train"
    index = "-".join(p.split("-")[1] for p in parts[2:])
    return root / split / f"libriheavy-{subset}-{index}.parquet"


def lh_capped(src: str, totals: Counter) -> bool:
    """True when the file's footer statistics show only speakers that already hold SPEAKER_HOURS."""
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{LIBRIHEAVY}/{src}", block_size=1 << 16, cache_type="readahead") as f:
        md = pq.ParquetFile(f).metadata
    col = next(i for i in range(md.num_columns) if md.schema.column(i).path == "speaker_id")
    stats = [md.row_group(g).column(col).statistics for g in range(md.num_row_groups)]
    return all(
        s is not None and s.has_min_max and s.min == s.max and totals[f"libriheavy/{s.min}"] >= SPEAKER_HOURS * 3600
        for s in stats
    )


def read_lh(src: str, seconds: tuple[float, float]) -> pa.Table:
    """The mono clips of one Libriheavy file that last within `seconds` and whose chapter
    recording reaches MIN_BANDWIDTH_HZ."""
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory() as tmp:
        local = hf_hub_download(LIBRIHEAVY, src, repo_type="dataset", local_dir=tmp)
        t = pq.read_table(local, columns=["audio", "id", "speaker_id", "text_original", "audio_duration"])
    # Keep mono rows only (~10% are stereo); the channel count is byte 37 of the Ogg Opus header.
    data = pc.struct_field(t["audio"], "bytes")
    if not pc.all(pc.equal(pc.binary_slice(data, 28, 36), pa.scalar(b"OpusHead"))).as_py():
        raise RuntimeError(f"{src}: audio is not single-stream Ogg Opus")
    total = t.num_rows
    mono = pc.equal(pc.binary_slice(data, 37, 38), pa.scalar(b"\x01"))
    t = t.filter(pc.and_(mono, in_range(t["audio_duration"], seconds)))
    kept = t.num_rows
    # A clip id is its chapter recording's id plus `_<segment>`.
    recordings = [i.rsplit("_", 1)[0] for i in t["id"].to_pylist()]
    clips: dict[str, list] = {}
    for rec, clip in zip(recordings, pc.struct_field(t["audio"], "bytes")):
        clips.setdefault(rec, []).append(clip)
    wide = {rec for rec, c in clips.items() if bandwidth(x.as_py() for x in c) >= MIN_BANDWIDTH_HZ}
    t = t.filter(pa.array([rec in wide for rec in recordings], pa.bool_()))
    log(f"read {src}: {total} rows, {kept} mono of {span(seconds)}, {t.num_rows} reaching {MIN_BANDWIDTH_HZ / 1000:g} kHz")
    columns = {
        "id": pc.binary_join_element_wise("libriheavy/", t["id"], ""),
        "audio": t["audio"],
        "text": t["text_original"],
        "speaker": pc.binary_join_element_wise("libriheavy/", t["speaker_id"], ""),
        "duration": t["audio_duration"].cast(pa.float32()),
    }
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def build_libriheavy(root: Path, target_hours: float, workers: int) -> None:
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(LIBRIHEAVY, repo_type="dataset") if f.endswith(".parquet"))
    subset = lambda f: f.split("/")[1]
    base = [f for f in files if subset(f) in ("dev", "small", "medium")]
    large = [f for f in files if subset(f) == "large"]
    random.Random(0).shuffle(large)
    top_up = span(TOP_UP_SECONDS)
    totals = train_seconds(root)
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True, exist_ok=True)

    def read(src: str, seconds: tuple[float, float]) -> pa.Table:
        if lh_output(root, src).parent.name == "train" and lh_capped(src, totals):
            log(f"skipped {src}: its footer shows only capped speakers")
            return SCHEMA.empty_table()
        return read_lh(src, seconds)

    def convert(srcs: list[str], seconds: tuple[float, float]) -> None:
        """Write the files' shards in order, so the speaker cap is applied in build order."""
        pending = deque()

        def finish() -> None:
            src, future = pending.popleft()
            t, dst = future.result(), lh_output(root, src)
            note = f"{t.num_rows} rows"
            if dst.parent.name == "train":
                t = keep_speakers(t, totals)
                note = f"{t.num_rows}/{note} under the speaker cap"
            write_shard(dst, t)
            log(f"wrote {dst.relative_to(root)} ({note})")

        for src in srcs:
            if not lh_output(root, src).exists():
                pending.append((src, ex.submit(read, src, seconds)))
                if len(pending) > workers:
                    finish()
        while pending:
            finish()

    with ThreadPoolExecutor(workers) as ex:
        convert(base, CLIP_SECONDS)
        while (have := split_hours(root, "train", TOP_UP_SECONDS)) < target_hours:
            todo = [f for f in large if not lh_output(root, f).exists()]
            if not todo:
                raise RuntimeError(f"Libriheavy large exhausted at {have:.1f} train hours of {top_up} clips")
            batch = todo[: max(1, int((target_hours - have) / LARGE_FILE_HOURS))]
            log(f"train {have:.1f}/{target_hours:.0f} h of {top_up} clips; trying {len(batch)} large files")
            convert(batch, TOP_UP_SECONDS)
    log(
        f"Libriheavy done: train {split_hours(root, 'train'):.1f} h, "
        f"{split_hours(root, 'train', TOP_UP_SECONDS):.1f} h of it in {top_up} clips"
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
(24 kHz) tokenization: {{TRAIN_HOURS}} h train ({{TOP_UP_RANGE_HOURS}} h of {{TOP_UP}} clips),
{{VALIDATION_HOURS}} h validation. Built by `training/prep/build_en_speech.py` in the
hibiki-zero repository.

| split | source | shards | rows | hours | {{TOP_UP}} hours | speakers | GB |
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

- Common Voice: MP3, 32 or 48 kHz, ~64 kbps; decodes ~4,000x realtime per core (libsndfile).
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
  than re-encoded). dev is the validation split; train is medium + small plus
  seeded-random large files, keeping only their {{TOP_UP}} clips, until the pooled
  train split holds the target hours of {{TOP_UP}} clips.
  `text` is the book text (`text_original`); the upper-case normalized copy, book
  ids and test sets are not kept.
- Both sources keep only clips of {{CLIP}}; longer clips can never fit the
  280-frame (12.5 Hz) training rows.
- Every recording reaches {{BANDWIDTH}} of audio bandwidth: the highest frequency whose
  mean power is within 50 dB of the spectral peak (HiFiTTS-2's estimate), over the
  first {{BANDWIDTH_SECONDS}} s of a Libriheavy chapter's clips in one source file, or a whole
  Common Voice clip.
- Train keeps each speaker's clips, in build order, until the speaker holds
  {{SPEAKER_HOURS}} h, so a few prolific readers don't dominate.

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
        .replace("{{TOP_UP_RANGE_HOURS}}", f"{sum(r[5] for r in rows if r[0] == 'train'):,.0f}")
        .replace("{{TOP_UP}}", span(TOP_UP_SECONDS))
        .replace("{{CLIP}}", span(CLIP_SECONDS))
        .replace("{{BANDWIDTH}}", f"{MIN_BANDWIDTH_HZ / 1000:g} kHz")
        .replace("{{BANDWIDTH_SECONDS}}", f"{BANDWIDTH_SECONDS}")
        .replace("{{SPEAKER_HOURS}}", f"{SPEAKER_HOURS:g}")
        .replace("{{VALIDATION_HOURS}}", f"{totals['validation']:,.0f}")
    )
    (root / "README.md").write_text(card)
    log(f"wrote {root / 'README.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("cv", "libriheavy", "card"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    # With the PhoMT cache's ~0.8k hours, training gets ~10k hours of 5-15 s clips.
    p.add_argument(
        "--target-hours", type=float, default=9_200, help="pooled train hours of 5-15 s clips (libriheavy step)"
    )
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
