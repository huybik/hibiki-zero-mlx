#!/usr/bin/env python
"""Publish the EN-VI speech dataset that SFT pair rows are cached from.

Each row holds an English clip with its transcript and Vietnamese translation, and the
Vietnamese clip where one exists:
  - en_speech (training/prep/build_en_speech.py) with the Vietnamese text exported by
    training/prep/translate_en_speech.py. `vi_audio` stays empty until the voice-transfer
    TTS step fills it. Rows without a translation (transcripts without letters) are left out.
  - PhoMT-en-vi-speech: TTS speech on both sides. Its rows are those of the grounded-v2
    cache in CACHE_REPO, which passed English CTC alignment, whose English clip lasts within
    en_speech's CLIP_SECONDS. Ids are the cache's, so cache rows map back. The PCM WAV is
    stored as lossless FLAC, about 40% of its size.

Run from the repo root as `python -m training.prep.publish_en_vi_speech <step>` with HF_TOKEN
exported. Steps:
  en_speech  one shard per non-empty en_speech shard, under the same name
  phomt      one shard per PhoMT source file
  card       README.md with statistics read back from the repo
Shards are written under --staging and committed to --repo COMMIT_FILES at a time from a
background thread, so building (PhoMT downloads) and uploading overlap, then deleted locally.
Shards already in the repo are skipped, so rerun to resume.

Every shard has one schema:
  data/{train,validation}/{cv27,libriheavy-*,phomt-*}.parquet
  id, en_audio{bytes,path}, text_en, text_vi, speaker, en_duration_s,
  vi_audio{bytes,path}, vi_duration_s, cross_lingual_timbre_matched
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem, hf_hub_download

from training.prep.build_en_speech import CLIP_SECONDS, in_range, log, span
from training.prep.translate_en_speech import exclusive

REPO = "huybik/en-vi-speech"
CACHE_REPO = "huybik/hibiki-zero-vi-full-sft"
PAIRS = [f"grounded-v2/pairs_w{w}.jsonl" for w in range(5)]
PHOMT = "anquachdev/PhoMT-en-vi-speech"
PHOMT_REVISION = "33400f73dde07da539e8326313cbabe20b757740"  # the revision grounded-v2 was built from
COMMIT_FILES = 25  # shards per commit: ~3-7 GB, so ~130 commits in all
# Batches staged while a commit uploads, so PhoMT downloads keep going: at most ~70 GB for
# en_speech and ~30 GB for PhoMT on disk.
QUEUED_COMMITS = 8
RETRIES = 5  # a failed build or commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
T = TypeVar("T")

AUDIO = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
FEATURES = {
    "id": {"dtype": "string", "_type": "Value"},
    "en_audio": {"_type": "Audio"},
    "text_en": {"dtype": "string", "_type": "Value"},
    "text_vi": {"dtype": "string", "_type": "Value"},
    "speaker": {"dtype": "string", "_type": "Value"},
    "en_duration_s": {"dtype": "float32", "_type": "Value"},
    "vi_audio": {"_type": "Audio"},
    "vi_duration_s": {"dtype": "float32", "_type": "Value"},
    "cross_lingual_timbre_matched": {"dtype": "bool", "_type": "Value"},
}
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("en_audio", AUDIO),
        ("text_en", pa.string()),
        ("text_vi", pa.string()),
        ("speaker", pa.string()),
        ("en_duration_s", pa.float32()),
        ("vi_audio", AUDIO),
        ("vi_duration_s", pa.float32()),
        ("cross_lingual_timbre_matched", pa.bool_()),
    ],
    metadata={"huggingface": json.dumps({"info": {"features": FEATURES}})},
)
# Audio bytes are already compressed; zstd only the other columns.
COMPRESSION = {c: "none" if c.endswith(".bytes") else "zstd" for c in SCHEMA.empty_table().flatten().column_names}

Task = tuple[str, Callable[[], pa.Table]]  # repo path, builder of its rows


def retried(what: str, fn: Callable[[], T]) -> T:
    """fn(), retried with backoff: downloads and Hub commits fail mostly transiently. A permanent
    error (auth, quota) surfaces after the last retry."""
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRIES:
                raise
            wait = RETRY_WAIT * 2**attempt
            log(f"{what} failed ({type(e).__name__}: {e}); retrying in {wait // 60} min")
            time.sleep(wait)


def publish(api: HfApi, repo: str, tasks: Iterator[Task], staging: Path, workers: int) -> None:
    """Build the shards the repo lacks, `workers` at a time and in order, and commit them
    COMMIT_FILES at a time from one background thread, up to QUEUED_COMMITS batches behind."""
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    present = set(api.list_repo_files(repo, repo_type="dataset"))

    def build(name: str, rows: Callable[[], pa.Table]) -> Path:
        path = staging / name.replace("/", "__")
        pq.write_table(retried(f"building {name}", rows), path, row_group_size=500, compression=COMPRESSION)
        return path

    def commit(batch: list[tuple[str, Path]]) -> None:
        ops = [CommitOperationAdd(name, path) for name, path in batch]
        message = f"Add {len(batch)} shards: {batch[0][0]} … {batch[-1][0]}"
        retried(f"commit of {batch[0][0]} …", lambda: api.create_commit(repo, ops, repo_type="dataset", commit_message=message))
        gb = sum(p.stat().st_size for _, p in batch) / 1e9
        for _, path in batch:
            path.unlink()
        log(f"committed {message} ({gb:.1f} GB)")

    batch: list[tuple[str, Path]] = []
    pending: deque = deque()
    commits: deque = deque()

    def queue_commit() -> None:
        commits.append(committer.submit(commit, batch.copy()))
        batch.clear()
        # Raise a failed commit at once; wait while too many batches are staged.
        while commits and (commits[0].done() or len(commits) > QUEUED_COMMITS):
            commits.popleft().result()

    def finish() -> None:
        name, future = pending.popleft()
        batch.append((name, future.result()))
        if len(batch) == COMMIT_FILES:
            queue_commit()

    staging.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(staging.with_suffix(".lock"), f"another run is publishing from {staging}"), \
            ThreadPoolExecutor(workers) as builder, ThreadPoolExecutor(1) as committer:
        # Staged shards are never reused: a killed run may leave partial or uncommitted ones.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            for name, rows in tasks:
                if name not in present:
                    pending.append((name, builder.submit(build, name, rows)))
                    if len(pending) > workers:
                        finish()
            while pending:
                finish()
            if batch:
                queue_commit()
            while commits:
                commits.popleft().result()
        except BaseException:
            # Don't wait out the retries of batches still queued behind a failure.
            builder.shutdown(cancel_futures=True)
            committer.shutdown(cancel_futures=True)
            raise


# ------------------------------------------------------------------ en_speech


def en_speech_tasks(src: Path, vi_text: Path) -> Iterator[Task]:
    for split in ("train", "validation"):
        export = pq.read_table(vi_text / f"{split}.parquet")
        vi = dict(zip(export["id"].to_pylist(), export["text_vi"].to_pylist()))
        shards = [p for p in sorted((src / split).glob("*.parquet")) if pq.read_metadata(p).num_rows]
        ids = set()
        for p in shards:
            ids.update(pq.read_table(p, columns=["id"])["id"].to_pylist())
        if stray := len(vi.keys() - ids):
            raise RuntimeError(f"{split}: {stray:,} translated ids are not in {src}; rerun translate_en_speech texts and export")
        log(f"{split}: {len(shards)} shards, {len(ids):,} rows, {len(ids) - len(vi):,} without a translation")
        for p in shards:
            yield f"data/{split}/{p.name}", lambda p=p, vi=vi: read_en_speech(p, vi)


def read_en_speech(path: Path, vi: dict[str, str]) -> pa.Table:
    t = pq.read_table(path)
    text_vi = pa.array([vi.get(i) for i in t["id"].to_pylist()], pa.string())
    keep = pc.is_valid(text_vi)
    t, text_vi = t.filter(keep), text_vi.filter(keep)
    n = t.num_rows
    columns = {
        "id": t["id"],
        "en_audio": t["audio"],
        "text_en": t["text"],
        "text_vi": text_vi,
        "speaker": t["speaker"],
        "en_duration_s": t["duration"],
        "vi_audio": pa.nulls(n, AUDIO),
        "vi_duration_s": pa.nulls(n, pa.float32()),
        "cross_lingual_timbre_matched": pa.nulls(n, pa.bool_()),
    }
    return pa.Table.from_pydict(columns, schema=SCHEMA)


# ---------------------------------------------------------------------- PhoMT


def phomt_tasks(staging: Path) -> Iterator[Task]:
    """One task per PhoMT source file, in cache id order: phomt_s00304r00012 is row 12 of the
    file the cache numbered 304."""
    picks: dict[str, list[tuple[int, str, bool]]] = {}
    for name in PAIRS:
        with open(hf_hub_download(CACHE_REPO, name, repo_type="dataset")) as f:
            for line in f:
                p = json.loads(line)
                src, row = p["en_audio"].removeprefix(f"hf://{PHOMT}/").split("#r")
                picks.setdefault(src, []).append((int(row), p["id"], p["cross_lingual_timbre_matched"]))
    log(f"PhoMT: {sum(map(len, picks.values())):,} cache rows in {len(picks)} source files")
    for src, rows in sorted(picks.items(), key=lambda kv: kv[1][0][1]):
        number = rows[0][1].removeprefix("phomt_").split("r")[0]
        yield f"data/train/phomt-{number}.parquet", lambda src=src, rows=rows: read_phomt(src, rows, staging)


def read_phomt(src: str, rows: list[tuple[int, str, bool]], staging: Path) -> pa.Table:
    with tempfile.TemporaryDirectory(dir=staging) as tmp:
        local = hf_hub_download(PHOMT, src, repo_type="dataset", revision=PHOMT_REVISION, local_dir=tmp)
        t = pq.read_table(local).take(pa.array([r for r, _, _ in rows]))
    keep = in_range(t["duration_en_s"], CLIP_SECONDS)
    t = t.append_column("id", pa.array([i for _, i, _ in rows])).append_column(
        "timbre", pa.array([m for _, _, m in rows], pa.bool_())
    ).filter(keep)
    log(f"read {src}: {len(rows)} cache rows, {t.num_rows} of {span(CLIP_SECONDS)}")
    columns = {
        "id": t["id"],
        "en_audio": flac(t["audio_en"]),
        "text_en": t["en"],
        "text_vi": t["vi"],
        "speaker": pa.nulls(t.num_rows, pa.string()),
        "en_duration_s": t["duration_en_s"].cast(pa.float32()),
        "vi_audio": flac(t["audio_vi"]),
        "vi_duration_s": t["duration_vi_s"].cast(pa.float32()),
        "cross_lingual_timbre_matched": t["timbre"],
    }
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def flac(audio: pa.ChunkedArray) -> pa.Array:
    """Re-encode mono 16-bit PCM WAV clips as FLAC, sample for sample."""
    out = []
    for clip in audio.to_pylist():
        with sf.SoundFile(io.BytesIO(clip["bytes"])) as f:
            if f.format != "WAV" or f.subtype != "PCM_16" or f.channels != 1:
                raise RuntimeError(f"{clip['path']}: {f.format} {f.subtype} {f.channels} ch, not mono PCM_16 WAV")
            samples, rate = f.read(dtype="int16"), f.samplerate
        encoded = io.BytesIO()
        sf.write(encoded, samples, rate, format="FLAC", subtype="PCM_16")
        out.append({"bytes": encoded.getvalue(), "path": clip["path"].removesuffix(".wav") + ".flac"})
    return pa.array(out, AUDIO)


# ------------------------------------------------------------------ dataset card

CARD = """---
language:
- en
- vi
license: other
license_name: mixed-research-only
task_categories:
- translation
- text-to-speech
- automatic-speech-recognition
pretty_name: EN-VI speech for Hibiki-Zero
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.parquet
  - split: validation
    path: data/validation/*.parquet
---

# en-vi-speech

English speech with English transcripts and Vietnamese translations, and Vietnamese speech
where it exists, for research on Vietnamese→English simultaneous speech translation
(Hibiki-Zero). Built by `training/prep/publish_en_vi_speech.py` in the hibiki-zero repository.
The Mimi-code training caches built from it are in
[huybik/hibiki-zero-vi-full-sft](https://huggingface.co/datasets/huybik/hibiki-zero-vi-full-sft).

| split | source | shards | rows | EN hours | VI audio hours | speakers | GB |
|---|---|---:|---:|---:|---:|---:|---:|
{{STATS}}

## Columns

| column | type | content |
|---|---|---|
| `id` | string | `cv27/<clip>`, `libriheavy/<subset>/<speaker>/<book>/<recording>_<segment>` or `phomt_s<file>r<row>` |
| `en_audio` | Audio | English clip; the `path` extension names the codec |
| `text_en` | string | cased, punctuated English transcript |
| `text_vi` | string | Vietnamese translation |
| `speaker` | string | `cv27/<client_id>` or `libriheavy/<speaker_id>`; empty for PhoMT |
| `en_duration_s` | float32 | seconds |
| `vi_audio` | Audio | Vietnamese clip; empty for Common Voice and Libriheavy until voice-transfer TTS fills it |
| `vi_duration_s` | float32 | seconds, empty with `vi_audio` |
| `cross_lingual_timbre_matched` | bool | PhoMT: whether the English and Vietnamese TTS voices match; empty elsewhere |

## Sources

Every clip lasts {{CLIP}} in English, so most fit the 280-frame (12.5 Hz) training rows.

- Common Voice 27.0 English and Libriheavy: the `en_speech` corpus, read speech whose
  recordings reach 10.5 kHz of audio bandwidth, at most 30 h per train speaker. Common Voice
  dev and Libriheavy dev are the validation split. Audio is each source's original codec
  (MP3, 48 kHz Ogg Opus). The English text was translated by Claude Sonnet 5.5, Gemini 3.8
  Flash and Muse Spark 1.3, in batches of neighbouring lines.
- PhoMT (`anquachdev/PhoMT-en-vi-speech` at `33400f7`): synthetic speech of PhoMT sentence pairs on both
  sides, limited to rows whose English passed CTC alignment. Its 24 kHz English and 48 kHz
  Vietnamese 16-bit WAV is stored as FLAC, sample for sample. All rows are train.

The audio columns hold codec bytes. Read raw rows with `pyarrow` and decode with `soundfile`
(`sf.read(io.BytesIO(row["en_audio"]["bytes"]))`), resampling to 24 kHz for Mimi, or with
`datasets` (needs `torchcodec`):
`load_dataset("huybik/en-vi-speech", streaming=True).cast_column("en_audio", Audio(sampling_rate=24_000))`.

## Use

For research only. Common Voice is CC0-1.0, distributed by the Mozilla Data Collective, whose
terms forbid re-hosting it and trying to identify its speakers. Libriheavy is Apache-2.0 over
public-domain LibriVox recordings. PhoMT-en-vi-speech is Apache-2.0 on its card, over the PhoMT
corpus.
"""


def write_card(api: HfApi, repo: str, staging: Path) -> None:
    fs = HfFileSystem()
    rows = []
    for split in ("train", "validation"):
        files = {f["name"]: f["size"] for f in fs.ls(f"datasets/{repo}/data/{split}", detail=True)}
        for source in ("cv27", "libriheavy", "phomt"):
            shards = sorted(f for f in files if f.rsplit("/", 1)[1].startswith(f"{source}-"))
            if not shards:
                continue
            t = pq.read_table(shards, filesystem=fs, columns=["speaker", "en_duration_s", "vi_duration_s"])
            rows.append(
                (
                    split,
                    source,
                    len(shards),
                    t.num_rows,
                    pc.sum(t["en_duration_s"]).as_py() / 3600,
                    (pc.sum(t["vi_duration_s"]).as_py() or 0) / 3600,
                    len(pc.unique(t["speaker"].drop_null())),
                    sum(files[f] for f in shards) / 1e9,
                )
            )
    table = "\n".join(
        f"| {s} | {src} | {n} | {r:,} | {h:,.1f} | {vh:,.1f} | {spk:,} | {gb:.1f} |"
        for s, src, n, r, h, vh, spk, gb in rows
    )
    path = staging / "README.md"
    staging.mkdir(parents=True, exist_ok=True)
    path.write_text(CARD.replace("{{STATS}}", table).replace("{{CLIP}}", span(CLIP_SECONDS)))
    api.upload_file(path_or_fileobj=path, path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="Update the dataset card")
    log(f"uploaded README.md:\n{table}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("en_speech", "phomt", "card"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    p.add_argument("--vi-text", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_text"))
    p.add_argument("--staging", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech_upload"))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    api = HfApi()
    if args.step == "en_speech":
        publish(api, args.repo, en_speech_tasks(args.src, args.vi_text), args.staging / "en_speech", args.workers)
    elif args.step == "phomt":
        publish(api, args.repo, phomt_tasks(args.staging / "phomt"), args.staging / "phomt", args.workers)
    else:
        write_card(api, args.repo, args.staging)


if __name__ == "__main__":
    sys.exit(main())
