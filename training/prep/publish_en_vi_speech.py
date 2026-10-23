#!/usr/bin/env python
"""Build the EN-VI speech dataset that SFT pair rows are cached from, and publish it.

Each row holds an English clip of the en_speech corpus (Common Voice 27 and Libriheavy read
speech) with its transcript and the Vietnamese translation exported by
training/prep/translate_en_speech.py. `vi_audio` stays empty until the voice-transfer TTS step
fills it. Rows without a translation (transcripts without letters) are dropped.

--root, laid out as the Hugging Face repo, is the one local copy: the en_speech step moves the
corpus into it. Run from the repo root as `python -m training.prep.publish_en_vi_speech <step>`.
Steps:
  en_speech  converts each shard of the en_speech corpus at --src into a dataset shard of the
             same name and deletes the original, then the emptied corpus folder
  card       README.md with statistics of the local dataset
  upload     commits the shards missing from the Hub, a few GB at a time, then the card
             (export HF_TOKEN); rerun it after adding shards
Every step may be stopped (Ctrl-C, kill) and resumes when run again; a lock stops a second run of
the same step.

Every shard has one schema:
  data/{train,validation}/{cv27,libriheavy-*}.parquet
  id, en_audio{bytes,path}, text_en, text_vi, speaker, en_duration_s,
  vi_audio{bytes,path}, vi_duration_s, cross_lingual_timbre_matched
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

# pyarrow's default pool on macOS (mimalloc) keeps freed shard buffers: 13 GB of footprint after
# converting 24 shards with nothing allocated. The system allocator returns them. Arrow reads
# this once, so it must be set before pyarrow is imported.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, RepoFile  # noqa: E402

from training.prep.translate_en_speech import LETTER, MAX_SECONDS, exclusive, log  # noqa: E402

REPO = "huybik/en-vi-speech"
CLIP_SECONDS = (5.0, MAX_SECONDS)  # en_speech's clip range; longer clips never fit a 280-frame row
SPLITS = ("train", "validation")
RETRIES = 5  # a failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
# Up to ~3 min of upload, all a stop loses; at the link's ~37 MB/s, ~30 commits an hour, well
# under the Hub's commit quota (~100 an hour).
COMMIT_BYTES = 4_000_000_000
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


def durable(path: Path) -> None:
    """Flush a file or directory to the drive itself; on macOS fsync stops at the drive's cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    finally:
        os.close(fd)


def write_shard(path: Path, t: pa.Table) -> None:
    """Write, check by reading back, flush, then rename: a shard under its name is whole on the
    drive before any original is deleted, and the upload never picks up a `.tmp`."""
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp, row_group_size=500, compression=COMPRESSION)
    if not pq.read_table(tmp).equals(t):
        raise RuntimeError(f"{tmp} does not read back as written")
    durable(tmp)
    tmp.replace(path)
    durable(path.parent)


def lock(root: Path, step: str):
    root.mkdir(parents=True, exist_ok=True)
    return exclusive(root / f".{step}.lock", f"another {step} run is using {root}")


def retried(what: str, fn: Callable[[], T]) -> T:
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRIES:
                raise
            wait = RETRY_WAIT * 2**attempt
            log(f"{what} failed ({type(e).__name__}: {e}); retrying in {wait // 60} min")
            time.sleep(wait)


# ------------------------------------------------------------------ en_speech


def move_en_speech(src: Path, vi_text: Path, root: Path, workers: int) -> None:
    if not src.exists():
        if not any(root.glob("data/*/*.parquet")):
            raise SystemExit(f"{src} does not exist and no en_speech shard is in {root}")
        log(f"{src} is gone: en_speech is already moved")
        return
    if src.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"--src {src} and --root {root} overlap")
    for split in SPLITS:
        export = pq.read_table(vi_text / f"{split}.parquet")
        vi = dict(zip(export["id"].to_pylist(), export["text_vi"].to_pylist()))
        out = root / "data" / split
        out.mkdir(parents=True, exist_ok=True)
        sources = sorted((src / split).glob("*.parquet"))
        moved = list(out.glob("*.parquet"))
        ids = set()
        for p in sources + moved:
            ids.update(pq.read_table(p, columns=["id"])["id"].to_pylist())
        if stray := len(vi.keys() - ids):
            raise RuntimeError(f"{split}: {stray:,} translated ids are in neither {src} nor {out}; "
                               "the export does not match the corpus")
        log(f"{split}: {len(sources)} shards to move, {len(moved)} moved, {len(ids) - len(vi):,} rows without a translation")

        def move(p: Path) -> None:
            # An empty shard only marked a source file done for the corpus builder. A shard whose
            # original survives is rebuilt, as a crash may have cut it off before the flush.
            if pq.read_metadata(p).num_rows:
                t = read_en_speech(p, vi)
                write_shard(out / p.name, t)
                log(f"moved {split}/{p.name} ({t.num_rows} rows)")
            p.unlink()

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(move, sources))
    if left := list(src.rglob("*.parquet")):
        raise RuntimeError(f"{len(left)} shards outside the splits are left in {src}, e.g. {left[0]}")
    shutil.rmtree(src)  # the corpus card and Finder files
    log(f"en_speech moved into {root}; {src} removed")


def read_en_speech(path: Path, vi: dict[str, str]) -> pa.Table:
    t = pq.read_table(path)
    text_vi = pa.array([vi.get(i) for i in t["id"].to_pylist()], pa.string())
    # Only transcripts without letters may lack a translation; anything else means the export is
    # stale or partial, and dropping the row would delete its clip.
    if lost := [x for x, v in zip(t["text"].to_pylist(), text_vi.to_pylist()) if v is None and LETTER.search(x)]:
        raise RuntimeError(f"{path}: {len(lost)} transcripts have no translation, e.g. {lost[0]!r}")
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

English read speech with English transcripts and Vietnamese translations, for research on
Vietnamese→English simultaneous speech translation (Hibiki-Zero). The Vietnamese speech columns
are empty until a voice-transfer TTS step fills them. Built by
`training/prep/publish_en_vi_speech.py` in the hibiki-zero repository.

| split | source | shards | rows | EN hours | VI audio hours | speakers | GB |
|---|---|---:|---:|---:|---:|---:|---:|
{{STATS}}

## Columns

| column | type | content |
|---|---|---|
| `id` | string | `cv27/<clip>`, `libriheavy/<subset>/<speaker>/<book>/<recording>_<segment>` |
| `en_audio` | Audio | English clip; the `path` extension names the codec |
| `text_en` | string | cased, punctuated English transcript |
| `text_vi` | string | Vietnamese translation |
| `speaker` | string | `cv27/<client_id>` or `libriheavy/<speaker_id>` |
| `en_duration_s` | float32 | seconds |
| `vi_audio` | Audio | Vietnamese clip; empty until voice-transfer TTS fills it |
| `vi_duration_s` | float32 | seconds, empty with `vi_audio` |
| `cross_lingual_timbre_matched` | bool | whether the Vietnamese voice matches the English speaker's; empty with `vi_audio` |

## Sources

Every English clip lasts {{CLIP}}, so most fit the 280-frame (12.5 Hz) training rows.

- Common Voice Scripted Speech 27.0 English: validated clips only. dev is the validation
  split; train is every other validated clip whose speaker and sentence are not in dev. The
  official test clips are folded into train, so CV test is not a clean benchmark here. Audio is
  the original 32 or 48 kHz MP3.
- Libriheavy (`mythicinfinity/libriheavy`, LibriVox audiobooks): mono rows only, in the
  original 48 kHz Ogg Opus. dev is the validation split; train is medium, small and
  seeded-random large files. `text_en` is the book text (`text_original`).
- Every Common Voice clip and Libriheavy chapter reaches 10.5 kHz of audio bandwidth (the
  highest frequency within 50 dB of the spectral peak, HiFiTTS-2's estimate), and train keeps
  each speaker's clips, in build order, until the speaker holds 30 h.
- Their English text was translated by Claude Sonnet 5.5, Gemini 3.8 Flash and Muse Spark 1.3,
  in batches of neighbouring lines.

The audio columns hold codec bytes. Read raw rows with `pyarrow` and decode with `soundfile`
(`sf.read(io.BytesIO(row["en_audio"]["bytes"]))`), resampling to 24 kHz for Mimi, or with
`datasets` (needs `torchcodec`):
`load_dataset("huybik/en-vi-speech", streaming=True).cast_column("en_audio", Audio(sampling_rate=24_000))`.

## Use

For research only. Common Voice is CC0-1.0, distributed by the Mozilla Data Collective, whose
terms forbid re-hosting it and trying to identify its speakers. Libriheavy is Apache-2.0 over
public-domain LibriVox recordings.
"""


def write_card(root: Path) -> None:
    rows = []
    for split in SPLITS:
        for source in ("cv27", "libriheavy"):
            shards = sorted((root / "data" / split).glob(f"{source}-*.parquet"))
            if not shards:
                continue
            t = pa.concat_tables(pq.read_table(p, columns=["speaker", "en_duration_s", "vi_duration_s"]) for p in shards)
            rows.append(
                (
                    split,
                    source,
                    len(shards),
                    t.num_rows,
                    pc.sum(t["en_duration_s"]).as_py() / 3600,
                    (pc.sum(t["vi_duration_s"]).as_py() or 0) / 3600,
                    len(pc.unique(t["speaker"]).drop_null()),
                    sum(p.stat().st_size for p in shards) / 1e9,
                )
            )
    table = "\n".join(
        f"| {s} | {src} | {n} | {r:,} | {h:,.1f} | {vh:,.1f} | {spk:,} | {gb:.1f} |"
        for s, src, n, r, h, vh, spk, gb in rows
    )
    tmp = root / "README.md.tmp"  # so a running upload never reads half a card
    tmp.write_text(CARD.replace("{{STATS}}", table).replace("{{CLIP}}", f"{CLIP_SECONDS[0]:g}-{CLIP_SECONDS[1]:g} s"))
    tmp.replace(root / "README.md")
    log(f"wrote {root / 'README.md'}:\n{table}")


def upload(root: Path, repo: str) -> None:
    """Commit the shards missing from the Hub, COMMIT_BYTES at a time, then the card. The Hub is
    the record of progress: a shard there at its local size is uploaded, so a stopped run loses
    only its unfinished commit. Shards written meanwhile go in later commits."""
    api = HfApi()
    remote = {f.path: f.size for f in api.list_repo_tree(repo, recursive=True, repo_type="dataset") if isinstance(f, RepoFile)}
    while todo := [p for p in sorted(root.glob("data/*/*.parquet")) if remote.get(p.relative_to(root).as_posix()) != p.stat().st_size]:
        batch, size = [], 0
        for p in todo:
            if batch and size + p.stat().st_size > COMMIT_BYTES:
                break
            batch.append(p)
            size += p.stat().st_size
        retried(f"committing {batch[0].name}", lambda: api.create_commit(
            repo, [CommitOperationAdd(p.relative_to(root).as_posix(), p) for p in batch],
            commit_message=f"Add {len(batch)} shards from {batch[0].name}", repo_type="dataset"))
        remote.update((p.relative_to(root).as_posix(), p.stat().st_size) for p in batch)
        log(f"committed {len(batch)} shards from {batch[0].name} ({size / 1e9:.1f} GB), {len(todo) - len(batch)} left")
    card = root / "README.md"
    if card.exists():  # create_commit skips it when unchanged
        retried("committing README.md", lambda: api.upload_file(
            path_or_fileobj=card, path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update the card"))
    log(f"uploaded {root} to {repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("en_speech", "card", "upload"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    p.add_argument("--vi-text", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_text"))
    p.add_argument("--workers", type=int, default=4, help="shards moved at once (en_speech)")
    args = p.parse_args()
    with lock(args.root, args.step):
        if args.step == "en_speech":
            move_en_speech(args.src, args.vi_text, args.root, args.workers)
        elif args.step == "card":
            write_card(args.root)
        else:
            upload(args.root, args.repo)


if __name__ == "__main__":
    sys.exit(main())
