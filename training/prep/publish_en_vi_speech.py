#!/usr/bin/env python
"""Build the EN-VI speech dataset that SFT pair rows are cached from, and publish it.

Each row holds an English clip of the en_speech corpus (Common Voice 27 and Libriheavy read
speech) with its transcript and the Vietnamese translation exported by
training/prep/translate_en_speech.py. The voice-transfer TTS step (training/prep/tts_en_speech.py)
fills `vi_audio` and `vi_duration_s` shard by shard. Rows without a translation (transcripts
without letters) were dropped when the corpus was moved in; the step that moved it is in git at
a236ad5.

--root, laid out as the Hugging Face repo, is the one local copy. Run from the repo root as
`python -m training.prep.publish_en_vi_speech <step>`. Steps:
  card       README.md with statistics of the local dataset
  upload     commits the shards missing from the Hub or voiced here since, a few GB at a time,
             downloads the shards voiced on the Hub by another machine, then commits a fresh card
             (export HF_TOKEN); with --follow it looks again every 10 min until every shard is
             voiced
Every step may be stopped (Ctrl-C, kill) and resumes when run again; a lock stops a second run of
the same step.

Every shard has one schema:
  data/{train,validation}/{cv27,libriheavy}-*.parquet
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
from huggingface_hub.utils import disable_progress_bars  # noqa: E402

from training.prep.translate_en_speech import MAX_SECONDS, exclusive, log  # noqa: E402

REPO = "huybik/en-vi-speech"
CLIP_SECONDS = (5.0, MAX_SECONDS)  # en_speech's clip range; longer clips never fit a 280-frame row
SPLITS = ("train", "validation")
RETRIES = 5  # a failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
# 2-7 min of upload, all a stop loses; ~20-30 commits an hour, well under the Hub's commit quota
# (~100 an hour).
COMMIT_BYTES = 4_000_000_000
FOLLOW_WAIT = 600  # a TTS shard is voiced every ~3 min
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
    drive, and the upload never picks up a `.tmp`."""
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp, row_group_size=500, compression=COMPRESSION)
    if not pq.read_table(tmp).equals(t):
        raise RuntimeError(f"{tmp} does not read back as written")
    durable(tmp)
    tmp.replace(path)
    durable(path.parent)


def voiced(path: Path) -> bool:
    """Whether the TTS step has filled the shard's Vietnamese speech."""
    return pq.read_table(path, columns=["vi_duration_s"])["vi_duration_s"].null_count == 0


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

English read speech with English transcripts, Vietnamese translations and Vietnamese speech in
the English speaker's voice, for research on Vietnamese→English simultaneous speech translation
(Hibiki-Zero). The Vietnamese speech is being synthesized shard by shard; shards not voiced yet
have it empty, and the table shows how far it has come. Built by
`training/prep/publish_en_vi_speech.py` and `training/prep/tts_en_speech.py` in the hibiki-zero
repository.

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
| `vi_audio` | Audio | `text_vi` spoken by VieNeu-TTS v3 Turbo, cloning the voice of `en_audio`'s first 8 s; 48 kHz mono Ogg Opus, ~32 kbps; empty in shards not voiced yet |
| `vi_duration_s` | float32 | seconds, empty with `vi_audio` |
| `cross_lingual_timbre_matched` | bool | whether the Vietnamese voice matches the English speaker's; empty until speaker similarity is measured |

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
- The Vietnamese speech is synthetic: VieNeu-TTS v3 Turbo (`pnnbao-ump/VieNeu-TTS-v3-Turbo`),
  run in MLX, takes the speaker x-vector and in-context reference from the row's own English
  clip. On 120 validation rows, its speaker similarity to the English clip (WeSpeaker ResNet34
  cosine) averaged 0.50 on Common Voice and 0.56 on Libriheavy, where another clip of the same
  speaker scores ~0.78 and a different speaker ~0.10; Whisper large-v3-turbo's character error
  rate against `text_vi` averaged 12% and 3%.

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


def upload(root: Path, repo: str, follow: bool) -> None:
    """Match the local shards and the Hub's, then commit a fresh card. A shard differing from its
    Hub copy in size is downloaded when only the Hub's is voiced (the TTS runs top down here and
    bottom up on another machine) and committed otherwise, COMMIT_BYTES at a time. The Hub is the
    record of progress: a shard there at its local size is uploaded, so a stopped run loses only
    its unfinished commit. Shards written meanwhile go in later commits. With follow, it looks
    again every FOLLOW_WAIT until every shard is voiced; the Hub's Xet storage sends only the new
    bytes of a rewritten shard."""
    disable_progress_bars()  # in a log, their redraws come to ~1 MB a commit
    api = HfApi()
    changed = True  # the first pass sends the card even when no shard changed
    while True:
        if not (local := sorted(root.glob("data/*/*.parquet"))):
            raise SystemExit(f"no shards in {root / 'data'}")  # a dropped drive, or the wrong --root
        done = not follow or all(voiced(p) for p in local)  # before the pass, so it sends the last
        remote = retried("listing the Hub", lambda: {
            f.path: f.size for f in api.list_repo_tree(repo, recursive=True, repo_type="dataset") if isinstance(f, RepoFile)})
        for p in local:
            name = p.relative_to(root).as_posix()
            if name in remote and remote[name] != p.stat().st_size and not voiced(p):
                pull(api, repo, root, p)
                changed = True
        while todo := [p for p in local if remote.get(p.relative_to(root).as_posix()) != p.stat().st_size]:
            batch, size = {}, 0
            for p in todo:
                if batch and size + p.stat().st_size > COMMIT_BYTES:
                    break
                batch[p] = p.stat().st_size
                size += batch[p]
            first = next(iter(batch)).name
            retried(f"committing {first}", lambda: api.create_commit(
                repo, [CommitOperationAdd(p.relative_to(root).as_posix(), p) for p in batch],
                commit_message=f"Add {len(batch)} shards from {first}", repo_type="dataset"))
            remote.update((p.relative_to(root).as_posix(), n) for p, n in batch.items())
            log(f"committed {len(batch)} shards from {first} ({size / 1e9:.1f} GB), {len(todo) - len(batch)} left")
            changed = True
        if changed:
            write_card(root)
            card = root / "README.md"  # create_commit skips it when unchanged
            retried("committing README.md", lambda: api.upload_file(
                path_or_fileobj=card, path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update the card"))
            log(f"uploaded {root} to {repo}")
        if done:
            return
        changed = False
        time.sleep(FOLLOW_WAIT)


def pull(api: HfApi, repo: str, root: Path, path: Path) -> None:
    """Replace a local shard with its Hub copy, voiced on another machine, once that checks out:
    the same schema, every column but the Vietnamese speech equal, and the speech filled."""
    name = path.relative_to(root).as_posix()
    got = Path(retried(f"downloading {name}", lambda: api.hf_hub_download(
        repo, name, repo_type="dataset", local_dir=root / ".pull")))

    def column(p: Path, c: str) -> pa.ChunkedArray:  # one at a time: en_audio alone is ~200 MB
        return pq.read_table(p, columns=[c])[c]

    if not (pq.read_schema(got).equals(SCHEMA) and column(got, "vi_audio").null_count == 0 and voiced(got)
            and all(column(got, c).equals(column(path, c)) for c in SCHEMA.names if c not in ("vi_audio", "vi_duration_s"))):
        raise RuntimeError(f"the Hub's {name} is not this shard voiced; it is left in {got}")
    durable(got)
    got.replace(path)
    durable(path.parent)
    shutil.rmtree(root / ".pull")  # the download's metadata
    log(f"took {name}, voiced on the Hub")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("card", "upload"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--follow", action="store_true", help="keep uploading until every shard is voiced (upload)")
    args = p.parse_args()
    with lock(args.root, args.step):
        if args.step == "card":
            write_card(args.root)
        else:
            upload(args.root, args.repo, args.follow)


if __name__ == "__main__":
    sys.exit(main())
