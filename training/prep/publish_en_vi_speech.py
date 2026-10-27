#!/usr/bin/env python
"""Publish the EN-VI speech dataset that SFT pair rows are cached from.

Each row holds an English clip of the en_speech corpus (Common Voice 27 and Libriheavy read
speech) with its transcript and the Vietnamese translation exported by
training/prep/translate_en_speech.py. The voice-transfer TTS step (training/prep/tts_en_speech.py)
fills `vi_audio` and `vi_duration_s` shard by shard. Rows without a translation (transcripts
without letters) were dropped when the corpus was moved in; the step that moved it is in git at
a236ad5. The `cv-anquach` rows come voiced from anquachdev/common-vi-en as it was before the merge:
the Common Voice clips the corpus lacks, with that dataset's translation and Vietnamese speech.
Both machines voice into huybik/en-vi-speech; once every shard is voiced, the dataset is uploaded
to anquachdev/common-vi-en (--repo, HF_TOKEN of that account), the one copy used from then on.

--root, laid out as the Hugging Face repo, is the one local copy. Run from the repo root as
`python -m training.prep.publish_en_vi_speech <step>`. Steps:
  anquach    data/train/cv-anquach-*.parquet from a snapshot of the earlier common-vi-en (--src)
  card       README.md with statistics of the local dataset
  upload     commits the shards voiced here since and those the Hub lacks, a few GB at a time,
             downloads the shards voiced on the Hub by another machine, then commits a fresh card
             (export HF_TOKEN); with --follow it looks again every 10 min until every shard is voiced
Every step may be stopped (Ctrl-C, kill) and resumes when run again; a lock stops a second run of
the same step.

Every shard has one schema:
  data/{train,validation}/{cv27,libriheavy}-*.parquet, data/train/cv-anquach-*.parquet
  id, en_audio{bytes,path}, text_en, text_vi, speaker, en_duration_s,
  vi_audio{bytes,path}, vi_duration_s, cross_lingual_timbre_matched
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
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
import soundfile as sf  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, RepoFile, snapshot_download  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402
from huggingface_hub.utils import disable_progress_bars  # noqa: E402

from training.paths import durable, exclusive  # noqa: E402
from training.prep.translate_en_speech import MAX_SECONDS, log  # noqa: E402

REPO = "huybik/en-vi-speech"  # where both machines voice; anquachdev/common-vi-en once all is voiced
ANQUACH_REPO = "anquachdev/common-vi-en"
CLIP_SECONDS = (5.0, MAX_SECONDS)  # en_speech's clip range; longer clips never fit a 280-frame row
SPLITS = ("train", "validation")
SOURCES = ("cv27", "cv-anquach", "libriheavy")  # shard and id prefixes
# common-vi-en before the merge: 739,346 Common Voice English rows, voiced, 16 kHz Ogg Opus.
ANQUACH_REVISION = "dba8e98cef20eb0493f747ef040180fa81512d17"
SHARD_ROWS = 20_000  # the cv27 builder's shard size, before the corpus filters thinned its shards
RETRIES = 5  # a failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
# 2-7 min of upload, all a stop loses; ~20-30 commits an hour, under the Hub's commit quota (~100
# an hour). Seeding a repository with bytes Xet already holds ran at ~110 an hour, none refused.
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


# ------------------------------------------------------------------ cv-anquach rows


def anquach(root: Path, src: Path) -> None:
    """Write data/train/cv-anquach-*.parquet from ANQUACH_REPO as it was before the merge: its rows
    whose English lasts CLIP_SECONDS, except the clips cv27 holds (ours stay) and the sentences of
    cv27's validation split, compared lower-cased with punctuation and spaces collapsed (cv27 train
    leaves them out too), and those whose translation holds Chinese, Japanese or Korean script
    (spoken from that broken text). They are clips cv27 lacks, dropped by its filters (such as the 10.5 kHz bandwidth floor or its
    validation speakers) or not among its validated clips, and their English is that dataset's
    16 kHz re-encode; their speaker is unknown. Rerunning writes the same shards."""

    def sentence(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    cjk = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

    snapshot_download(ANQUACH_REPO, repo_type="dataset", revision=ANQUACH_REVISION, local_dir=src,
                      allow_patterns=["data/train-heavy-*.parquet"])
    ours, dev = set(), set()
    for path in root.glob("data/*/cv27-*.parquet"):
        t = pq.read_table(path, columns=["id", "text_en"])
        ours.update(i.removeprefix("cv27/") for i in t["id"].to_pylist())
        if path.parent.name == "validation":
            dev.update(sentence(s) for s in t["text_en"].to_pylist())
    rows, shards, seen, kept = [], 0, 0, 0

    def flush() -> None:
        nonlocal rows, shards
        write_shard(root / "data" / "train" / f"cv-anquach-{shards:05d}.parquet", pa.Table.from_pylist(rows, schema=SCHEMA))
        shards += 1
        rows = []

    for path in sorted((src / "data").glob("train-heavy-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
            for r in batch.to_pylist():
                seen += 1
                clip, en, vi = r["source_clip"].removesuffix(".mp3"), r["audio_en"]["bytes"], r["audio_vi"]["bytes"]
                en_s = sf.info(io.BytesIO(en)).duration
                if not CLIP_SECONDS[0] <= en_s <= CLIP_SECONDS[1] or clip in ours or sentence(r["en"]) in dev or cjk.search(r["vi"]):
                    continue
                vi_s = sf.info(io.BytesIO(vi)).duration
                if not (r["en"].strip() and r["vi"].strip() and vi_s > 0):
                    raise ValueError(f"{path.name}: {clip} has no English or Vietnamese text, or no Vietnamese speech")
                rows.append({
                    "id": f"cv-anquach/{clip}", "en_audio": {"bytes": en, "path": f"{clip}.opus"},
                    "text_en": r["en"], "text_vi": r["vi"], "speaker": None, "en_duration_s": en_s,
                    "vi_audio": {"bytes": vi, "path": f"{clip}.opus"}, "vi_duration_s": vi_s,
                    "cross_lingual_timbre_matched": None})
                kept += 1
                if len(rows) == SHARD_ROWS:
                    flush()
    if rows:
        flush()
    log(f"wrote {shards} cv-anquach shards: {kept:,} of common-vi-en's {seen:,} rows")


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

# {{NAME}}

English read speech with English transcripts, Vietnamese translations and Vietnamese speech,
mostly in the English speaker's voice, for research on Vietnamese→English simultaneous speech translation
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
| `id` | string | `cv27/<clip>`, `cv-anquach/<clip>`, `libriheavy/<subset>/<speaker>/<book>/<recording>_<segment>` |
| `en_audio` | Audio | English clip; the `path` extension names the codec |
| `text_en` | string | cased, punctuated English transcript |
| `text_vi` | string | Vietnamese translation |
| `speaker` | string | `cv27/<client_id>` or `libriheavy/<speaker_id>`; empty for `cv-anquach` |
| `en_duration_s` | float32 | seconds |
| `vi_audio` | Audio | `text_vi` spoken; `cv27` and `libriheavy`: by VieNeu-TTS v3 Turbo, cloning the voice of `en_audio`'s first 8 s, 48 kHz mono Ogg Opus, ~32 kbps, empty in shards not voiced yet; `cv-anquach`: 16 kHz mono Ogg Opus, ~16 kbps |
| `vi_duration_s` | float32 | seconds, empty with `vi_audio` |
| `cross_lingual_timbre_matched` | bool | whether the Vietnamese voice matches the English speaker's; empty until speaker similarity is measured |

## Sources

Every English clip lasts {{CLIP}}, so most fit the 280-frame (12.5 Hz) training rows.

- `cv27`, Common Voice Scripted Speech 27.0 English: validated clips only. dev is the validation
  split; train is every other validated clip whose speaker and sentence are not in dev. The
  official test clips are folded into train, so CV test is not a clean benchmark here. Audio is
  the original 32 or 48 kHz MP3.
- Libriheavy (`mythicinfinity/libriheavy`, LibriVox audiobooks): mono rows only, in the
  original 48 kHz Ogg Opus. dev is the validation split; train is medium, small and
  seeded-random large files. `text_en` is the book text (`text_original`).
- Every `cv27` clip and Libriheavy chapter reaches 10.5 kHz of audio bandwidth (the
  highest frequency within 50 dB of the spectral peak, HiFiTTS-2's estimate), and train keeps
  each speaker's clips, in build order, until the speaker holds 30 h.
- `cv-anquach` (train only) holds the earlier rows of `anquachdev/common-vi-en` (revision
  `dba8e98`, 739,346 Common Voice English clips with Vietnamese translations and speech) whose
  English lasts {{CLIP}}, except the clips `cv27` holds and the sentences of its validation
  split. They are clips `cv27` lacks, dropped by its filters above (such as the 10.5 kHz floor)
  or not among its validated clips. Their English is a 16 kHz mono Ogg Opus re-encode
  (~16 kbps), so it carries nothing above 8 kHz. Their speaker is not recorded, so some may also
  speak in the validation split, and their Vietnamese speech, from that dataset's own TTS, is not
  known to match the English speaker's voice.
- The `cv27` and `libriheavy` English text was translated by Claude Sonnet 5.5, Gemini 3.8 Flash
  and Muse Spark 1.3, in batches of neighbouring lines.
- The `cv27` and `libriheavy` Vietnamese speech is synthetic: VieNeu-TTS v3 Turbo
  (`pnnbao-ump/VieNeu-TTS-v3-Turbo`), run in MLX, takes the speaker x-vector and in-context
  reference from the row's own English clip. On 120 validation rows, its speaker similarity to
  the English clip (WeSpeaker ResNet34 cosine) averaged 0.50 on Common Voice and 0.56 on
  Libriheavy, where another clip of the same speaker scores ~0.78 and a different speaker ~0.10;
  Whisper large-v3-turbo's character error rate against `text_vi` averaged 12% and 3%.

The audio columns hold codec bytes. Read raw rows with `pyarrow` and decode with `soundfile`
(`sf.read(io.BytesIO(row["en_audio"]["bytes"]))`), resampling to 24 kHz for Mimi, or with
`datasets` (needs `torchcodec`):
`load_dataset("{{REPO}}", streaming=True).cast_column("en_audio", Audio(sampling_rate=24_000))`.

## Use

For research only. Common Voice is CC0-1.0, distributed by the Mozilla Data Collective, whose
terms forbid re-hosting it and trying to identify its speakers. Libriheavy is Apache-2.0 over
public-domain LibriVox recordings.
"""


def write_card(root: Path, repo: str) -> None:
    rows = []
    for split in SPLITS:
        for source in SOURCES:
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
        f"| {s} | {src} | {n} | {r:,} | {h:,.1f} | {vh:,.1f} | {f'{spk:,}' if spk else 'unknown'} | {gb:.1f} |"
        for s, src, n, r, h, vh, spk, gb in rows
    )
    tmp = root / f"README.md.{os.getpid()}.tmp"  # so no run reads half a card, or writes into another's
    card = CARD.replace("{{STATS}}", table).replace("{{CLIP}}", f"{CLIP_SECONDS[0]:g}-{CLIP_SECONDS[1]:g} s")
    tmp.write_text(card.replace("{{REPO}}", repo).replace("{{NAME}}", repo.split("/")[1]))
    tmp.replace(root / "README.md")
    log(f"wrote {root / 'README.md'}:\n{table}")


def upload(root: Path, repo: str, follow: bool) -> None:
    """Match the local shards and the Hub's, then commit a fresh card. The TTS runs top down here
    and bottom up on another machine. A voiced local shard that differs from its Hub copy in size
    is committed, COMMIT_BYTES at a time; an unvoiced one is replaced by its Hub copy, voiced
    there. An unvoiced shard is committed only when the Hub has none of it, so it never replaces
    a voiced Hub copy, and the other machine finds every shard there to voice. The Hub
    is the record of progress: a shard there at its local size is uploaded, so a stopped run loses
    only its unfinished commit. Shards written meanwhile go in later commits. With follow, it
    looks again every FOLLOW_WAIT until every shard is voiced; the Hub's Xet storage sends only
    the new bytes of a rewritten shard."""
    disable_progress_bars()  # in a log, their redraws come to ~1 MB a commit
    api = HfApi()

    def hub(p: Path) -> str:
        return p.relative_to(root).as_posix()

    def commit_next(local: list[Path]) -> tuple[dict[str, int], dict[Path, int], int]:
        """List the Hub and commit the next batch on top of that listing's revision, so a commit
        landing in between (the other machine's) fails this one, and it lists again at once: an
        unvoiced shard never lands on a voiced copy committed meanwhile. Returns the listing, the
        batch committed (empty when nothing is left) and the shards left before it."""
        while True:
            sha = api.repo_info(repo, repo_type="dataset").sha
            remote = {f.path: f.size for f in api.list_repo_tree(repo, recursive=True, revision=sha, repo_type="dataset")
                      if isinstance(f, RepoFile)}
            todo = [p for p in local if hub(p) not in remote or (remote[hub(p)] != p.stat().st_size and voiced(p))]
            batch, size = {}, 0
            for p in todo:
                if batch and size + p.stat().st_size > COMMIT_BYTES:
                    break
                batch[p] = p.stat().st_size
                size += batch[p]
            if not batch:
                return remote, batch, 0
            try:
                api.create_commit(repo, [CommitOperationAdd(hub(p), p) for p in batch], repo_type="dataset",
                                  commit_message=f"Add {len(batch)} shards from {next(iter(batch)).name}", parent_commit=sha)
            except HfHubHTTPError as e:
                if e.response.status_code not in (409, 412):  # anything but a moved branch
                    raise
                log(f"the Hub moved past {sha[:7]} during the commit; listing again")
                continue
            return remote, batch, len(todo)

    changed = True  # the first pass sends the card even when no shard changed
    while True:
        if not (local := sorted(root.glob("data/*/*.parquet"))):
            raise SystemExit(f"no shards in {root / 'data'}")  # a dropped drive, or the wrong --root
        done = not follow or all(voiced(p) for p in local)  # before the pass, so it sends the last
        while True:
            remote, batch, todo = retried("committing to the Hub", lambda: commit_next(local))
            if not batch:
                break
            log(f"committed {len(batch)} shards from {next(iter(batch)).name} ({sum(batch.values()) / 1e9:.1f} GB), "
                f"{todo - len(batch)} left")
            changed = True
        for p in local:
            if hub(p) in remote and remote[hub(p)] != p.stat().st_size and not voiced(p):
                pull(api, repo, root, p)
                changed = True
        if changed:
            write_card(root, repo)
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
    the same schema, every column but the Vietnamese speech equal, and the speech filled. Comparing
    the en_audio columns adds ~1 GB for the ~20 s of a pull."""
    name = path.relative_to(root).as_posix()
    got = Path(retried(f"downloading {name}", lambda: api.hf_hub_download(
        repo, name, repo_type="dataset", local_dir=root / ".pull")))

    def column(p: Path, c: str) -> pa.ChunkedArray:  # one at a time: en_audio alone is ~200 MB
        return pq.read_table(p, columns=[c])[c]

    def spoken() -> bool:  # every row has audio bytes and a positive duration
        audio = pc.struct_field(column(got, "vi_audio"), "bytes")  # null under a null row too
        seconds = column(got, "vi_duration_s")
        return (audio.null_count == seconds.null_count == 0 and pc.all(pc.greater(pc.binary_length(audio), 0)).as_py()
                and pc.all(pc.greater(seconds, 0)).as_py())  # NaN fails too

    if not (pq.read_schema(got).equals(SCHEMA) and spoken()
            and all(column(got, c).equals(column(path, c)) for c in SCHEMA.names if c not in ("vi_audio", "vi_duration_s"))):
        raise RuntimeError(f"the Hub's {name} is not this shard voiced; it is left in {got}")
    durable(got)
    if voiced(path):  # the TTS voiced it here while it downloaded: the first voicing stays
        shutil.rmtree(root / ".pull")
        log(f"kept {name}, voiced here while the Hub's copy downloaded")
        return
    got.replace(path)
    durable(path.parent)
    shutil.rmtree(root / ".pull")  # the download's metadata
    log(f"took {name}, voiced on the Hub")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("anquach", "card", "upload"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/common_vi_en_src"),
                   help="where the earlier common-vi-en is downloaded to (anquach)")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--follow", action="store_true", help="keep uploading until every shard is voiced (upload)")
    args = p.parse_args()
    with lock(args.root, args.step):
        if args.step == "anquach":
            anquach(args.root, args.src)
        elif args.step == "card":
            write_card(args.root, args.repo)
        else:
            upload(args.root, args.repo, args.follow)


if __name__ == "__main__":
    sys.exit(main())
