#!/usr/bin/env python
"""Build the EN-VI speech dataset that SFT pair rows are cached from, and publish it.

Each row holds an English clip with its transcript and Vietnamese translation, and the
Vietnamese clip where one exists:
  - en_speech: Common Voice 27 and Libriheavy read speech, with the Vietnamese text exported by
    training/prep/translate_en_speech.py. `vi_audio` stays empty until the voice-transfer TTS
    step fills it. Rows without a translation (transcripts without letters) are dropped.
  - PhoMT-en-vi-speech: TTS speech on both sides. Its rows are those of the grounded-v2 cache
    in CACHE_REPO, which passed English CTC alignment, whose English clip lasts CLIP_SECONDS.
    Ids are the cache's, so cache rows map back. The PCM WAV is stored as lossless FLAC, about
    40% of its size.

--root, laid out as the Hugging Face repo, is the one local copy: the en_speech step moves the
corpus into it. Run from the repo root as `python -m training.prep.publish_en_vi_speech <step>`.
Steps:
  en_speech  converts each shard of the en_speech corpus at --src into a dataset shard of the
             same name and deletes the original, then the emptied corpus folder
  phomt      downloads each PhoMT source file and writes its dataset shard
  card       README.md with statistics of the local dataset
  upload     uploads what the repo lacks with upload_large_folder (export HF_TOKEN); rerun
             it after adding shards
Every step resumes when run again; a lock stops a second run of the same step.

Every shard has one schema:
  data/{train,validation}/{cv27,libriheavy-*,phomt-*}.parquet
  id, en_audio{bytes,path}, text_en, text_vi, speaker, en_duration_s,
  vi_audio{bytes,path}, vi_duration_s, cross_lingual_timbre_matched
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
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
import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from training.prep.translate_en_speech import MAX_SECONDS, exclusive, log  # noqa: E402

REPO = "huybik/en-vi-speech"
CACHE_REPO = "huybik/hibiki-zero-vi-full-sft"
PAIRS = [f"grounded-v2/pairs_w{w}.jsonl" for w in range(5)]
PHOMT = "anquachdev/PhoMT-en-vi-speech"
PHOMT_REVISION = "33400f73dde07da539e8326313cbabe20b757740"  # the revision grounded-v2 was built from
CLIP_SECONDS = (5.0, MAX_SECONDS)  # en_speech's clip range; longer clips never fit a 280-frame row
SPLITS = ("train", "validation")
RETRIES = 5  # a failed PhoMT file is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
UPLOAD_WORKERS = 4  # more only adds memory: the link (~37 MB/s) is the limit
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
    """Write, check by reading back, then rename: a shard under its name is always whole, and
    the upload never picks up a `.tmp`."""
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp, row_group_size=500, compression=COMPRESSION)
    if not pq.read_table(tmp).equals(t):
        raise RuntimeError(f"{tmp} does not read back as written")
    tmp.replace(path)


def lock(root: Path, step: str):
    root.mkdir(parents=True, exist_ok=True)
    return exclusive(root / f".{step}.lock", f"another {step} run is using {root}")


# ------------------------------------------------------------------ en_speech


def move_en_speech(src: Path, vi_text: Path, root: Path, workers: int) -> None:
    for split in SPLITS:
        export = pq.read_table(vi_text / f"{split}.parquet")
        vi = dict(zip(export["id"].to_pylist(), export["text_vi"].to_pylist()))
        out = root / "data" / split
        out.mkdir(parents=True, exist_ok=True)
        sources = sorted((src / split).glob("*.parquet"))
        moved = [p for p in out.glob("*.parquet") if not p.name.startswith("phomt-")]
        ids = set()
        for p in sources + moved:
            ids.update(pq.read_table(p, columns=["id"])["id"].to_pylist())
        if stray := len(vi.keys() - ids):
            raise RuntimeError(f"{split}: {stray:,} translated ids are in neither {src} nor {out}; "
                               "rerun translate_en_speech texts and export")
        log(f"{split}: {len(sources)} shards to move, {len(moved)} moved, {len(ids) - len(vi):,} rows without a translation")

        def move(p: Path) -> None:
            dst = out / p.name
            # An empty shard only marked a source file done for the corpus builder.
            if not dst.exists() and pq.read_metadata(p).num_rows:
                t = read_en_speech(p, vi)
                write_shard(dst, t)
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


def build_phomt(root: Path, workers: int) -> None:
    """One shard per PhoMT source file, named by the cache's file number: phomt_s00304r00012 is
    row 12 of the file the cache numbered 304."""
    picks: dict[str, list[tuple[int, str, bool]]] = {}
    for name in PAIRS:
        with open(hf_hub_download(CACHE_REPO, name, repo_type="dataset")) as f:
            for line in f:
                p = json.loads(line)
                src, row = p["en_audio"].removeprefix(f"hf://{PHOMT}/").split("#r")
                picks.setdefault(src, []).append((int(row), p["id"], p["cross_lingual_timbre_matched"]))
    out = root / "data" / "train"
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for src, rows in picks.items():
        dst = out / f"phomt-{rows[0][1].removeprefix('phomt_').split('r')[0]}.parquet"
        if not dst.exists():
            todo.append((dst, src, rows))
    log(f"PhoMT: {sum(map(len, picks.values())):,} cache rows in {len(picks)} source files, {len(todo)} to build")

    def build(dst: Path, src: str, rows: list[tuple[int, str, bool]]) -> None:
        t = retried(f"reading {src}", lambda: read_phomt(src, rows))
        write_shard(dst, t)
        log(f"wrote {dst.name}: {t.num_rows} of {len(rows)} cache rows")

    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(build, *task) for task in sorted(todo)]:
            f.result()
    log("PhoMT done")


def read_phomt(src: str, rows: list[tuple[int, str, bool]]) -> pa.Table:
    with tempfile.TemporaryDirectory() as tmp:
        local = hf_hub_download(PHOMT, src, repo_type="dataset", revision=PHOMT_REVISION, local_dir=tmp)
        t = pq.read_table(local).take(pa.array([r for r, _, _ in rows]))
    keep = pc.and_(pc.greater_equal(t["duration_en_s"], CLIP_SECONDS[0]), pc.less_equal(t["duration_en_s"], CLIP_SECONDS[1]))
    t = t.append_column("id", pa.array([i for _, i, _ in rows])).append_column(
        "timbre", pa.array([m for _, _, m in rows], pa.bool_())
    ).filter(keep)
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
- PhoMT (`anquachdev/PhoMT-en-vi-speech` at `33400f7`): synthetic speech of PhoMT sentence
  pairs on both sides, limited to rows whose English passed CTC alignment. Its 24 kHz English
  and 48 kHz Vietnamese 16-bit WAV is stored as FLAC, sample for sample. All rows are train.

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


def write_card(root: Path) -> None:
    rows = []
    for split in SPLITS:
        for source in ("cv27", "libriheavy", "phomt"):
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
    (root / "README.md").write_text(
        CARD.replace("{{STATS}}", table).replace("{{CLIP}}", f"{CLIP_SECONDS[0]:g}-{CLIP_SECONDS[1]:g} s")
    )
    log(f"wrote {root / 'README.md'}:\n{table}")


def upload(root: Path, repo: str) -> None:
    HfApi().upload_large_folder(repo, root, repo_type="dataset", allow_patterns=["data/*.parquet", "README.md"],
                                num_workers=UPLOAD_WORKERS)
    log(f"uploaded {root} to {repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("en_speech", "phomt", "card", "upload"))
    p.add_argument("--root", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"))
    p.add_argument("--repo", default=REPO)
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    p.add_argument("--vi-text", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_text"))
    p.add_argument("--workers", type=int, default=4, help="shards built at once (en_speech, phomt)")
    args = p.parse_args()
    with lock(args.root, args.step):
        if args.step == "en_speech":
            move_en_speech(args.src, args.vi_text, args.root, args.workers)
        elif args.step == "phomt":
            build_phomt(args.root, args.workers)
        elif args.step == "card":
            write_card(args.root)
        else:
            upload(args.root, args.repo)


if __name__ == "__main__":
    sys.exit(main())
