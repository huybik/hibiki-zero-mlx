# English read-speech corpus

Read when touching `training/prep/build_en_speech.py` or `/Volumes/data/datasets/en_speech`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Format](#format) · [Build](#build) · [Rules](#rules)

## Purpose

This is the main training data source: real English speech with gold text, to be turned into
voice-matched VI→EN pair rows by the data phase of
[docs/vi_en_parity_plan.md](../vi_en_parity_plan.md). Its transcripts are being translated to
Vietnamese ([en-vi-translation](en-vi-translation.md)); the TTS step is not built, so no pair
rows exist yet. Those rows would then go through `cache_codes` and train together with the
~0.8k-hour PhoMT cache ([sft-data-caches](sft-data-caches.md#phomt-cache)).

## Contents

`/Volumes/data/datasets/en_speech/` (279 GB) holds `train/` (600 shards) and `validation/`
(3 shards). Its `README.md` is the dataset card with statistics and loading code.

| split | source | rows | hours |
|---|---|---:|---:|
| train | Common Voice 27.0 EN (`cv27`) | 1,728,261 | 2,486.3 |
| train | Libriheavy small + medium + part of large | 1,828,349 | 7,523.9 |
| validation | Common Voice dev | 16,403 | 27.2 |
| validation | Libriheavy dev | 5,014 | 20.9 |

## Format

- Parquet with HF `Audio` feature metadata. Columns: `id`, `audio{bytes,path}`, `text` (cased,
  punctuated), `speaker`, `duration` (float32 seconds).
- Audio keeps the original codec bytes: 48 kHz mono MP3 for CV and 48 kHz mono Ogg Opus for
  Libriheavy. Resample to 24 kHz before Mimi.
- Metadata columns are zstd-compressed and `audio.bytes` is uncompressed. Row groups hold 500
  rows; CV shards hold up to 20,000 rows.
- Libriheavy ids and speakers carry a `libriheavy/` prefix.

## Build

The steps are `cv`, `libriheavy` and `card`, with commands in
[training/prep/README.md](../../training/prep/README.md); each step runs as
`python -m training.prep.build_en_speech <step>` from the repo root in the conda env.

- `cv` calls the Mozilla Data Collective API (dataset `cmu5jplf300nwmh07iqvk9leo`) with
  `MDC_API_KEY`. It downloads in 256 MiB parallel ranges with a resumable `.done` log, verifies
  SHA-256, streams the tar into shards, then deletes the archive.
- CV split: dev becomes validation; train is every other validated clip whose speaker and
  sentence are not in dev. The official CV test set is folded into train, so CV test scores are
  not held out.
- `libriheavy` pulls `mythicinfinity/libriheavy` through `huggingface_hub`, so export `HF_TOKEN`
  from `.env` first. It takes dev, small and medium, then large files in `random.Random(0)` order,
  keeping only their 5–15 s clips, until pooled train holds `--target-hours` (9,200) of 5–15 s
  clips. About 10% of Libriheavy rows are stereo and are dropped by checking OpusHead byte 37
  (`libriheavy-medium-00088` was all stereo, so its shard is empty). Files whose output shard
  exists are skipped. A large file yields ~5.5 h of 5–15 s clips in ~0.18 GB.
- Every source keeps only clips of 5–22.4 s (`CLIP_SECONDS`). 22.4 s is an upper bound: a
  cache row also holds the English delay, U[0, 0.5 × Vietnamese duration], so most 15–22.4 s
  clips will still exceed the trainer's 280 frames. The first 194 large files were written
  before the 5–15 s rule and keep all their 5–22.4 s clips; the card does not say so.
- Shards are written to `.tmp` and then renamed, so an interrupted write never looks finished.

## Rules

- The Mozilla Data Collective terms forbid re-hosting or re-sharing the CV portion, so never
  upload this corpus.
- The corpus must hold ~10k hours of 5–15 s clips, with the PhoMT cache counting toward them,
  so `--target-hours` is 10k minus PhoMT's hours. The user asked for this on 2026-10-22, when
  only 4,372 of the corpus's 10,010 hours were 5–15 s clips. What gets translated is in
  [en-vi-translation](en-vi-translation.md).
- Rows outside 5–22.4 s were deleted from the existing shards rather than rebuilt (user's
  choice), and new large files keep only their 5–15 s clips, because `/Volumes/data` had no room
  for whole files (~0.47 GB each). Bringing the deleted rows back means downloading them again.
