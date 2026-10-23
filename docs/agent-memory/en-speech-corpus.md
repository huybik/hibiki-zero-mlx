# English read-speech corpus

Read when touching `training/prep/build_en_speech.py` or `/Volumes/data/datasets/en_speech`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Format](#format) · [Build](#build) · [Quality filters](#quality-filters) · [Rules](#rules)

## Purpose

This is the main training data source: real English speech with gold text, to be turned into
voice-matched VI→EN pair rows by the data phase of
[docs/vi_en_parity_plan.md](../vi_en_parity_plan.md). Its transcripts are being translated to
Vietnamese ([en-vi-translation](en-vi-translation.md)); the TTS step is not built, so no pair
rows exist yet. Those rows would then go through `cache_codes` and train together with the
~0.8k-hour PhoMT cache ([sft-data-caches](sft-data-caches.md#phomt-cache)).

## Contents

`/Volumes/data/datasets/en_speech/` (294 GB) holds `train/` (2,094 shards, 262 of them empty
markers) and `validation/` (3 shards). Its `README.md` is the dataset card with statistics and
loading code.

| split | source | rows | hours | 5–15 s hours | speakers |
|---|---|---:|---:|---:|---:|
| train | Common Voice 27.0 EN (`cv27`) | 522,251 | 943.5 | 942.6 | 42,680 |
| train | Libriheavy small + medium + 1,688 of 2,955 large files | 2,958,316 | 9,542.2 | 7,217.6 | 4,459 |
| validation | Common Voice dev | 6,021 | 11.6 | 11.5 | 3,188 |
| validation | Libriheavy dev | 3,501 | 13.3 | 6.0 | 125 |

## Format

- Parquet with HF `Audio` feature metadata. Columns: `id`, `audio{bytes,path}`, `text` (cased,
  punctuated), `speaker`, `duration` (float32 seconds).
- Audio keeps the original codec bytes: 32 or 48 kHz mono MP3 for CV and 48 kHz mono Ogg Opus for
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
  from `.env` first. It takes dev, medium and small, then large files in `random.Random(0)` order
  until pooled train holds `--target-hours` (9,200). About 10% of Libriheavy rows are stereo and
  are dropped by checking OpusHead byte 37. Files whose output shard exists are skipped, and
  shards emptied by the filters stay as markers. A downloaded large file (~0.47 GB) adds ~8.3 h
  after the filters (3.9 h of 5–15 s clips measured × 2.16, the first 194 files' ratio).
- Every source keeps only clips of 5–22.4 s (`CLIP_SECONDS`). 22.4 s is an upper bound: a
  cache row also holds the English delay, U[0, 0.5 × Vietnamese duration], so most 15–22.4 s
  clips will still exceed the trainer's 280 frames. Large files 195–1,688 in build order keep
  only 5–15 s clips (an earlier target); the card does not say so.
- Shards are written to `.tmp` and then renamed, so an interrupted write never looks finished.

## Quality filters

- Bandwidth uses HiFiTTS-2's estimate: the highest frequency whose Welch power is within 50 dB
  of the peak, 0 for silence. It is taken per recording: a Libriheavy chapter's first kept clips
  in one source file (clip id minus `_<segment>`) totalling ≥30 s, or a whole CV clip. Single
  clips of one chapter scatter ±1.5 kHz around the chapter estimate.
- Recordings under 10.5 kHz are dropped. Mimi codes up to 12 kHz. CV's 32 kHz MP3s cut off at
  ~11 kHz (a spike at 10.7–11.3 kHz) that an 11 kHz floor would split. Before the filter,
  42% of CV train hours and 17% of Libriheavy's were under 10.5 kHz.
- Train keeps a speaker's clips in build order (CV archive order, then Libriheavy medium, small
  and large) until the speaker holds 30 h (`SPEAKER_HOURS`); no CV speaker reaches it.
  A Libriheavy file holds one or a few readers, so a train file whose parquet footer
  (per-row-group speaker statistics, ~3 s to read) shows only capped readers gets an empty
  shard without downloading.
- The cap came from a simulation over the measured corpus and the footers of all 2,955 large
  files, counting 5–15 s hours as the target then did. Without it, 3,642 Libriheavy readers
  fill 9,200 h and one holds 261 h. 30 h left ~490 large files to spare; 20 h couldn't reach
  9,200 h.

## Rules

- The Mozilla Data Collective terms forbid re-hosting or re-sharing the CV portion, so never
  upload this corpus.
- Training gets ~10k hours of 5–22.4 s clips, PhoMT included, so `--target-hours` is 10k minus
  PhoMT's hours. The user first asked for 5–15 s clips; on 2026-10-23 they chose 5–22.4 s and
  stopped the top-up at 10,486 train hours (8,160 of them 5–15 s), knowing the trainer drops
  rows over 280 frames. Translation scope is in [en-vi-translation](en-vi-translation.md).
- Rows outside 5–22.4 s were deleted from the existing shards rather than rebuilt (user's
  choice), and large files keep only their clips, as `/Volumes/data` has no room for whole
  files. The quality filters were applied the same way on 2026-10-23 (1.51M rows). Bringing
  deleted rows back means downloading them again.
- Train clip `cv27/common_voice_en_20040849` (9.4 s) was deleted on 2026-10-23, and the same
  sentence was removed from the translation selection. Sonnet returns an empty refusal for it,
  which stops `translate` ([en-vi-translation](en-vi-translation.md)). Rebuilding Common Voice
  from the archive would put the clip back.
- No other corpus is mixed in (user's choice, 2026-10-23, wanting many speakers, good
  transcripts and clean recordings). The English audio is a generated target, so 16 kHz
  corpora (LibriSpeech, GigaSpeech, People's Speech, VoxPopuli, Loquacious Set) lack the
  8–12 kHz band; upsampling adds none. Loquacious text has no case or punctuation, TED-LIUM 3
  is CC BY-NC-ND, GigaSpeech's HF terms are non-commercial, and HiFiTTS-2 is Libriheavy's own
  LibriVox audio. Emilia-YODAS English (CC BY, 24 kHz, spontaneous, unchecked Whisper-medium
  text) is the candidate for a spontaneous domain, with its own validation, once the first
  1k-hour tranche has trained.
