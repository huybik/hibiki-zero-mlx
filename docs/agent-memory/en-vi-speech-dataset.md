# EN-VI speech dataset on Hugging Face

Read when touching `training/prep/publish_en_vi_speech.py`, the HF dataset
`huybik/en-vi-speech` or the cache repo `huybik/hibiki-zero-vi-full-sft`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Publishing](#publishing) · [Cache repo](#cache-repo)

## Purpose

The user asked on 2026-10-23 for one public research dataset holding every pair row source:
English speech, its transcript and Vietnamese translation, and Vietnamese speech where it exists.
The voice-transfer TTS step will later fill the Vietnamese speech of the `en_speech` rows.
`cache_codes` pair rows are then made from it ([sft-data-caches](sft-data-caches.md)).

## Contents

```text
en_speech (CV + Libriheavy audio, text) ─┐
en_speech_vi_text/{split}.parquet ───────┼─► publish_en_vi_speech ─► huybik/en-vi-speech
grounded-v2 pairs + PhoMT-en-vi-speech ──┘     data/{train,validation}/{cv27,libriheavy-*,phomt-*}.parquet
```

- Columns: `id`, `en_audio`, `text_en`, `text_vi`, `speaker`, `en_duration_s`, `vi_audio`,
  `vi_duration_s`, `cross_lingual_timbre_matched`; pair rows use the same names.
- `en_speech` rows ([en-speech-corpus](en-speech-corpus.md)) keep their audio bytes and shard
  names, with the exported translations ([en-vi-translation](en-vi-translation.md));
  `vi_audio`, `vi_duration_s` and the timbre flag are empty. The one train row without a
  translation (no letters) is left out.
- Common Voice is published too: the user chose this, for research use, knowing the Mozilla
  Data Collective terms forbid re-hosting it.
- PhoMT rows are the grounded-v2 cache rows ([sft-data-caches](sft-data-caches.md#phomt-cache))
  whose English clip lasts 5–22.4 s, the `en_speech` rule (user's choice): about 367k rows, all
  train, all with both voices. Ids are the cache's (`phomt_s00304r00012`: row 12 of the source
  file the cache numbered 304), and `cross_lingual_timbre_matched` comes from its pairs. Texts
  and durations come from `anquachdev/PhoMT-en-vi-speech` at `33400f7`.
- PhoMT audio is 16-bit mono WAV (English 24 kHz, Vietnamese 48 kHz) re-encoded as FLAC (user's
  choice): bit-identical on decode, ~39% of the WAV size (English 47%, Vietnamese 36%),
  ~0.45 MB a row.

## Publishing

- Steps `en_speech`, `phomt` and `card`; commands are in
  [training/prep/README.md](../../training/prep/README.md). Export `HF_TOKEN` from `.env` first.
- The link measured ~37 MB/s each way (2026-10-23), so the run is network-bound: ~475 GB up
  (~315 GB `en_speech`, ~165 GB PhoMT) and ~513 GB of whole PhoMT source files down, about
  4 hours each. Kept rows are spread through every row group, so partial reads save nothing.
- So the two steps run at once, and each commits from a background thread, 25 shards a
  commit, up to 8 batches behind. Builds (PhoMT downloads) never wait on uploads, and staging
  (`/Volumes/data/datasets/en_vi_speech_upload/<step>`) peaks at ~70 GB for `en_speech` and
  ~30 GB for PhoMT.
- A rerun skips shards already in the repo and empties the step's staging folder first; a lock
  beside it stops a second run of the same step. A failed build or commit is retried after 1,
  2, 4, 8 and 16 minutes, then stops the run.
- `en_speech` first checks that every exported id is in the corpus, so a stale export stops it.
- A PhoMT source file takes ~1 minute to download and convert (373 MB in, ~120 MB out); every
  file keeps rows.
- `card` reads the statistics back from the repo, so run it last.
- The account is on the free plan, whose public storage is "best-effort"; if the Hub refuses
  storage, commits fail after their retries and the run stops resumable.
- Filling `vi_audio` later means rewriting the `en_speech` shards; keep their row groups so Xet
  deduplicates the unchanged English audio.

## Cache repo

`huybik/hibiki-zero-vi-full-sft` holds only `phomt-grounded-v2-5s/` (the training cache) and
`grounded-v2/` (its source). On 2026-10-23 the obsolete v1 cache, the root FLEURS cache and
pairs were deleted and the history squashed to one commit (user's choice), leaving 10.4 GB.
