# EN-VI speech dataset

Read when touching `training/prep/publish_en_vi_speech.py`, `/Volumes/data/datasets/en_vi_speech`,
the HF dataset `huybik/en-vi-speech` or the cache repo `huybik/hibiki-zero-vi-full-sft`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Building and publishing](#building-and-publishing) · [Cache repo](#cache-repo)

## Purpose

The user asked on 2026-10-23 for one research dataset holding every pair row source: English
speech, its transcript and Vietnamese translation, and Vietnamese speech where it exists. It is
kept in full on the Mac, where it replaced the `en_speech` corpus folder, and published as
`huybik/en-vi-speech` (user's choices). The voice-transfer TTS step will later fill the
Vietnamese speech of the `en_speech` rows; `cache_codes` pair rows are then made from it
([sft-data-caches](sft-data-caches.md)).

## Contents

```text
/Volumes/data/datasets/en_vi_speech/        (laid out as the HF repo, ~480 GB)
  data/{train,validation}/cv27-*, libriheavy-*  ◄─ en_speech step: the corpus, moved in
  data/train/phomt-s*                         ◄─ phomt step: grounded-v2 pairs + PhoMT-en-vi-speech
  README.md                                   ◄─ card step
        └─► upload step (upload_large_folder) ─► huybik/en-vi-speech
```

- Columns: `id`, `en_audio`, `text_en`, `text_vi`, `speaker`, `en_duration_s`, `vi_audio`,
  `vi_duration_s`, `cross_lingual_timbre_matched`; pair rows use the same names.
- `en_speech` rows ([en-speech-corpus](en-speech-corpus.md)) keep their audio bytes and shard
  names, with the exported translations ([en-vi-translation](en-vi-translation.md));
  `vi_audio`, `vi_duration_s` and the timbre flag are empty.
- Common Voice is published too: the user chose this, for research use, knowing the Mozilla
  Data Collective terms forbid re-hosting it.
- PhoMT rows are the grounded-v2 cache rows ([sft-data-caches](sft-data-caches.md#phomt-cache))
  whose English clip lasts 5–22.4 s, the `en_speech` rule (user's choice): 367,064 rows, all
  train, all with both voices. Ids are the cache's (`phomt_s00304r00012`: row 12 of the source
  file the cache numbered 304), and `cross_lingual_timbre_matched` comes from its pairs. Texts
  and durations come from `anquachdev/PhoMT-en-vi-speech` at `33400f7`.
- PhoMT audio is 16-bit mono WAV (English 24 kHz, Vietnamese 48 kHz) re-encoded as FLAC (user's
  choice): bit-identical on decode, ~39% of the WAV size (English 47%, Vietnamese 36%),
  ~0.45 MB a row.

## Building and publishing

- Steps `en_speech`, `phomt`, `card` and `upload`; commands are in
  [training/prep/README.md](../../training/prep/README.md). Each resumes when run again and holds
  a `.<step>.lock` in the dataset folder.
- Every shard is written as `.tmp`, read back and compared, then renamed. `en_speech` deletes an
  original shard only after its dataset shard exists, and removes the emptied corpus folder at
  the end. It first checks that every exported id is in the corpus or already moved.
- `upload` runs `upload_large_folder` on `data/*.parquet` and `README.md`. It keeps its
  resume state in `.cache/huggingface/`, so run it again after adding shards; HF asks for one
  upload process at a time.
- The link measured ~37 MB/s each way (2026-10-23): ~480 GB up and ~513 GB of whole PhoMT
  source files down (kept rows are spread through every row group), about 4 hours each. So the
  `en_speech` move and upload run while `phomt` downloads, and `card` plus a final `upload`
  follow.
- A PhoMT source file takes ~1 minute to download and convert (373 MB in, ~120 MB out). A
  failed file is retried after 1, 2, 4, 8 and 16 minutes, then stops the run.
- pyarrow's default memory pool on macOS (mimalloc) keeps freed buffers: converting 24 shards
  left a 13 GB footprint with nothing allocated, and live runs reached 12–16 GB, pushing swap to
  25 GB. The publisher sets `ARROW_DEFAULT_MEMORY_POOL=system` before importing pyarrow
  (`pa.set_memory_pool` does not reach the Parquet reader), which stays at ~2 GB. `ps` RSS hides
  this, since macOS compresses the idle pages; check `footprint -p <pid>`.
- The first attempt (a custom commit loop, 6 downloads plus a 6 GB commit at once) stalled at
  ~1 MB/s under that memory pressure, so `phomt` runs 3 workers and `upload` 4.
- The account is on the free plan, whose public storage is "best-effort"; if the Hub refuses
  storage, `upload` fails and can be rerun later.

## Cache repo

`huybik/hibiki-zero-vi-full-sft` holds only `phomt-grounded-v2-5s/` (the training cache) and
`grounded-v2/` (its source). On 2026-10-23 the obsolete v1 cache, the root FLEURS cache and
pairs were deleted and the history squashed to one commit (user's choice), leaving 10.4 GB.
