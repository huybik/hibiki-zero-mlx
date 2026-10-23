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
Vietnamese speech of its rows; `cache_codes` pair rows are then made from it
([sft-data-caches](sft-data-caches.md)).

## Contents

```text
/Volumes/data/datasets/en_vi_speech/        (laid out as the HF repo, ~316 GB)
  data/{train,validation}/cv27-*, libriheavy-*  ◄─ en_speech step: the corpus, moved in
  README.md                                   ◄─ card step
        └─► upload step (4 GB commits) ─► huybik/en-vi-speech
```

- Columns: `id`, `en_audio`, `text_en`, `text_vi`, `speaker`, `en_duration_s`, `vi_audio`,
  `vi_duration_s`, `cross_lingual_timbre_matched`; pair rows use the same names.
- The rows are the `en_speech` corpus ([en-speech-corpus](en-speech-corpus.md)): 3.49M rows,
  10,510.6 h, all 5–22.4 s. Shards keep their audio bytes and names and gain the exported
  translations ([en-vi-translation](en-vi-translation.md)); `vi_audio`, `vi_duration_s` and the
  timbre flag are empty.
- Common Voice is published too: the user chose this, for research use, knowing the Mozilla
  Data Collective terms forbid re-hosting it.
- PhoMT is not in the dataset: the user dropped it on 2026-10-24, after 46 of its 1,377 shards
  were built, and none was uploaded. Its training cache stays in the [cache repo](#cache-repo).
  The removed `phomt` step is in git at `7e3b954`.

## Building and publishing

- Steps `en_speech`, `card` and `upload`; commands are in
  [training/prep/README.md](../../training/prep/README.md). Each resumes when run again and holds
  a `.<step>.lock` in the dataset folder.
- Every shard is written as `.tmp`, read back and compared, flushed to the drive with
  `F_FULLFSYNC` (plain fsync stops at the drive's cache), then renamed. Only then does
  `en_speech` delete the original, so a power loss or USB drop can't lose a clip. An original
  that survives a crash is converted again.
- `en_speech` stops if a transcript with a letter has no translation, as dropping the row
  would delete its clip; only the one `"."` row lacks one. It also checks that every exported id
  is in the corpus or already moved. It removes the emptied corpus folder at the end. Run
  `translate_en_speech texts` only after the move: before it, `texts` sees part of the corpus.
- `upload` commits the local shards that are missing from the Hub, or differ from it in size, in
  commits of up to 4 GB (~2 min at the link's speed), then `README.md`. The Hub is the record of
  progress, so a stop loses only the unfinished commit. The user asked on 2026-10-23 that a stop
  never lose more than that: `upload_large_folder` committed nothing in its first 37 min, while
  it hashed and pre-uploaded batches of up to 256 files, and lost them when stopped. `upload`
  looks for new shards after each commit and ends when it finds none, so rerun it after adding
  shards. A failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run.
- `upload` compares sizes only, and never deletes: a shard rewritten at the same size is not
  sent again, and a shard deleted locally stays on the Hub, where the card's
  `data/train/*.parquet` still reads it. After rebuilding or deleting a shard, delete its Hub
  copy with `HfApi().delete_file`.
- Upload alone ran at 30–36 MB/s (2026-10-24). Downloads share the link: with ~27 MB/s coming
  down, the upload fell to ~11 MB/s.
- pyarrow's default memory pool on macOS (mimalloc) keeps freed buffers: converting 24 shards
  left a 13 GB footprint with nothing allocated, and live runs reached 12–16 GB, pushing swap to
  25 GB. The publisher sets `ARROW_DEFAULT_MEMORY_POOL=system` before importing pyarrow
  (`pa.set_memory_pool` does not reach the Parquet reader), which stays at ~2 GB. `ps` RSS hides
  this, since macOS compresses the idle pages; check `footprint -p <pid>`. The user runs
  memory-heavy TTS work on the same Mac, so keep the publisher's memory low; `upload` holds
  ~1.4 GB.
- A launcher that logs `cmd; echo "[$(date +%T)] exited with status $?"` always logs 0, because
  `$(date)` resets `$?`; save `s=$?` first.
- The account is on the free plan, whose public storage is "best-effort"; if the Hub refuses
  storage, the commits fail and `upload` stops after its retries.

## Cache repo

`huybik/hibiki-zero-vi-full-sft` holds only `phomt-grounded-v2-5s/` (the training cache) and
`grounded-v2/` (its source). On 2026-10-23 the obsolete v1 cache, the root FLEURS cache and
pairs were deleted and the history squashed to one commit (user's choice), leaving 10.4 GB.
