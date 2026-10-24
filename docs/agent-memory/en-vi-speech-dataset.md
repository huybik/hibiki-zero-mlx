# EN-VI speech dataset

Read when touching `training/prep/publish_en_vi_speech.py`, `/Volumes/data/datasets/en_vi_speech`,
the HF dataset `huybik/en-vi-speech` or the cache repo `huybik/hibiki-zero-vi-full-sft`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[Building and publishing](#building-and-publishing) · [Cache repo](#cache-repo)

## Purpose

The user asked on 2026-10-23 for one research dataset of pair-row sources: English speech, its
transcript and Vietnamese translation, with the Vietnamese speech to come. It is kept in full on
the Mac, where it replaced the `en_speech` corpus folder, and published as `huybik/en-vi-speech`
(user's choices). The voice-transfer TTS step ([vi-tts](vi-tts.md)) fills the Vietnamese speech
of its shards in place, and `upload --follow` publishes each as it is voiced; `cache_codes` pair
rows are then made from it ([sft-data-caches](sft-data-caches.md)).

## Contents

```text
/Volumes/data/datasets/en_vi_speech/        (laid out as the HF repo, ~316 GB + ~120 GB of VI)
  data/{train,validation}/cv27-*, libriheavy-*  ◄─ the corpus, moved in (en_speech step, retired)
                                              ◄─ tts_en_speech: vi_audio filled, shard by shard
  README.md                                   ◄─ card step, and each upload pass
        └─► upload step (4 GB commits; --follow every 10 min) ─► huybik/en-vi-speech
```

- Columns: `id`, `en_audio`, `text_en`, `text_vi`, `speaker`, `en_duration_s`, `vi_audio`,
  `vi_duration_s`, `cross_lingual_timbre_matched`; pair rows use the same names.
- The rows are the `en_speech` corpus ([en-speech-corpus](en-speech-corpus.md)): 3.49M rows,
  10,510.6 h, all 5–22.4 s. Shards keep their audio bytes and names and gain the exported
  translations ([en-vi-translation](en-vi-translation.md)). The TTS step fills `vi_audio` and
  `vi_duration_s`; the timbre flag is empty until speaker similarity is measured (the parity
  plan's gate).
- Common Voice is published too: the user chose this, for research use, knowing the Mozilla
  Data Collective terms forbid re-hosting it.
- PhoMT is not in the dataset: the user dropped it on 2026-10-24, after 46 of its 1,377 shards
  were built, and none was uploaded. Its training cache stays in the [cache repo](#cache-repo).
  The removed `phomt` step is in git at `7e3b954`.

## Building and publishing

- Steps `card` and `upload`; commands are in
  [training/prep/README.md](../../training/prep/README.md). Each resumes when run again and holds
  a `.<step>.lock` in the dataset folder.
- The `en_speech` step moved the corpus in, converting each shard and deleting its original;
  with the corpus in, it is removed (in git at `a236ad5`).
- Every shard is written as `.tmp`, read back and compared, flushed to the drive with
  `F_FULLFSYNC` (plain fsync stops at the drive's cache), then renamed, so a power loss or USB
  drop leaves the old shard or the new one, never half of one.
- `upload` commits the voiced local shards that are missing from the Hub, or differ from it in
  size, in commits of up to 4 GB (2–7 min each, depending on what else shares the link), then writes and
  commits a fresh `README.md`. The Hub is the record of
  progress, so a stop loses only the unfinished commit. The user asked on 2026-10-23 that a stop
  never lose more than that: `upload_large_folder` committed nothing in its first 37 min, while
  it hashed and pre-uploaded batches of up to 256 files, and lost them when stopped. `upload`
  looks for new shards after each commit and ends when it finds none. With `--follow` (the
  user asked on 2026-10-24 that the TTS output go up continuously) it lists the Hub again every
  10 min until every shard is voiced, so start it beside the TTS run; a TTS shard is voiced
  every ~3 min.
  A failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run. It also stops
  when `data/` holds no shards (a dropped drive or a wrong `--root`), rather than end as if
  every shard were voiced.
- Unvoiced shards are never committed, so an unvoiced local copy never replaces a voiced Hub one.
  A Hub shard whose size differs from an unvoiced local shard was voiced by the other machine
  ([vi-tts](vi-tts.md#running)): after its commits, `upload` downloads it into `.pull/` and puts
  it in place of the local copy (~20 s for 280 MB) only if its schema matches, every column but
  `vi_audio` and `vi_duration_s` equals the local one, and every row has Vietnamese audio bytes
  and a positive duration. Comparing the `en_audio` columns adds ~1 GB for those ~20 s. A shard
  the TTS voiced here meanwhile is kept, and goes up in the next pass.
- A rejected download stops the run, and every rerun stops on it again after its commits, while
  the TTS keeps voicing. Have the other machine commit a fixed copy (a new version is downloaded
  afresh), or put the local copy back with `HfApi().upload_file` for the TTS here to voice; then
  `rm -r .pull` and rerun.
- A voiced shard sent again costs only its new bytes: the Hub's Xet storage chunks files by
  content, and the rewritten shard keeps the English columns' bytes. The first 4 GB commit of
  voiced shards sent 1.15 GB (2026-10-24), about their Vietnamese audio.
- `upload` compares sizes only, and never deletes: a shard rewritten at the same size is not
  sent again, and a shard deleted locally stays on the Hub, where the card's
  `data/train/*.parquet` still reads it; delete its Hub copy with `HfApi().delete_file`.
- A kill in the middle of a shard rewrite can leave its `.parquet.tmp` (200–300 MB), which no
  step reads; delete it while the TTS is stopped. A killed download leaves its partial file in
  `.pull/` until the next pull clears it.
- huggingface_hub's progress bars are disabled in `upload`: in a log, Xet's redraws came to
  ~1 MB a commit.
- Upload alone ran at 30–36 MB/s (2026-10-24). Downloads share the link: with ~27 MB/s coming
  down, the upload fell to ~11 MB/s.
- pyarrow's default memory pool on macOS (mimalloc) keeps freed buffers: converting 24 shards
  left a 13 GB footprint with nothing allocated, and live runs reached 12–16 GB, pushing swap to
  25 GB. The publisher sets `ARROW_DEFAULT_MEMORY_POOL=system` before importing pyarrow
  (`pa.set_memory_pool` does not reach the Parquet reader), which stays at ~2 GB. `ps` RSS hides
  this, since macOS compresses the idle pages; check `footprint -p <pid>`. The user runs
  memory-heavy TTS work on the same Mac, so keep the publisher's memory low; `upload` holds
  ~1.4 GB, ~2.5 GB during a pull.
- A launcher that logs `cmd; echo "[$(date +%T)] exited with status $?"` always logs 0, because
  `$(date)` resets `$?`; save `s=$?` first.
- The account is on the free plan, whose public storage is "best-effort"; if the Hub refuses
  storage, the commits fail and `upload` stops after its retries.

## Cache repo

`huybik/hibiki-zero-vi-full-sft` holds only `phomt-grounded-v2-5s/` (the training cache) and
`grounded-v2/` (its source). On 2026-10-23 the obsolete v1 cache, the root FLEURS cache and
pairs were deleted and the history squashed to one commit (user's choice), leaving 10.4 GB.
