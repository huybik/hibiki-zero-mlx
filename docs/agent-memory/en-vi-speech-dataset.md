# EN-VI speech dataset

Read when touching `training/prep/publish_en_vi_speech.py`, `/Volumes/data/datasets/en_vi_speech`,
the HF datasets `huybik/en-vi-speech` and `anquachdev/common-vi-en` or the cache repo
`huybik/hibiki-zero-vi-full-sft`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Contents](#contents) ·
[cv-anquach rows](#cv-anquach-rows) · [Building and publishing](#building-and-publishing) ·
[Cache repo](#cache-repo)

## Purpose

The user asked on 2026-10-23 for one research dataset of pair-row sources: English speech, its
transcript and Vietnamese translation, with the Vietnamese speech to come. It is kept in full on
the Mac, where it replaced the `en_speech` corpus folder (user's choice). The voice-transfer TTS
step ([vi-tts](vi-tts.md)) fills the Vietnamese speech of its shards in place, and
`upload --follow` publishes each as it is voiced; `cache_codes` then builds the training caches
straight from its shards ([sft-data-caches](sft-data-caches.md)).

It was published as `huybik/en-vi-speech` while it was voiced: the Mac and anquachdev's
machine, which voiced bottom up, both pushed there. Once every shard was voiced and the
retranslated rows voiced again (`tts_en_speech --texts`, [vi-tts](vi-tts.md#running)), the whole
dataset went to `anquachdev/common-vi-en` on 2026-10-27, merged into that contributor's voiced
Common Voice set, and from then on only that repository is used (user's plan, 2026-10-24):

```bash
HF_TOKEN=$HF_TOKEN_AN python -m training.prep.publish_en_vi_speech upload --repo anquachdev/common-vi-en
```

On 2026-10-28 both repositories held the 1,842 shards at the Mac's byte sizes
(`anquachdev/common-vi-en` revision `2ce8777`), with none of his old files left.

## Contents

```text
/Volumes/data/datasets/en_vi_speech/        (laid out as the HF repo, ~316 GB + ~120 GB of VI)
  data/{train,validation}/cv27-*, libriheavy-*  ◄─ the corpus, moved in (en_speech step, retired)
                                              ◄─ tts_en_speech: vi_audio filled, shard by shard
  data/train/cv-anquach-*                     ◄─ anquach step, voiced already
  README.md                                   ◄─ card step, and each upload pass
        └─► upload step (4 GB commits; --follow every 10 min) ─► huybik/en-vi-speech
                                                 (all voiced) ─► anquachdev/common-vi-en
```

- Columns: `id`, `en_audio`, `text_en`, `text_vi`, `speaker`, `en_duration_s`, `vi_audio`,
  `vi_duration_s`, `cross_lingual_timbre_matched`; pair rows use the same names.
- The `cv27` and `libriheavy` rows are the `en_speech` corpus
  ([en-speech-corpus](en-speech-corpus.md)): 3.49M rows, 10,510.6 h, all 5–22.4 s. Shards keep their audio bytes and names and gain the exported
  translations ([en-vi-translation](en-vi-translation.md)). The TTS step fills `vi_audio` and
  `vi_duration_s`; the timbre flag is empty until speaker similarity is measured (the parity
  plan's gate).
- Common Voice is published too: the user chose this, for research use, knowing the Mozilla
  Data Collective terms forbid re-hosting it.
- PhoMT is not in the dataset: the user dropped it on 2026-10-24, after 46 of its 1,377 shards
  were built, and none was uploaded. Its training cache stays in the [cache repo](#cache-repo).
  The removed `phomt` step is in git at `7e3b954`.

## cv-anquach rows

`anquachdev/common-vi-en` held, before the merge (revision `dba8e98`), 739,346 Common Voice
English rows (the first ~1,000 h in clip-name order, no length filter), each with a Vietnamese
translation and Vietnamese speech: `en`, `vi`, `audio_en`, `audio_vi`, `source_clip`, both
audio columns 16 kHz mono Ogg Opus at ~16 kbps. The `anquach` step makes them dataset rows
(user's choices, 2026-10-24):

- Kept: English 5–22.4 s, as in the corpus (321,028 rows); the Vietnamese length is free.
- 170,820 of those clips (307 h) are `cv27` rows, all in shards already voiced here
  (`cv27-00000`…`00032` and validation `cv27-00000`); ours stay whole and his copies are dropped.
- Of the other 150,208, the 11,212 whose English sentence is a `cv27` validation sentence,
  compared lower-cased with punctuation and spaces collapsed, are dropped, as `cv27` train drops
  them (plain lower-casing let 126 through), and the 78 whose `vi` holds Chinese, Japanese or
  Korean script (`Thùng啤酒`), spoken from that broken text. The rest go to `train` as
  `cv-anquach-*` shards of 20,000 rows: 138,918 rows, 256.1 h English, 211.6 h Vietnamese
  (2 rows over 22.4 s).
- They are clips `cv27` lacks: dropped by its filters (such as the 10.5 kHz bandwidth floor) or
  not among its validated clips; which, is not recorded. Their English is his 16 kHz re-encode,
  so nothing above 8 kHz; Mimi codes up to 12 kHz. His Vietnamese speech comes from his own TTS,
  not known to match the English voice. `speaker` is empty (his set has
  none, and the Common Voice metadata is not kept), so speaker overlap with `cv27` validation is
  unchecked. `id` is `cv-anquach/<clip>`, and both audio paths are `<clip>.opus`.
- His old files (`data/train-heavy-*.parquet`, the empty `data/train-00000-of-00001.parquet`) are
  deleted by hand from `anquachdev/common-vi-en` after the final upload
  (`HfApi().delete_files(repo, delete_patterns=…)`);
  the step downloads them from the pinned revision, so never squash the repository's history.
  The download sits in `/Volumes/data/datasets/common_vi_en_src` (12 GB) and can be deleted.

## Building and publishing

- Steps `anquach`, `card` and `upload`; commands are in
  [training/prep/README.md](../../training/prep/README.md). Each resumes when run again and holds
  a `.<step>.lock` in the dataset folder.
- The `en_speech` step moved the corpus in, converting each shard and deleting its original;
  with the corpus in, it is removed (in git at `a236ad5`).
- Every shard is written as `.tmp`, read back and compared, flushed to the drive with
  `F_FULLFSYNC` (plain fsync stops at the drive's cache), then renamed, so a power loss or USB
  drop leaves the old shard or the new one, never half of one.
- `upload` commits the local shards that are missing from the Hub, and the voiced ones that differ
  from it in size, in commits of up to 4 GB (2–7 min each, depending on what else shares the
  link), then writes and commits a fresh `README.md`. It lists the Hub before each commit and
  makes the commit on top of the listed revision (`parent_commit`), so a commit landing in
  between fails it (409 or 412) and it lists again at once. Seeding the merged repository went at 15–45 s a
  commit (~110 an hour, above the Hub's rough limit of ~100 an hour, yet none refused): Xet had the
  bytes from `huybik/en-vi-speech` already. The Hub is the record of
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
- An unvoiced shard is committed only when the Hub has none of it (how the merged repository got
  every shard), so an unvoiced local copy never replaces a voiced Hub one.
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
- `upload` writes with `HF_TOKEN`; for `anquachdev/common-vi-en`, set it to `HF_TOKEN_AN`, a
  write token of that account ([environment](environment.md)).
- Free-plan public storage is "best-effort" (huybik's plan; anquachdev's is not recorded); if
  the Hub refuses storage, the commits fail and `upload` stops after its retries.

## Cache repo

`huybik/hibiki-zero-vi-full-sft` (a dataset) holds the training caches: `phomt-grounded-v2-5s/`
with `grounded-v2/` (its source), and since 2026-10-28 the caches of this dataset,
`en-vi-grounded-v2/` and `en-vi-grounded-v2-cv-anquach/`, committed shard by shard as they are
built ([sft-data-caches](sft-data-caches.md#publishing)). On 2026-10-23 the obsolete v1 cache,
the root FLEURS cache and pairs were deleted and the history squashed to one commit (user's
choice), leaving 10.4 GB. The model repository of the same name holds training runs, not
caches.
