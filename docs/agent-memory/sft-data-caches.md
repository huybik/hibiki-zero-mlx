# SFT grounded caches

Read when touching `training/prep/cache_codes.py`, `training/prep/publish_caches.py` or
`training/paths.py`: the EN-VI dataset's caches, their builds and publishing, the PhoMT cache and
the row layout every cache shares. How rows are built and the models run, on either device, is in
[cache-build](cache-build.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [Builds](#builds) ·
[Publishing](#publishing) · [PhoMT cache](#phomt-cache) · [Row layout](#row-layout)

## Flow

```text
/Volumes/data/datasets/en_vi_speech/data/<split>/<name>.parquet
  ─► cache_codes (--device mlx on the Mac, or cuda) ─► <out-dir>/<split>/shard_<name>.pt
       read: every row, no gates of its own; the build is cache_build's (cache-build)
  ─► publish_caches ─► HF cache repo <out-dir name>/<split>/shard_<name>.pt ◄─ second machine
HF <prefix>/ ─► data/cache/<prefix> on the pod (docs/finetune.md)
<prefix>/train + phomt-grounded-v2-5s ─► training.finetune.train --cache-dir (several)
<prefix>/validation ─► --val-cache-dir
```

- The trainer runs on the H100 pod; the caches reach it through the cache repo
  ([Publishing](#publishing), [docs/finetune.md](../finetune.md)). `CACHE_FORMAT` lives in
  `training/paths.py` so the trainer never imports the build.

- The source is the EN-VI dataset ([en-vi-speech-dataset](en-vi-speech-dataset.md)), read
  straight from its parquet shards, the split taken from the folder. FLEURS is dropped.
- `--shards` picks `.parquet` source shards by glob under `data/` (default: every train and
  validation shard, `cv-anquach` included). Put `--out-dir` on `/Volumes/data`: the codes come to
  ~70 GB and the internal disk is too small.
- The full build (user's choice, 2026-10-27: every row that fits the trainer) goes to two
  out-dirs, see [Builds](#builds).
- Grounded-v2 (`hibiki_vn_grounded_cache_v2`) is the only cache format; `streams.load_rows`
  rejects anything else:
  - Mimi at 24 kHz / 12.5 Hz;
  - English audio delayed by a draw from U[0, min(0.5 × Vietnamese duration, room)], seeded by
    `1234:{split}:{id}` with the duration rounded to the 0.01 s it stores, and inserted as left
    silence. `room` is what the English clip leaves of the frame cap, so a long clip gets a
    shorter delay rather than being dropped (chosen 2026-10-27 for the full build: it keeps 197k
    rows, 1,033 English hours). Where the room does not bind, the delay equals the old rule's; on
    `libriheavy-dev-00001` 1,040 rows came out byte-identical, 349 got a shorter delay and 299
    more were kept (and 8 with leading symbols). Over the main build the room binds on 443k rows,
    ~246k of which the old rule kept with a longer delay. PhoMT's cache was filtered under the old rule (rows over the cap dropped);
  - English text word-timed by CTC alignment with `facebook/wav2vec2-base-960h`; rows that
    fail alignment or score below 0.5 are rejected.
- `training/paths.py` holds the repo defaults (`weights/` artifacts), file guards, `durable` and
  `cpu_budget`.
- A sample also holds `text_vi` and `cross_lingual_timbre_matched`; a shard's `source` holds the
  file and its byte size. A rerun skips shards that exist, without checking their source again;
  `--overwrite` rebuilds. A lock (`.cache_codes.lock`) keeps a second run off the same
  `--out-dir`. Run a multi-day build under `caffeinate -i -s`.

## Builds

The full build, started 2026-10-27 and restarted on the faster builder on 2026-10-28 after the
first 68 `cv27` shards, ran the two out-dirs one after the other, each under
`caffeinate -i -s` with its log in `<out-dir>/build.log`, and finished on 2026-11-02 (main at
19:05, `cv-anquach` at 21:42, both with status 0). Every shard is on the Hub. It holds ~22.1k h of
Mimi audio (21,558 h main, 521 h `cv-anquach`) and ran 150–170x with the Mac in use (200–225x
idle), ~5.5 days on the Mac alone.

| Out-dir on `/Volumes/data/datasets/` | `--shards` | Source rows | Built |
|---|---|---|---|
| `en-vi-grounded-v2` | `train/cv27-*`, `train/libriheavy-*`, `validation/*` | 3.49M, 10,510 h English | 1,835 shards, 3.45M rows (1.1% rejected), 73 GB |
| `en-vi-grounded-v2-cv-anquach` | `train/cv-anquach-*` | 139k, 256 h English | 7 shards, 136k rows (2.2% rejected), 1.8 GB |

- `build.log` has no line for three main shards (`libriheavy-large-00001-00031`, `-00033`,
  `-00036`): the 2026-10-28 replug below lost them while the builder kept building. Their
  shards match their neighbours' layout and hold 2,783 rows.
- To stop it, `pkill -f training.prep.cache_codes`: the launch script then exits with status 143
  and does not start `cv-anquach`, but only once the two orphaned decode workers
  (`multiprocessing.spawn` and its `resource_tracker`, parent pid 1) are killed too, since they
  hold its log pipe open. Stop the publisher the same way (`training.prep.publish_caches`) after its
  log shows `0 left`; it logs only commits and retries, so a quiet log means nothing is waiting.
- Nothing restarts it on its own, after a stop, a logout or a restart (a macOS update logs out).
  Run from the repo root on `main` `nohup bash <main out-dir>/launch.sh >/dev/null 2>&1 &`, and
  the same for `publish.sh` if it stopped; the builder skips the built shards and rebuilds the
  unfinished one.
- `cv-anquach` is kept apart because its English targets are 16 kHz
  ([en-vi-speech-dataset](en-vi-speech-dataset.md#cv-anquach-rows)): pass its `train/` as one
  more `--cache-dir` to use it.
- `publish.sh` beside `launch.sh` in the main out-dir runs `publish_caches --follow` on the two
  out-dirs in the same order, logging to `<out-dir>/publish.log`.
- Both scripts append each log line on its own: unplugging `/Volumes/data` kills a log a
  process holds open. On 2026-10-28 a replug lost the builder's lines while it kept building, and
  `publish_caches` died on its next one (exit 120: Python could not write its stdout). A process
  reading the drive while it is gone can still fail, so stop the build before unplugging.
- Keep the Mac on AC: `caffeinate -s` holds only on AC, and on 2026-10-29 the build ran on battery
  until Low Power Sleep at 1% (23:49), losing 7 h. A process started from a terminal reads
  `/Volumes/data` under that terminal app's privacy grant (Removable Volumes). On 2026-10-30
  Ghostty 1.3.1 kept running after 1.4.0 replaced it, its old bundle moved and half deleted;
  after the wake macOS could not verify it and refused every drive read from it ("Operation not
  permitted", also on `ls`), so the builder died. Restoring the bundle's files from the official
  1.3.1 image let reads through again an hour later; quitting and reopening the terminal also fixes it.
- Frame-cap survey of every source row (2026-10-27): 21.4k rows (111 English hours) have a
  Vietnamese clip over 22.32 s and 454 an English one, and are rejected; everything else fits.
  CTC rejections come on top: ~0.6% on `libriheavy-dev-00001`, ~0.4% on `cv27`.

## Publishing

The user asked on 2026-10-28 that the caches go to the cache repo while they are built, and that
anquachdev can build part of them, as the two machines voiced the dataset
([vi-tts](vi-tts.md#running)):

```text
train/cv27-00000 ... ─► Mac, cache_codes top down
                ... meet ...
... validation/*     ◄─ second machine, cache_codes --bottom-up
both: publish_caches --follow ─► huybik/hibiki-zero-vi-full-sft (dataset) <out-dir name>/<split>/
```

- The Hub prefix is the out-dir's name (`en-vi-grounded-v2/`, `en-vi-grounded-v2-cv-anquach/`),
  so the second machine names its out-dir the same. Only the source shards matching `--shards`
  count; the publisher requires the builder's `--shards`, since the default globs would put
  `cv-anquach` shards under the main prefix, where the pod's restore takes them as main rows.
- A pass lists the Hub and commits the local shards it lacks, up to 4 GB a commit (~60 shards of
  ~65 MB), on top of the listed revision (`parent_commit`, as the dataset publisher): a commit
  landing in between fails with 409 or 412 and it lists again. The Hub is the record: a shard
  there is never sent again, so a stop loses only the unfinished commit. Failed commits are
  retried after 1, 2, 4, 8 and 16 minutes, then stop the run.
- Then it downloads, into `<out-dir>/.pull/`, each Hub shard missing here and puts it in place,
  so the builder, which checks for a shard as it reads it (a shard and a window ahead of the
  build), skips it. Each machine so ends with
  the whole cache (the second machine downloads the Mac's part too). A pulled shard must have
  the settings of a shard already in the out-dir (format, rates, `max_frames`,
  `alignment_min_score`, model config, `direction` and `tokenizer`, which a shard built before
  2026-11-10 lacks and is read as `vi-en` on the base tokenizer; pulls wait until one exists),
  the same source file at the
  same byte size, and every source row once, kept or rejected; otherwise the run stops with it
  left in `.pull/`, and every rerun stops on it again. Then fix the cause, or delete the bad
  shard on the Hub (`HfApi().delete_file`) for this machine to build; then `rm -r .pull` and
  rerun. Row contents are not checked, nor the delay seed and ratio, which a shard does not
  record: both machines run `main` with the default settings, and the second checks once that it
  rebuilds a Mac shard ([training/prep/README.md](../../training/prep/README.md)).
- Near the meeting point a few shards may be built on both machines: each sees the other's work
  only after its commit and the 10 min wait. The Hub keeps the copy committed first; the other
  machine's local copy may be its own build, the same rows up to float near-ties. The pod
  trains on the Hub's.
- With `--follow` it looks again every 10 min and ends once every source shard is cached here
  and on the Hub. It stops when the out-dir is gone (a dropped drive), rather than download onto
  the internal disk.
- Each direction's aligner is pinned (`text_timing.ALIGNERS`: `facebook/wav2vec2-base-960h` at
  `22aad52` for English text, omniASR for Vietnamese, [two-way](two-way.md#vietnamese-aligner)),
  its weights, `config.json` and `vocab.json` only, so every machine times with the same model. The builder no longer sets `HF_HUB_OFFLINE`, so a
  machine's first run can download them; offline it uses the cached copy (0.4 s on a refused
  connection, ~75 s on a hanging one).
- The second machine needs Apple Silicon, or a CUDA GPU with `--device cuda` (its rows differ
  from the Mac's by float near-ties, [cache-build](cache-build.md#across-devices)), and write
  access to the cache repo; commands are in [training/prep/README.md](../../training/prep/README.md).

## PhoMT cache

- Pooled with the `en_speech` caches by passing it as one more `--cache-dir`. It holds 804 hours
  of English speech (~770 h of them 5–15 s clips), on top of the corpus's 10,486 train hours
  ([en-speech-corpus](en-speech-corpus.md#rules)).
- It is the HF dataset `huybik/hibiki-zero-vi-full-sft` prefix `phomt-grounded-v2-5s/`: 1,377
  `shard_*.pt` (6.5 GB) holding 361,968 `train` rows, restored as in
  [docs/finetune.md](../finetune.md). The Mac copy is
  `/Volumes/data/datasets/phomt-grounded-v2-5s/`.
- It follows `en_speech`'s clip rule (user's choice, 2026-10-23): the rows of the full
  `grounded-v2/` cache in the same repo (690,067 rows, `cache_chunk_{0..7}.tar.zst` with
  `pairs_w*.jsonl` and `manifest.json`) whose English clip lasts at least 5 s
  (`en_duration_s`) and whose codes fit 280 frames. Kept rows, shard names and metadata are
  unchanged.
- `grounded-v2/` was built from `anquachdev/PhoMT-en-vi-speech` (renamed
  `anquachdev/en-vi-speech-20k` on 2026-11-06; the old name redirects) at revision `33400f7` (565 GB of
  TTS audio on both sides) by the since-removed `cache_phomt_stream.py`, so it is never
  rebuilt. Filter from it if the rule changes. A row's cached English length counts the
  inserted delay; `en_frames - target_delay_frames` is within a frame of the clip.
- 190,911 of the kept rows carry `cross_lingual_timbre_matched`.

## Row layout

- Codes are int32 `[33, T]`:
  - row 0 is text;
  - rows 1–16 are target Mimi (English in a `vi-en` shard, every one built before 2026-11-10);
  - rows 17–32 are source Mimi (Vietnamese there; the languages trade places in an `en-vi`
    shard, [two-way](two-way.md)).
- Model config: `n_q` 32, `dep_q` 16, `card` 2048, `text_card` 48000, text pad id 3.
- Text stream: PAD fill, tokens at their CTC frames, then tokenizer EOS at the frame equal to
  the target code length, then -1.
- Target audio is -1 after its length, with no explicit EOS.
- Source audio has codec EOS 2048 at `source_len`, then -1.
- T = max(text end, target length, source length + 1).
- The trainer joins rows into multi-sentence streams without changing the cache
  ([sft-training](sft-training.md#trainer)); batches are right-padded with -1 to a multiple of
  the frame bucket. It reads the rows from packs it derives beside the shards
  (`<cache-dir>/.packed/`, 70 GB for `en-vi-grounded-v2/train`).
- Checked on the pod after download (2026-11-03): all 1,835 `en-vi-grounded-v2` shards match
  the Hub's SHA-256; the 3,441,391 train and 9,313 validation rows all have this layout (int32,
  T = max(en, vi) + 1, PAD before the delay, one tokenizer EOS value, codes in range); no id
  repeats; every planned stream assembles.
