# SFT grounded caches

Read when touching `training/prep/cache_codes.py`, `training/prep/publish_caches.py`,
`training/prep/mimi_mlx.py`, `training/prep/wav2vec2_mlx.py`, `training/prep/text_timing.py` or
`training/paths.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [Builder](#builder) · [Speed](#speed) · [Builds](#builds) ·
[Publishing](#publishing) · [PhoMT cache](#phomt-cache) · [Row layout](#row-layout)

## Flow

```text
/Volumes/data/datasets/en_vi_speech/data/<split>/<name>.parquet
  ─► cache_codes (Mac, MLX) ─► <out-dir>/<split>/shard_<name>.pt
       workers: PyAV decode; soxr to 24 kHz (Mimi) and 16 kHz (CTC)
       gates: decoded duration ─► frame cap ─► text groups ─► wav2vec2 CTC (text_timing)
       GPU: CTC (wav2vec2_mlx), then Mimi (mimi_mlx); the CPU Viterbi runs while Mimi encodes
  ─► publish_caches ─► HF cache repo <out-dir name>/<split>/shard_<name>.pt ◄─ second machine
HF <prefix>/ ─► data/cache/<prefix> on the pod (docs/finetune.md)
<prefix>/train + phomt-grounded-v2-5s ─► training.finetune.train --cache-dir (several)
<prefix>/validation ─► --val-cache-dir
```

- The trainer runs on the H100 pod; the caches reach it through the cache repo
  ([Publishing](#publishing), [docs/finetune.md](../finetune.md)). `CACHE_FORMAT` lives in
  `training/paths.py` so the trainer never imports the builder's Mac-only packages (PyAV, MLX,
  soxr), which the pod does not install.

- The source is the EN-VI dataset ([en-vi-speech-dataset](en-vi-speech-dataset.md)), read
  straight from its parquet shards, the split taken from the folder. FLEURS is dropped.
- `--shards` picks `.parquet` source shards by glob under `data/` (default: every train and
  validation shard, `cv-anquach` included). Put `--out-dir` on `/Volumes/data`: the codes come to
  ~70 GB and the internal disk is too small.
- The full build (user's choice, 2026-10-27: every row that fits the trainer) goes to two
  out-dirs, see [Builds](#builds).
- Grounded-v2 (`hibiki_vn_grounded_cache_v2`) is the only cache format; `CachedCodeDataset`
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
- `training/paths.py` holds the repo defaults (`weights/` artifacts), file guards and `durable`.

## Builder

- Gates run cheapest first, and a row is rejected, with its reason, at the first one it fails:
  a decoded clip more than 0.02 s off its stored duration (PyAV decodes a truncated or damaged
  clip without error, only shorter); more frames than `--max-frames` (280, the trainer's cap),
  counted from the decoded lengths as max(English frames, Vietnamese frames) + 1 before the delay;
  the text grouping; CTC alignment; token timing. Mimi is queued with the CTC forward, so the
  ~0.5% of rows rejected at alignment or token timing are encoded and dropped.
- Text grouping: SentencePiece pieces group into words, and a group with nothing to speak (a
  stray `*`, `-` or `"`; a lone `'` counts as spoken) joins the word before it. Before the first
  word it is the lead, timed on the frames just before that word's start, so the word keeps its
  own frame; when the lead does not fit after the delay it starts at the delay and pushes the
  word later. An unspoken piece inside a word (`"It`, `(see`) is spread over that word's span
  with it, as before (half of Libriheavy rows; kept unchanged in this build). Leading ones were rejected before 2026-10-27 (51k Libriheavy
  rows, 1.7%).
- Only data faults reject a row: `cache_codes.Rejected` (undecodable or non-mono audio, a duration
  mismatch, the frame cap) and `text_timing.AlignmentError`. Any other error, such as a dead
  decode worker (`BrokenProcessPool`), an MLX error or PyAV running out of memory, stops the run
  before the shard is written, so a rerun rebuilds it rather than skipping lost rows. A shard
  whose rows all fail on data is written, empty (some source shards hold only 3 rows).
- A shard holds `samples` (the row fields plus `source_row`, `data/<split>/<file>#r<index>`),
  `rejected` (`id`, `reason`), `source` (file, bytes) and `max_frames`. Its values are Python
  types and tensors only: the trainer's `torch.load` keeps the `weights_only` default, which
  refuses NumPy scalars.
- A shard is written as `.tmp`, flushed with `F_FULLFSYNC`, then renamed. A rerun skips shards
  that exist, without checking their source again; `--overwrite` rebuilds. A lock
  (`.cache_codes.lock`) keeps a second run off the same `--out-dir`. Run a multi-day build under
  `caffeinate -i -s`.
- Decoding: PyAV for every stream (the Opus clips and cv27's MP3); anything but mono `fltp`
  is rejected. Opus decodes at 48 kHz and cv27's MP3 at 32 or 48 kHz
  ([en-speech-corpus](en-speech-corpus.md)); soxr resamples from that rate straight to each target.
  `sphn.read` cannot decode Opus, `sphn.read_opus_bytes` keeps the 312-sample Opus pre-skip, and
  `sphn.resample` delays its output (256 samples at 48 → 24 kHz).
- Mimi: `mimi_mlx.MimiEncoder`, an encode-only MLX port that loads the PyTorch checkpoint. Each
  clip is encoded whole and alone, padded to a multiple of 16 frames (so the compiled encoder
  sees at most 18 shapes; [mlx-inference](mlx-inference.md#memory)), then cut to
  ceil(samples / 1920) frames. Against moshi 0.2.13 PyTorch on CPU in float32, 99.98% of frames
  are equal on all 16 codebooks and every first codebook (65,237 frames of 504 clips; 12,704
  frames of 40 built rows); the rest are near-ties in the residual codebooks.
- CTC: `text_timing.EnglishCTCAligner` runs `wav2vec2_mlx.Wav2Vec2CTC` in float32, one clip
  at a time, then a numpy Viterbi (the same frames as the former torch loop). The port pads each
  clip to a multiple of 1.28 s and masks the padding (first GroupNorm statistics, positional
  convolution input, attention keys), so it computes the clip alone and compiles over ~18 shapes.
  It builds the same rows as mlx-audio 0.6.1's eager wav2vec2 but for float near-ties (on 2 of
  ~3,200 rows a word sits 1–2 frames apart; see [Speed](#speed)), and that model on 252 rows
  rejects the same rows as PyTorch CPU float32 and gives the same spans on 6,180 of 6,181 words.
- Rejected after measurement (2026-10-27):
  - upstream moshi_mlx 0.3.0 Mimi: no attention mask and tanh GELU, 30% of codes equal on whole
    clips (with both fixed it matches, which `mimi_mlx` does);
  - rustymimi 0.4.1: an attention window of 251 frames, not 250, so codes differ past 10 s;
  - streaming or chunked encoding: differs past 250 encoder frames (10 s);
  - float16 / bfloat16 Mimi: 89% / 45% of frames equal; float16 wav2vec2: 99.3% of word spans;
  - padded wav2vec2 without a mask: changes the emissions, since the base model has none.
- Upstream (checked 2026-10-27): moshi 0.2.13, rustymimi 0.4.1 and moshi_mlx 0.3.0 are the
  newest releases; the Mimi stack of moshi 0.2.13 equals kyutai's main, and the local Mimi
  checkpoint is the file upstream Hibiki-Zero loads (sha256 `09b782f0…`).

## Speed

Measured on the M4 Pro Mac (10 performance + 4 efficiency cores, 48 GB), 2026-10-27/28.

- Speed is real time on the Mimi audio (both clips plus the delay). It swings ±20% with the
  Mac's other load (a stuck `ANECompilerService`, busy browser tabs, other sessions' builds), so
  compare variants interleaved on the same rows in one process, never one run after another.
- Interleaved on the same rows (2026-10-28), the builder before → now: `cv27` 101x → 153x,
  `libriheavy` 164x → 225x, with identical rows. Each part counts: the 16 GB MLX cache (4 GB 195x,
  8 GB 209x; Mimi alone over real clips 275x at 1 GB, 399x at 8 GB, ~500x on one repeated
  shape), wav2vec2 over buckets (its unique raw lengths had filled any cache with buffers never
  reused), and the Viterbi running while Mimi encodes (+5–11%). The footprint grows to ~19 GB.
- Whole shard: `libriheavy-dev-00001` (1,738 rows, 13.2 h) built at 189x, the old builder at
  140–166x. Same rows kept and rejected, every Mimi code equal; on 2 rows the text of one or two
  words sits 1–2 frames apart.
- The full build holds ~22.1k h of Mimi audio (both clips and the mean delay: 21,558 h main,
  521 h `cv-anquach`), ~95–110 h at 200–225x. With the Mac in use it ran 150–170x after the
  restart (busy browser tabs and window rendering share the GPU), ~5.5 days for the Mac alone.
- Alone: Mimi ~500x on one shape (PyTorch CPU 80x; PyTorch MPS 300–350x, but only with length
  buckets: per-clip shapes grow MPS memory to 28–30 GB); wav2vec2 240–290x (PyTorch CPU 86x, MPS
  193x), its 12 layers at ~3.9 TFLOP/s in float32. One clip per forward already fills the GPU.
- PyAV decodes Opus at ~1,430x real time per core (soundfile 413x); two workers keep the GPU fed.
- Tried without gain: batching Mimi or wav2vec2 clips (MLX: −40% to +25%, no pattern; MPS: none
  to −10%); grouping the GPU work by stage; wav2vec2 in CPU processes beside the GPU. With the
  GPU busy, four CPU processes fell from 123x to 20x in total and Mimi slowed ~45%: the CPU and
  GPU share this laptop's power.
- Neural Engine (Core ML 9, checked 2026-10-27): not used. It computes in float16 only, which
  rules out Mimi (above) and moves 0.7% of wav2vec2's word spans. Compiling one fixed 10 s
  wav2vec2 shape ran over 27 min in `ANECompilerService` (root, one core, cutting the build to
  ~150x) without finishing. At best it would take wav2vec2's ~40% of GPU time off the GPU.
- Headroom (2026-10-29, measured beside the build at 240x): the GPU is 96–98% busy, and a row
  takes as long as its two models do alone, so only less GPU work per row would help. On a mean
  `libriheavy-large` row (Vietnamese 11.5 s, English 13.6 s, delay 2.2 s) Mimi takes ~55%
  (SEANet 59%, transformer 24%, quantizer 15%) and wav2vec2 ~45% (its convolutions a third).
  Bucket padding adds 4.5% to each; the delay's silence is 8% of Mimi's frames and must be encoded.
  Convolutions rewritten as matmuls gain nothing on Mimi's k=3 and k=7, and 8% of wav2vec2's
  convolutions (~1% overall): MLX's conv1d is as fast. Left: a second machine, the Mac kept idle,
  and High Power Mode (`powermode 0`, Automatic, so far; untested).

## Builds

The full build, started 2026-10-27 and restarted on the faster builder on 2026-10-28 after the
first 68 `cv27` shards, runs the two out-dirs one after the other, each under
`caffeinate -i -s` with its log in `<out-dir>/build.log`:

| Out-dir on `/Volumes/data/datasets/` | `--shards` | Source rows |
|---|---|---|
| `en-vi-grounded-v2` | `train/cv27-*`, `train/libriheavy-*`, `validation/*` | 3.49M, 10,510 h English |
| `en-vi-grounded-v2-cv-anquach` | `train/cv-anquach-*` | 139k, 256 h English |

- Paused by the user on 2026-10-29 at 19:18, 36.6% of the audio in (767 main shards, through
  `libriheavy-large-00010-00021`, all on the Hub; `cv-anquach` not started). Resume from the
  repo root on `main` with `nohup bash <main out-dir>/launch.sh >/dev/null 2>&1 &` and the same
  for `publish.sh`; the builder skips the built shards.
- `cv-anquach` is kept apart because its English targets are 16 kHz
  ([en-vi-speech-dataset](en-vi-speech-dataset.md#cv-anquach-rows)): pass its `train/` as one
  more `--cache-dir` to use it.
- `publish.sh` beside `launch.sh` in the main out-dir runs `publish_caches --follow` on the two
  out-dirs in the same order, logging to `<out-dir>/publish.log`.
- Both scripts append each log line on its own: unplugging `/Volumes/data` kills a log a
  process holds open. On 2026-10-28 a replug lost the builder's lines while it kept building, and
  `publish_caches` died on its next one (exit 120: Python could not write its stdout). A process
  reading the drive while it is gone can still fail, so stop the build before unplugging.
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
  so the builder, which checks for a shard as it starts it, skips it. Each machine so ends with
  the whole cache (the second machine downloads the Mac's part too). A pulled shard must have
  the settings of a shard already in the out-dir (format, rates, `max_frames`,
  `alignment_min_score`, model config; pulls wait until one exists), the same source file at the
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
- The aligner is pinned to `facebook/wav2vec2-base-960h` at `22aad52`, its two files only, so
  both machines time with the same model. The builder no longer sets `HF_HUB_OFFLINE`, so a
  machine's first run can download them; offline it uses the cached copy (0.4 s on a refused
  connection, ~75 s on a hanging one).
- The second machine needs Apple Silicon (the builder is MLX only) and write access to the
  cache repo; commands are in [training/prep/README.md](../../training/prep/README.md).

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
- `grounded-v2/` was built from `anquachdev/PhoMT-en-vi-speech` at revision `33400f7` (565 GB of
  TTS audio on both sides) by the since-removed `cache_phomt_stream.py`, so it is never
  rebuilt. Filter from it if the rule changes. A row's cached English length counts the
  inserted delay; `en_frames - target_delay_frames` is within a frame of the clip.
- 190,911 of the kept rows carry `cross_lingual_timbre_matched`.

## Row layout

- Codes are int32 `[33, T]`:
  - row 0 is text;
  - rows 1–16 are English target Mimi;
  - rows 17–32 are Vietnamese source Mimi.
- Model config: `n_q` 32, `dep_q` 16, `card` 2048, `text_card` 48000, text pad id 3.
- Text stream: PAD fill, tokens at their CTC frames, then tokenizer EOS at the frame equal to
  the English code length, then -1.
- Target audio is -1 after its length, with no explicit EOS.
- Source audio has codec EOS 2048 at `source_len`, then -1.
- T = max(text end, target length, source length + 1).
- Batches are right-padded with -1 to a multiple of the frame bucket. There is no dataset
  transform at training time.
