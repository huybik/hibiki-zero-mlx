# Cache build

Read when touching `training/prep/cache_build.py`, `training/prep/cache_rows.py`,
`training/prep/mimi_mlx.py`, `training/prep/mimi_cuda.py`, `training/prep/wav2vec2_mlx.py`,
`training/prep/wav2vec2_cuda.py`, `training/prep/wav2vec2_coreml.py`, `training/prep/text_timing.py`,
`training/prep/_ctc_viterbi.c` or `training/prep/audio.py`, and before running Mimi or a CTC
aligner anywhere else, on MLX or CUDA: how every grounded cache is built on either device, and
what each device's speed depends on. The builders on top are in [sft-data-caches](sft-data-caches.md)
(`cache_codes`, the EN-VI dataset) and [tranche-cache](tranche-cache.md) (`training.tranche.cache`,
the tranche); the row layout is in [sft-data-caches](sft-data-caches.md#row-layout).
Back to [CONTEXT.md](../../CONTEXT.md) · [Shape](#shape) · [Rows](#rows) · [MLX](#mlx) ·
[CUDA](#cuda) · [Across devices](#across-devices) · [Speed on the Mac](#speed-on-the-mac) ·
[Changing it](#changing-it); the speed and memory on CUDA are in [cache-speed-cuda](cache-speed-cuda.md)

## Shape

```text
builder (cache_codes | tranche.cache): its shard keys, read(key) -> Shard (rows, its own gates)
  └─ cache_build.Builder(device).run(keys, read, workers) ─► each Shard filled, in key order
       read             a thread, one shard ahead
       decode workers   cache_rows.prepare_rows, jobs of 32 rows of a shard: PyAV, soxr to 24 kHz
                        (Mimi) and 16 kHz (CTC, normalized), then cache_rows.plan (decode gates,
                        delay draw, text groups); one window of rows ahead of the device; clips
                        back through /dev/shm (inline where there is none) ─► windows, across
                        shard ends
       device           MLXModels | CUDAModels .encode: CTC ─► the Viterbi in the workers
                        (cache_rows.align_rows, a job a CTC batch) while the device runs Mimi
       cache_rows       sample ─► its shard; the builder writes payload() with write()
```

- One implementation per model per device. Every grounded cache runs Mimi and wav2vec2 through
  `cache_build`, and the noisy re-encode takes its CUDA Mimi from it (`cuda_mimi`;
  [noisy-source](noisy-source.md#noisy-cache)). A builder adds only its source's reading and gates,
  the sample fields it keeps (`fields`) and where its shards go. A new source is a new `read`, a
  new device a new `*Models` class with `window`, `cardinality` and `encode`; never a second
  pipeline (2026-11-09: an MLX path written inside the tranche's builder without reading
  `cache_codes` missed its measured MLX levers and decoded whole shards into memory).
- `cache_build` imports MLX, moshi, torch and transformers lazily, and `cache_rows` only torch (a
  sample's codes, the write), so the box runs without MLX and the Mac without CUDA.
- Windows (each models class's `window`): 64 rows on MLX, 512 on CUDA. Decodes run one window
  ahead (the pool runs jobs in order, so more would only queue a window's alignments behind more
  decodes), so the PCM of several windows is held: ~0.3 GB on MLX; on CUDA the window sets the
  build's RAM, ~5 GB plus ~10 MB a window row on Libriheavy
  ([cache-speed-cuda](cache-speed-cuda.md#memory), where 4,096 rows ran a 31 GB box out of memory). On top, the
  builder's reads hold two shards' compressed audio (1–2.5 GB a GigaSpeech 2 shard). Decoding a
  whole shard at once held ~23 GB for a GigaSpeech 2 shard, and swapped the 48 GB Mac. A window
  spans shard ends, so the device does not wait on a shard's tail or the next one's first decodes;
  a shard is yielded once its last row is in, so the log line gives the build's speed so far, not
  a shard's.
- Workers (`default_workers`): 2 on MLX, where 2, 4 and 8 ran alike ([Speed on the
  Mac](#speed-on-the-mac)) and the CPU and GPU share one power budget; on CUDA the CPU budget less
  the main process (`paths.cpu_budget`, the container's quota). Each
  decodes and runs the Viterbi with its own `CTCWordAligner` (`init_worker`). A job holds many rows
  (32 a decode job; a CTC batch's an alignment job on CUDA, one row on MLX) and returns each row's
  `Rejected` or `AlignmentError` in its place: a job a row cost the main process each row's pool
  round trip under the GIL its GPU thread needs ([cache-speed-cuda](cache-speed-cuda.md)).

## Rows

- Gates run cheapest first, and a row is rejected, with its reason, at the first one it fails:
  a decoded clip more than 0.02 s off its stored duration (PyAV decodes a truncated or damaged
  clip without error, only shorter); more frames than `max_frames` (280, the trainer's cap),
  counted from the decoded lengths as max(target frames, source frames) + 1 before the delay;
  a target clip shorter than one CTC frame; the text grouping; CTC alignment (fewer CTC frames
  than the transcript's characters plus its doubled ones, the least a CTC path takes, 2N + 1
  before 2026-11-22, which rejected 2.8% of GigaSpeech 2's rows as fast speech; or a score below
  the aligner's `min_score`: 0.5 English, 0.55 Vietnamese); token timing; a target running on past
  its last text token longer than the aligner's `max_tail_s` (1.5 s Vietnamese, for VieNeu
  speaking past its text; none English, whose recordings end on silence:
  [sft-training](sft-training.md#target-timing), which applies it again to caches built before it). Mimi is queued with the CTC forward, so the
  rows rejected at alignment or token timing are encoded and dropped.
- The direction (`Settings.direction`, `vi-en` by default, `en-vi` for two-way) says which clip
  is the source and times the target's text against the target clip with that language's CTC
  model (`text_timing.ALIGNERS`: wav2vec2-base-960h for English, wav2vec2-base-vietnamese-250h
  for Vietnamese, [two-way](two-way.md#vietnamese-aligner)). The CTC hands the host only the columns its Viterbi
  reads (`vocab_columns`).
- The delay is drawn from U[0, min(0.5 × source duration, room)], seeded by
  `<seed>:<split>:<id>`, the duration rounded to the 0.01 s it stores; `room` is what the target
  clip leaves of the frame cap ([sft-data-caches](sft-data-caches.md#flow) has why).
- Text grouping: SentencePiece pieces group into words, and a group with nothing to speak (a
  stray `*`, `-` or `"`; a lone `'` counts as spoken) joins the word before it. Before the first
  word it is the lead, timed on the frames just before that word's start, so the word keeps its
  own frame; when the lead does not fit after the delay it starts at the delay and pushes the
  word later. An unspoken piece inside a word (`"It`, `(see`) is spread over that word's span
  with it (half of Libriheavy rows). Leading ones were rejected before 2026-10-27 (51k Libriheavy
  rows, 1.7%).
- Only data faults reject a row: `audio.Rejected` (undecodable or non-mono audio, a duration
  mismatch, the frame cap) and `text_timing.AlignmentError`. Any other error, such as a dead
  worker (`BrokenProcessPool`), a device error or PyAV running out of memory, stops the run
  before the shard is written, so a rerun rebuilds it rather than skipping lost rows. A shard
  whose rows all fail on data is written, empty.
- A sample holds the row fields, the builder's `fields` and `source_row` (`<file>#r<index>`); a
  shard holds `samples` in source order, `rejected` (`id`, `reason`: the builder's own gates
  first, then the build's, in source order), `source`, `max_frames`, `alignment_min_score`, `max_tail_s` (none before 2026-11-22),
  `direction` and `tokenizer`. Values are Python types and tensors only: the trainer's
  `torch.load` keeps the `weights_only` default, which refuses NumPy scalars.
- A shard is written as a hidden `.tmp`, flushed with `paths.durable` (`F_FULLFSYNC` on macOS),
  then renamed.
- Decoding: PyAV for every stream (Opus, cv27's MP3), in `audio.decode`; anything but mono `fltp`
  is rejected. Opus decodes at 48 kHz and MP3 at 32 or 48 kHz; soxr resamples from that rate
  straight to each target. `sphn.read` cannot decode Opus, `sphn.read_opus_bytes` keeps the
  312-sample Opus pre-skip, and `sphn.resample` delays its output (256 samples at 48 → 24 kHz).
- wav2vec2's input normalization (zero mean, unit variance over the clip) runs in the workers
  (`prepare`) for both devices, off the main thread.

## MLX

- Mimi: `mimi_mlx.MimiEncoder`, an encode-only MLX port that loads the PyTorch checkpoint. Each
  clip is encoded whole and alone, padded to a multiple of 16 frames (so the compiled encoder
  sees at most 18 shapes, traced once a stream; [mlx-inference](mlx-inference.md#memory)), then
  cut to ceil(samples / 1920) frames. Successive calls go to four GPU streams in turn, so several
  clips run at once; those streams exist only on the thread that made the encoder (MLX 0.32.4), so
  build and call it on one thread, as every builder does. The residual blocks of SEANet's 64-, 128-
  and 256-channel stages run as one Metal kernel each (`_residual`, `mx.fast.metal_kernel`, the
  hidden channels in registers). Against moshi 0.2.13 PyTorch on CPU in float32, 99.96% of frames
  are equal on all 16 codebooks and every first codebook (101,277 frames of 800 `en_speech`
  clips, 2026-11-22; the port before the kernel the same); on 230 PhoAudiobook clips one first
  codebook frame was a near-tie (distances 7.6e-6 apart). The rest are near-ties in the residual
  codebooks.
- CTC: `wav2vec2_mlx.Wav2Vec2CTC` in float32 by default; `half=True` (`tranche.cache --half-ctc`)
  for float16 weights, measured on omniASR (the former Vietnamese aligner) only; `Builder` refuses
  it on CUDA, whose CTC always runs in FP16. One clip per forward, then the C Viterbi
  (`text_timing.CTCWordAligner`) in a worker. The port pads each
  clip to a multiple of 1.28 s and masks the padding (first GroupNorm statistics, positional
  convolution input, attention keys), so it computes the clip alone and compiles over ~18 shapes.
  It ports the base layout both aligners have (GroupNorm on the first convolution, post-norm
  layers) and refuses another. On 30 `en_speech` clips the Vietnamese aligner's Viterbi columns kept
  every argmax of transformers' CPU float32 (13,143 frames, log-probabilities within 2.3e-3,
  2026-11-22). It builds the same rows as mlx-audio 0.6.1's eager wav2vec2 but for float near-ties
  (2 of ~3,200 rows had a word 1–2 frames apart), and that model on 252 rows rejects the same rows
  as PyTorch CPU float32 and gives the same spans on 6,180 of 6,181 words. In float16 the
  GroupNorm statistics stay in float32 (`first_frames` passes 65504 past 20.47 s, the sum of squares
  overflows past ~8 s), keeping ~81% of the gain (355× vs 367× on a 13.6 s clip; float32 305×).
  Against transformers' CPU float32 (40 clips each, 2026-11-22), `half=True` on Metal moved 0.035% of
  the Vietnamese aligner's argmaxes and 0.14% of the English one's (worst clip 0.76%); float32 moved
  none. On MLX's CPU backend English moved 1.4–1.6%.
- `ane_ctc` (`cache_codes --ane-ctc`): the transformer on the Neural Engine in its own process
  (`wav2vec2_coreml`, with the float16 fixes), on `Wav2Vec2CTC.transformer_input` from the GPU in
  float32; the longest rows' CTC stays on the GPU (`_split`, [speed](#speed-on-the-mac)). `en-vi`
  only (`Builder` refuses another direction): the English aligner there kept 99.35% of MLX float32's
  argmaxes (worst clip 98.39%), the Vietnamese 99.94%.
- `MLXModels.encode` queues a window's CTC forwards first (`mx.async_eval`), then its Mimi
  encodes (`mx.async_eval`), hands each CTC result to the workers as it lands (with `ane_ctc`, the
  Neural Engine's rows together, then the GPU's; they align while the GPU encodes Mimi), and
  returns lazy Mimi codes. The *next* window's CTC is queued in
  `_encode_start` before the previous window's Mimi codes are collected in `_encode_finish`; Mimi
  runs on its own four streams, so Mimi_N runs beside CTC_N+1 on the default stream.
  `mx.set_cache_limit(16 GB)` keeps freed buffers for the ~36 bucketed shapes ([speed](#speed-on-the-mac)).
- Rejected after measurement (2026-10-27): upstream moshi_mlx 0.3.0 Mimi (no attention mask and
  tanh GELU, 30% of codes equal on whole clips); rustymimi 0.4.1 (an attention window of 251
  frames, not 250, so codes differ past 10 s); streaming or chunked encoding (differs past 250
  encoder frames); float16 / bfloat16 Mimi (89% / 45% of frames equal); padded wav2vec2 without
  a mask (changes the emissions).
- Upstream (checked 2026-10-27): moshi 0.2.13, rustymimi 0.4.1 and moshi_mlx 0.3.0 are the
  newest; the Mimi stack of moshi 0.2.13 equals kyutai's main, and the local Mimi checkpoint is
  the file upstream Hibiki-Zero loads (sha256 `09b782f0…`). MLX 0.32.4 (checked 2026-10-30): none
  of the 57 later commits on `main` changes these rows; its new sliding-window attention would save
  under 2%, attention being a small part of Mimi's transformer.

## CUDA

- `CUDAModels` (and `CUDAModelsDual` / `DualBuilder` for single-pass `--direction both` in
  `tranche.cache`): runs `mimi_cuda.FastMimiEncoder` (`cuda_mimi`) and `wav2vec2_cuda.FastWav2Vec2CTC`,
  each a single FP16 path, with `CUDA_TF32` enabled, over padded batches of up to `CUDA_BATCH_S`
  (320 s) of audio, shortest first (`batches`; both clips of a window in one sort for Mimi, delay
  silence applied zero-copy in `_stage`). `encode` runs in the build's GPU thread
  (`threaded`) with preallocated double-buffered pinned H2D/D2H pools (`_PinnedOutputPool`), one window
  queued behind it.
- Workers (`cache_rows.prepare_rows` / `prepare_dual_rows`) pack each 32-row decode job into ONE
  contiguous `/dev/shm` `.f32` slab file instead of per-row files, each row's 24 kHz clips and its
  normalized 16 kHz target. `text_timing.CTCWordAligner` runs the Viterbi in a C99 shared library
  (`_ctc_viterbi.c`), built with `cc -O3` into `__pycache__` beside it on first use (again when
  the source is newer); without a compiler, or if the build or load fails, it raises, on MLX too.
  `cache_codes` writes shards on a background thread; `tranche.cache` prefetches up to `--prefetch 3` Hub shards and gates metadata columns before
  reading audio bytes.
- `FastWav2Vec2CTC` keeps `c0_conv`, the Triton two-pass masked GroupNorm+GELU statistics, and `lm_head`
  in FP32, runs `c1..c6` as channels-last FP16 `conv2d`, folds `pos_conv_embed` `weight_norm` once at
  load, fuses QKV (`768 -> 2304`), runs SDPA with an additive float key mask (cuDNN attention), and
  compiles the convolutions and the 12 layers (`torch.compile`, always). `mimi_cuda` and
  `wav2vec2_cuda` import Triton at the top and refuse any device but CUDA, so the CUDA path runs only
  on a box with Triton (`box/envs.sh`'s cache env). Measured RTX 5090 parity and speed, and the
  levers measured and removed, are in
  [cache-speed-cuda](cache-speed-cuda.md#blackwell-kernels-and-pipeline-2026-11-28).
- transformers reports `wav2vec2.masked_spec_embed` as missing from the checkpoint: it only masks
  in training, so the CTC is unaffected.
- `DEVICE` is the torch device of `CUDAModels`: CUDA only (pinned staging, `torch.cuda.Event`, the
  Triton kernels). Check a change to it on a box.

## Across devices

Checked on 2026-11-09, after both builders moved onto this build:

- On 400 rows of `validation/libriheavy-dev-00000`, `cache_codes` on MLX gave byte-identical
  samples and rejections to the build before the change and to the production shard.
- On viVoice shard `train_00003_of_00354` (831 rows), the MLX build kept and rejected the same
  rows as the box's CUDA build on the Hub, every text row, delay and field equal (alignment scores
  within 0.005) and 0.047% of audio frames different (float near-ties). The CUDA path run on the
  CPU over the first 80 rows differed on 0.019%.
- `en-vi` (2026-11-21), `cv27-00000` built on MLX against the Hub's CUDA (TF32) shard: the same
  rows but one at the score floor (0.498 against 0.5004), 5 of 2,936 text rows a token frame
  apart, 94.8% of audio codes equal (first codebooks 99.8%). That gap is TF32's: on 40 rows MLX
  matched moshi's CPU float32 on 99.99% of codes (first codebooks all), the Hub's shard 94.7%.
- So a cache may be built on either device, or partly on each: rows differ only by near-ties.
- `cache_codes` now lists an English clip shorter than one CTC frame as `Rejected`; it used to stop
  the run with wav2vec2_mlx's ValueError.

## Speed on the Mac

Measured on the M4 Pro (10 performance + 4 efficiency cores, 48 GB), 2026-10-27/28, on the EN-VI
dataset's rows; speed is real time on the Mimi audio (both clips plus the delay).

- It swings ±20% with the Mac's other load (busy browser tabs, other sessions' builds), so compare
  variants interleaved on the same rows, never one run after another, and check that no other job
  holds the GPU (`ioreg -r -d 1 -c IOAccelerator` "Device Utilization %").
- Heat: at thermal state "fair" the build ran 120–200x, not ~240x (2026-11-02). Read it with
  `osascript -l JavaScript -e 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'`
  (0 nominal, 1 fair, 2 serious, 3 critical); `pmset -g therm` is silent at "fair".
- Each lever counts (interleaved on the same rows, 2026-10-28: `cv27` 101x → 153x, `libriheavy`
  164x → 225x): the 16 GB MLX cache (4 GB 195x, 8 GB 209x; Mimi alone over real clips 275x at
  1 GB, 399x at 8 GB, ~500x on one repeated shape), wav2vec2 over buckets (raw lengths filled any
  cache with buffers never reused), and the Viterbi running while Mimi encodes (+5–11%). The
  footprint grows to ~19 GB.
- Whole shard: `libriheavy-dev-00001` (1,738 rows, 13.2 h): 189x; this build 258–263x.
- Alone: Mimi ~500x on one shape (PyTorch CPU 80x; PyTorch MPS 300–350x, but only with length
  buckets: per-clip shapes grow MPS memory to 28–30 GB); wav2vec2 240–290x (PyTorch CPU 86x, MPS
  193x), its 12 layers at ~3.9 TFLOP/s in float32. One clip per forward keeps the GPU ~95% busy.
- PyAV decodes Opus at ~1,430x real time per core (soundfile 413x).
- Tried without gain: batching Mimi or wav2vec2 clips (MLX: −40% to +25%, no pattern; MPS: none
  to −10%); grouping the GPU work by stage; wav2vec2 in CPU processes beside the GPU. With the
  GPU busy, four CPU processes running wav2vec2 fell from 123x to 20x in total and Mimi slowed
  ~45%: the CPU and GPU share this laptop's power.
- `en-vi` build (omniASR CTC on the Vietnamese clip; 2026-11-21, 500 `libriheavy-large` and 1,000
  `cv27` rows, interleaved, thermal state "fair"): fp32 150–161× (CTC ~68% of the GPU), `half_ctc`
  175–186×, `ane_ctc` 293–309×, the GPU 97–99% busy (`ANE_GPU_SHARE` 0.2; at 0 the GPU waited on
  long rows, 261×; 0.1 ran 278–301×). Against fp32, `ane_ctc` moved 21 of 43,092 text tokens a
  frame (fp16 17), kept one row fp32 rejects (0.515 vs 0.498, floor 0.5; fp16 too), audio codes
  equal; on a whole `libriheavy-large` shard, 259× vs 154×, 12 of 63,507 tokens a frame apart. In a
  thread it ran 207×: Core ML's predict holds the GIL, which `mx.async_eval` (blocked half the build
  on the GPU's queue) needs. macOS caches the first load's compile (~1.5–3 min).
- `en-vi` build with wav2vec2-base-vietnamese-250h, the Vietnamese aligner since 2026-11-22
  (`ane_ctc`; 1,000 `cv27-00058` and 500 `libriheavy-large-00012-00103` rows, each pair interleaved,
  thermal state "fair"): 323× and 275× against omniASR's 286× and 251×, then 237× and 163× against
  215× and 157× as the Mac heated: 4–13% faster, Mimi now the most of the GPU's work.
  `ANE_GPU_SHARE` 0.1, 0.2 and 0.3 ran alike (256–299×, interleaved twice), 0 ran 187–225×. The
  Neural Engine alone runs its transformer at 285–434× of Vietnamese audio, against MLX float32
  99.90–99.93% of argmaxes equal. The same rows built with each aligner: audio codes equal; 82–84%
  of text tokens in the same frame, 99.7–99.8% within one; a token over 5 frames apart in 1.5% of
  the Common Voice rows (mostly in a clip's last three tokens, the last word or its punctuation
  over a long tail; at most 160 frames) and 0.8% of the Libriheavy ones; alignment rejected 1 of
  the 1,000 Common Voice rows (omniASR 13).
- Neural Engine for Mimi and wav2vec2-base (Core ML 9, 2026-10-27): float16 only rules out Mimi and
  moved 0.7% of the base's word spans; one fixed 10 s shape compiled over 27 min without finishing.
- Mimi (2026-11-22; 400 `cv27` and 400 `libriheavy-large` English clips, interleaved, thermal
  state "fair"): the residual blocks of SEANet's 64-, 128- and 256-channel stages run as one Metal
  kernel each (`mimi_mlx._residual`: 3.3×, 2.6× and 1.8× those blocks; the 512-channel one gained
  nothing), the downsampling weights are concatenated once at load (the 512-channel stage's 33 MB
  were concatenated every call), and four clips run at once on their own GPU streams (+2–7% on
  Mimi alone): Mimi alone 395–457× → 590–704× (`cv27`) and 306–518× → 539–730× (`libriheavy`); a
  whole `cv27` shard's noisy copy 288–326× → 391–447×, the GPU still 93–95% busy; the `en-vi`
  build (`ane_ctc`, 1,000 `cv27-00058` and 500 `libriheavy-large-00012-00103` rows, three rounds)
  254–271× → 330–340× and 225–263× → 285–319×, streams there neither helping nor hurting
  (314–325× and 298–312× on one); the tranche's PhoAudiobook build 200–235× → 267–290×. Each
  kept and rejected the same rows with the same text and fields, audio codes 99.97–99.99% equal.
  A 10 s clip then spends 44% in the downsampling convolutions (their bare matmuls run 4.4–5.5
  TFLOP/s, a large square one 6.5), 21% the transformer, 17% the residual kernels, 7% the
  quantizer, 5% the first convolution. Without gain: the residual blocks as im2col or polyphase
  matmuls, the downsampling as a Metal kernel or one matmul over an overlapping strided view (its
  padding copy costs what it saves), the first convolution as a matmul.
- Before that (2026-10-29, at 240x): the GPU 96–98% busy and a row as long as its two models
  alone, so only less GPU work per row helps; the delay's silence is 8% of Mimi's frames and must
  be encoded. Untried: High Power Mode (`powermode 0`, Automatic, so far).
- On GigaSpeech 2's short rows (4.3 s), interleaved on 2,000 rows of archive 147 (1,550 kept,
  4.77 h of Mimi audio, 2026-11-09): this build 210–226x with 2, 4 or 8 workers alike, the GPU ~95%
  busy and MLX's peak 2.2 GB; the tranche's MLX path it replaced, with its streaming fix, 219–223x.
  That path before the fix decoded whole shards and ran 130–136x in swap. At ~220x the 131
  GigaSpeech 2 archives left take ~30 h of the Mac (~14 min an archive).
- Batching a window's clips into one Mimi and one CTC forward (PR #87, closed) was slower (long
  rows 268–277× → 183–194×), lost the CTC/Mimi overlap and peaked at 15.3 GB: the serial path stays.

## Changing it

- N+1 GPU pipelining (PR #88, 2026-11-09): N-1 is collected and N+1 planned while Mimi_N runs. On
  MLX its gain sat within the ±20% swing (204× against 210–226× on GigaSpeech 2, not interleaved);
  on CUDA the read-ahead thread hides Hub downloads behind a shard's encoding.
- A speed lever, a device or a model goes here, for every builder at once; read both Speed
  sections first, since each device's levers were measured there and do not carry over (MLX runs
  one clip per forward, CUDA large padded batches).
- Check a change against rows already built: the old build's shards (`cache_codes`'
  `en-vi-grounded-v2/` and the tranche's, both on the Hub) must come out equal, or equal but for
  near-ties on another device or with `ane_ctc`, whose device for a row's CTC depends on its window,
  so on where the build started. Compare speeds interleaved on the same rows.
