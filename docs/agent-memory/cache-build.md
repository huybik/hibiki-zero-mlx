# Cache build

Read when touching `training/prep/cache_build.py`, `training/prep/cache_rows.py`,
`training/prep/mimi_mlx.py`, `training/prep/wav2vec2_mlx.py`, `training/prep/text_timing.py` or
`training/prep/audio.py`, and before running Mimi or wav2vec2 anywhere else, on MLX or CUDA: how
every grounded cache is built on either device, and what each device's speed depends on. The
builders on top are in [sft-data-caches](sft-data-caches.md) (`cache_codes`, the EN-VI dataset) and
[tranche-cache](tranche-cache.md) (`training.tranche.cache`, the tranche); the row layout is in
[sft-data-caches](sft-data-caches.md#row-layout).
Back to [CONTEXT.md](../../CONTEXT.md) · [Shape](#shape) · [Rows](#rows) · [MLX](#mlx) ·
[CUDA](#cuda) · [Across devices](#across-devices) · [Speed on the Mac](#speed-on-the-mac) ·
[Speed on CUDA](#speed-on-cuda) · [Changing it](#changing-it)

## Shape

```text
builder (cache_codes | tranche.cache): its shard keys, read(key) -> Shard (rows, its own gates)
  └─ cache_build.Builder(device).run(keys, read, workers) ─► each Shard filled, in key order
       read             a thread, one shard ahead
       decode workers   cache_rows.prepare: PyAV, soxr to 24 kHz (Mimi) and 16 kHz (CTC,
                        normalized); one window of rows ahead of the device
       cache_rows.plan  decode gates, delay draw, text groups ─► windows, across shard ends
       device           MLXModels | CUDAModels .encode: CTC ─► the Viterbi in the workers
                        (cache_rows.align) while the device runs Mimi
       cache_rows       sample ─► its shard; the builder writes payload() with write()
```

- One implementation per model per device. Every grounded cache runs Mimi and wav2vec2 through
  `cache_build`, and the noisy re-encode takes its CUDA Mimi from it (`cuda_mimi`, `mimi_batch`;
  [noisy-source](noisy-source.md#noisy-cache)). A builder adds only its source's reading and gates,
  the sample fields it keeps (`fields`) and where its shards go. A new source is a new `read`, a
  new device a new `*Models` class with `window`, `cardinality` and `encode`; never a second
  pipeline (2026-11-09: an MLX path written inside the tranche's builder without reading
  `cache_codes` missed its measured MLX levers and decoded whole shards into memory).
- `cache_build` imports MLX, moshi, torch and transformers lazily, and `cache_rows` only torch (a
  sample's codes, the write), so the box runs without MLX and the Mac without CUDA.
- Windows (each models class's `window`): 64 rows on MLX, 4,096 on CUDA. Decodes run one window
  ahead (the pool runs jobs in order, so more would only queue a window's alignments behind more
  decodes), so about two windows of PCM are held: ~0.3 GB on MLX; on CUDA ~10 GB of GigaSpeech 2's
  short rows, ~27 GB of PhoAudiobook's long ones (3.3 MB a row), unmeasured on the box. On top, the
  builder's reads hold two shards' compressed audio (1–2.5 GB a GigaSpeech 2 shard). Decoding a
  whole shard at once held ~23 GB for a GigaSpeech 2 shard, and swapped the 48 GB Mac. A window
  spans shard ends, so the device does not wait on a shard's tail or the next one's first decodes;
  a shard is yielded once its last row is in, so the log line gives the build's speed so far, not
  a shard's.
- Workers (`default_workers`): 2 on MLX, where 2, 4 and 8 ran alike ([Speed on the
  Mac](#speed-on-the-mac)) and the CPU and GPU share one power budget; on CUDA the CPU budget less
  the main process (`paths.cpu_budget`, the container's quota). Each
  decodes and runs the Viterbi with its own `CTCWordAligner` (`init_worker`).

## Rows

- Gates run cheapest first, and a row is rejected, with its reason, at the first one it fails:
  a decoded clip more than 0.02 s off its stored duration (PyAV decodes a truncated or damaged
  clip without error, only shorter); more frames than `max_frames` (280, the trainer's cap),
  counted from the decoded lengths as max(English frames, Vietnamese frames) + 1 before the delay;
  an English clip shorter than one CTC frame; the text grouping; CTC alignment (score below 0.5);
  token timing. Mimi is queued with the CTC forward, so the ~0.5% of rows rejected at alignment or
  token timing are encoded and dropped.
- The delay is drawn from U[0, min(0.5 × Vietnamese duration, room)], seeded by
  `<seed>:<split>:<id>`, the duration rounded to the 0.01 s it stores; `room` is what the English
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
  first, then the build's, in source order), `source`, `max_frames` and `alignment_min_score`. Values are Python types and tensors only: the trainer's `torch.load`
  keeps the `weights_only` default, which refuses NumPy scalars.
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
  sees at most 18 shapes; [mlx-inference](mlx-inference.md#memory)), then cut to
  ceil(samples / 1920) frames. Against moshi 0.2.13 PyTorch on CPU in float32, 99.98% of frames
  are equal on all 16 codebooks and every first codebook (65,237 frames of 504 clips); the rest
  are near-ties in the residual codebooks.
- CTC: `wav2vec2_mlx.Wav2Vec2CTC` in float32, one clip per forward, then the numpy Viterbi
  (`text_timing.CTCWordAligner`) in a worker. The port pads each clip to a multiple of 1.28 s and
  masks the padding (first GroupNorm statistics, positional convolution input, attention keys), so
  it computes the clip alone and compiles over ~18 shapes. It builds the same rows as mlx-audio
  0.6.1's eager wav2vec2 but for float near-ties (2 of ~3,200 rows had a word 1–2 frames apart),
  and that model on 252 rows rejects the same rows as PyTorch CPU float32 and gives the same spans
  on 6,180 of 6,181 words.
- `MLXModels.encode` queues a window's CTC forwards, then its Mimi encodes (`mx.async_eval`), and
  hands each CTC result to the workers as it lands, so they align while the GPU encodes.
  `mx.set_cache_limit(16 GB)` keeps freed buffers for the ~36 bucketed shapes
  ([Speed on the Mac](#speed-on-the-mac)).
- Rejected after measurement (2026-10-27): upstream moshi_mlx 0.3.0 Mimi (no attention mask and
  tanh GELU, 30% of codes equal on whole clips); rustymimi 0.4.1 (an attention window of 251
  frames, not 250, so codes differ past 10 s); streaming or chunked encoding (differs past 250
  encoder frames); float16 / bfloat16 Mimi (89% / 45% of frames equal); float16 wav2vec2 (99.3%
  of word spans); padded wav2vec2 without a mask (changes the emissions).
- Upstream (checked 2026-10-27): moshi 0.2.13, rustymimi 0.4.1 and moshi_mlx 0.3.0 are the
  newest; the Mimi stack of moshi 0.2.13 equals kyutai's main, and the local Mimi checkpoint is
  the file upstream Hibiki-Zero loads (sha256 `09b782f0…`). MLX 0.32.4 (checked 2026-10-30): none
  of the 57 later commits on `main` changes these rows; its new sliding-window attention would save
  under 2%, attention being a small part of Mimi's transformer.

## CUDA

- `CUDAModels`: moshi's Mimi (`cuda_mimi`) and transformers' wav2vec2-base-960h, both in float32
  with TF32 off (TF32 changes Mimi's residual codes as bf16 does, and moves CTC near-ties), over
  padded batches of up to 640 s of audio, shortest first (`batches`). The CTC takes the first
  GroupNorm's statistics over each clip's own frames and gives the encoder the attention mask, as
  `wav2vec2_mlx` masks; Mimi needs no mask (its convolutions and attention are causal).
- On CPU, 8 clips of a shard batched against each alone kept every argmax, log-probabilities within
  2.9e-3 (fp32 rounding: transformers' own fp32 and fp64 forwards differ by 3.1e-3); Mimi's
  batched codes equal the clip-alone ones on all 16 codebooks.
- transformers reports `wav2vec2.masked_spec_embed` as missing from the checkpoint: it only masks
  in training, so the CTC is unaffected.
- `DEVICE` is the torch device; set to `"cpu"` it runs the CUDA path on a Mac, slowly (~3× real
  time), which is how to check a change to it without a box.

## Across devices

Checked on 2026-11-09, after both builders moved onto this build:

- On 400 rows of `validation/libriheavy-dev-00000`, `cache_codes` on MLX gave byte-identical
  samples and rejections to the build before the change and to the production shard.
- On viVoice shard `train_00003_of_00354` (831 rows), the MLX build kept and rejected the same
  rows as the box's CUDA build on the Hub, every text row, delay and field equal (alignment scores
  within 0.005) and 0.047% of audio frames different (float near-ties). The CUDA path run on the
  CPU over the first 80 rows differed on 0.019%.
- So a cache may be built on either device, or partly on each: rows differ only by near-ties.
- `cache_codes` now lists an English clip shorter than one CTC frame as `Rejected`; it used to stop
  the run with wav2vec2_mlx's ValueError.

## Speed on the Mac

Measured on the M4 Pro (10 performance + 4 efficiency cores, 48 GB), 2026-10-27/28, on the EN-VI
dataset's rows; speed is real time on the Mimi audio (both clips plus the delay).

- It swings ±20% with the Mac's other load (busy browser tabs, other sessions' builds), so compare
  variants interleaved on the same rows, never one run after another, and check that no other job
  holds the GPU (`ioreg -r -d 1 -c IOAccelerator` "Device Utilization %").
- Heat costs more: at macOS's thermal state "fair" the build ran 120–200x instead of ~240x, with
  the builder alone on the GPU (2026-11-02). Read the state with `osascript -l JavaScript -e
  'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'` (0 nominal, 1 fair,
  2 serious, 3 critical); `pmset -g therm` reports nothing at "fair".
- Each lever counts (interleaved on the same rows, 2026-10-28: `cv27` 101x → 153x, `libriheavy`
  164x → 225x): the 16 GB MLX cache (4 GB 195x, 8 GB 209x; Mimi alone over real clips 275x at
  1 GB, 399x at 8 GB, ~500x on one repeated shape), wav2vec2 over buckets (raw lengths filled any
  cache with buffers never reused), and the Viterbi running while Mimi encodes (+5–11%). The
  footprint grows to ~19 GB.
- Whole shard: `libriheavy-dev-00001` (1,738 rows, 13.2 h) built at 189x.
- Alone: Mimi ~500x on one shape (PyTorch CPU 80x; PyTorch MPS 300–350x, but only with length
  buckets: per-clip shapes grow MPS memory to 28–30 GB); wav2vec2 240–290x (PyTorch CPU 86x, MPS
  193x), its 12 layers at ~3.9 TFLOP/s in float32. One clip per forward already fills the GPU.
- PyAV decodes Opus at ~1,430x real time per core (soundfile 413x).
- Tried without gain: batching Mimi or wav2vec2 clips (MLX: −40% to +25%, no pattern; MPS: none
  to −10%); grouping the GPU work by stage; wav2vec2 in CPU processes beside the GPU. With the
  GPU busy, four CPU processes running wav2vec2 fell from 123x to 20x in total and Mimi slowed
  ~45%: the CPU and GPU share this laptop's power.
- Neural Engine (Core ML 9, checked 2026-10-27): not used. It computes in float16 only, which
  rules out Mimi and moves 0.7% of wav2vec2's word spans; compiling one fixed 10 s wav2vec2 shape
  ran over 27 min in `ANECompilerService` without finishing.
- Headroom (2026-10-29, at 240x): the GPU is 96–98% busy, and a row takes as long as its two
  models do alone, so only less GPU work per row would help. On a mean `libriheavy-large` row
  (Vietnamese 11.5 s, English 13.6 s, delay 2.2 s) Mimi takes ~55% (SEANet 59%, transformer 24%,
  quantizer 15%) and wav2vec2 ~45%. Bucket padding adds 4.5% to each; the delay's silence is 8% of
  Mimi's frames and must be encoded. Convolutions rewritten as matmuls gain nothing. Untried: the
  Mac kept idle, and High Power Mode (`powermode 0`, Automatic, so far).
- On GigaSpeech 2's short rows (4.3 s), interleaved on 2,000 rows of archive 147 (1,550 kept,
  4.77 h of Mimi audio, 2026-11-09): this build 210–226x with 2, 4 or 8 workers alike, the GPU ~95%
  busy and MLX's peak 2.2 GB; the tranche's MLX path it replaced, with its streaming fix, 219–223x.
  That path before the fix decoded whole shards and ran 130–136x in swap. At ~220x the 131
  GigaSpeech 2 archives left take ~30 h of the Mac (~14 min an archive).
- PR #87 (unmerged) batches a window's clips by padded length into one Mimi and one CTC forward
  and reported 189x against a 148–150x baseline on archives 149–152, the runs one after another
  beside other GPU jobs. The interleaved runs above put the unbatched path at ~220x, and batching
  gained nothing before ("Tried without gain" above), so it is unproven on this Mac; measure it interleaved before
  adopting it, in `MLXModels`.
- After the move onto this build (2026-11-09, 400 `libriheavy-dev` rows, the Mac otherwise in
  light use): 258–263x against the former builder's 265–270x, within the noise above.

## Speed on CUDA

Measured on the box's H100 on the tranche's rows (2026-11-06/07), before the move onto this build,
which then encoded each shard in one pass; the windows and the cross-shard stream are unprofiled
on CUDA. Profile them on the next box before a full run (`--limit`).

- A warm PhoAudiobook shard (1,299 rows, 9.6 h of Mimi audio) in 21 s, 1,609x real time, the GPU
  74–100% busy; a GigaSpeech 2 shard (19,967 rows, 49.9 h) in 161 s with the model load. ~49 GB
  of GPU at its peak.
- Per-row costs: GigaSpeech 2 ran at 825x (4 min a 20k-row shard) against PhoAudiobook's 1,478x
  and viVoice's 1,172x; its rows are short (4.3 s), so the main thread's per-row Python (`plan`,
  collecting the alignments, `assemble_codes`, the sample dicts) likely holds the GPU idle longer
  per hour of audio. Measure it, and move it into the workers if so.
- Measure where the GPU time goes, Mimi against wav2vec2 (never split yet), with `nvidia-smi`
  every 0.2 s and the gaps lined up with the log; aim for under 10% idle.
- Mimi stays float32 with TF32 off (other precisions change the codes); `torch.compile` or CUDA
  graphs on its encoder keep the codes. wav2vec2 may run in TF32 or bf16 if word starts stay
  within a frame of float32 on real rows.
- CPU: 25 workers on the 26.9-CPU quota averaged 10.5 CPUs but throttled 14% of the 100 ms
  periods in bursts (`/sys/fs/cgroup/cpu.stat`); check the main process is not the one stalled.

## Changing it

- Untried on both devices: queue window N+1 on the device before collecting window N's alignments
  and samples, so the device does not idle while the main thread plans and collects.

- A speed lever, a device or a model goes here, for every builder at once; read both Speed
  sections first, since each device's levers were measured there and do not carry over (MLX runs
  one clip per forward, CUDA large padded batches).
- Check a change against rows already built: the old build's shards (`cache_codes`' on
  `/Volumes/data/datasets/en-vi-grounded-v2`, the tranche's on the Hub) must come out equal, or
  equal but for near-ties on another device. Compare speeds interleaved on the same rows.
