# Swift runtime

Read when touching `inference/swift/` (the MLX Swift runtime, the `hibiki` CLI, the HibikiBench
iOS app) or when measuring on a phone.
Back to [CONTEXT.md](../../CONTEXT.md) · [Layout](#layout) · [Hibiki-Zero deltas](#hibiki-zero-deltas) ·
[Parity](#parity) · [Mimi on the CPU](#mimi-on-the-cpu) · [Speed and memory](#speed-and-memory) ·
[Device test](#device-test)

## Layout

- `inference/swift/` is a SwiftPM package pinned to mlx-swift 0.32.3 (MLX core 0.32.2; the Python
  side runs mlx 0.32.4). Build commands are in its [README](../../inference/swift/README.md).
  `swift build` cannot compile MLX's Metal shaders; build with `xcodebuild`.
- `Sources/HibikiCore/` is the library:
  - Mimi's encoder and the transformer are vendored from
    [moshi-swift](https://github.com/kyutai-labs/moshi-swift) (MIT, commit `df64ffd`,
    `LICENSE-moshi-swift`), cut to Mimi's streaming encoder (`encodeStep`) and the transformer
    layers the LM reuses. Its ring KV cache is replaced by `WindowKVCache` ([chunks](#chunks)).
  - `NeuralDecoder.swift` runs Mimi's decoder on the Neural Engine, one frame per call, from the
    compiled Core ML model `mimi-decoder.mlmodelc` in the model directory
    ([mimi-npu](mimi-npu.md)).
  - `LM.swift` (config, weights, LM, depformer), `Decode.swift` (compiled one-token paths),
    `Generator.swift` (delayed streams), `Translator.swift` (k 80 ms frames per step: the LM on
    the GPU, beside it Mimi encoding on the CPU and decoding on the Neural Engine), `Bench.swift` (sustained-load test), `Vocabulary.swift` (reads
    the SentencePiece `.model` protobuf directly, and `TextDecoder`) and `Audio.swift` (file reading, the WAV
    writer, and `Leveller`, the input level stage `Translator.step` runs before Mimi: Python's
    `hibiki_mlx/level.py`, equal sample for sample; `reset` starts its stream again;
    [sft-evaluation](sft-evaluation.md#input-level)).
- `Translator.Output.text` is what the step's tokens complete: `TextDecoder` assembles UTF-8
  from byte-fallback pieces (`<0xE1>`, type BYTE) across frames and steps, as `TextDecoder` in
  `pipeline.py` does, an invalid sequence as U+FFFD; `flush` returns the rest. Fed the same
  token streams (2026-11-14, scratch harness compiling `Vocabulary.swift`: 364 streams on both
  tokenizers, PADs between pieces), it returns Python's text call for call, and the same whole
  text at 2, 3 and 5 tokens per call. `config.json` must name `moshi_name`, `tokenizer_name` and
  `lm_gen_config`, which sets `Translator`'s samplers, and may hold `quantization`
  ([mlx-inference](mlx-inference.md#model-directories)).
- `Translator.endInput()` ends a file's input: one LM step whose 16 source codes are all the
  source EOS (`card`) while Mimi codes a silent frame, as `pipeline.translate` does.
- `Sources/hibiki/` is the macOS CLI: `translate` mirrors `inference/main.py` (the clip
  zero-padded to whole frames, `endInput`, silence to 250 frames after the clip, stop at the end
  of the first step after `endInput` with the text EOS, seed 299792458; `--model` defaults to
  `weights/hibiki-zero-3b-q4`); `bench` runs
  `Bench`, which streams `<base>.jsonl` (one sample per step) and `<base>.wav` to disk as it
  runs and writes `<base>.txt` (summary, text) at the end. Both take `--chunk K` frames per
  step (default 1), as the iOS app does. `Audio.Writer` writes 16-bit WAV
  clipped like `sphn.write_wav` and rewrites the header after every step: a run killed at
  frame 153 left 153 samples and a valid 12.08 s WAV.
- `HibikiBench/` is the iOS app: a hand-written Xcode project (synchronized folder `App/`) that
  depends on the local package, with the increased-memory-limit entitlement. `--model NAME` reads
  `Documents/NAME/`, so several models can stay on the phone. The team ID is
  passed on the command line (`DEVELOPMENT_TEAM=`), not stored in the project. It caps MLX's
  buffer cache at 256 MB through `Translator(cacheLimit:)` (freed buffers count toward the
  app's memory limit) and runs the bench on a user-interactive thread. iOS refuses GPU work
  from the background, so the app must stay in the foreground; locking the phone ends a run.
- The public API is `Translator`, `Bench`, `Audio`, `Sampler`, `Vocabulary`, `HibikiConfig` and
  `HibikiError`; the vendored and LM types are internal (`@testable import` reaches them).
- Hibiki Edge (An Quách's 1B Swift app) has no licence file, so none of its code is used here.

## Hibiki-Zero deltas

Stock moshi-swift runs none of the four; this runtime implements them. It reads `config.json`
with `LmConfig.from_config_dict`'s defaults and fails on any setting it does not implement
(cross-attention, conditioners, layer scale, other norms or gating):

| Delta | Where |
|---|---|
| FFN width `hidden_scale × dim` | `HibikiConfig.transformer` / `.depformer` |
| Grouped-query attention, `kv_repeat` 2 | `Attention.projections` (Transformer.swift) |
| `rope_concat` = MLX RoPE `traditional: false` | `PositionalEmbedding.ropeConcat` |
| Learned per-slice depformer LayerNorm | `DepformerSlice.norm` |

- Depformer weights per step: a staged weights file holds a full copy for each of the 16
  codebook steps, but steps 8–15 share one weight set (`depformer_weights_per_step_schedule`).
  `Depformer.remap` loads each set once into `depformer.cores.N`: of the 3B's q4 file (2.65 GB),
  2.19 GB of LM weights (computed from its header; uniform q4, which the device numbers below
  ran, loaded 1.96 of 2.41 GB). MLX loads safetensors lazily, so the dropped copies are never
  read.
- Quantized weights load as `pipeline.py` loads them: every `Quantizable` (Linear, Embedding) at
  the `quantization` record's default or its own entry, looked up by the checkpoint's path
  (`Depformer.checkpointPath` names a core by its first slice), then a strict update.
- mlx-swift 0.32 treats every `MLXArray` property of a module as a parameter unless its name
  starts with `_`, so Mimi's derived codebook arrays are `_embedding` and `_c2`.
  The Mimi checkpoint holds 32 codebooks; `Mimi.load` keeps the source stream's 16 and skips
  the decoder's weights. `Mimi.resetState` clears every streaming layer, the downsampler
  included, so a reset session starts from edge padding as a fresh rustymimi tokenizer does.

## Parity

Against `inference/hibiki_mlx` on `inference/assets/leon.wav` (2026-11-01, scratch scripts, not
kept):

- Mimi: Swift MLX codes equal rustymimi's in 98% of frames on codebook 0, falling to 87% on
  codebook 15. Against the training caches' encoder (`training/prep/mimi_mlx.py`, whole clip)
  on the same 800 frames, they are equal on 100% of frames in codebooks 0–9 and 99% in 10–15,
  at any chunk size (2026-11-02).
- LM, teacher-forced on the Python run's greedy tokens and rustymimi codes: 99.0% of text
  tokens and 98.6% of codebook-0 tokens equal; each further codebook matches 93–96% of the time
  given the previous ones, with no step at codebook 8 where the shared weights start. Free
  running, the first 25 frames match token for token, then a near-tie flip diverges the streams.
  Python's own compiled and eager paths are bit-identical, so the residue is the MLX version and
  kernel difference.
- Sampled translations of leon.wav read as well as the Python runtime's.

## Mimi on the CPU

`Translator.step` starts LM step t on the GPU with `asyncEval`, meanwhile decodes the audio of
step t − 1 on the Neural Engine ([mimi-npu](mimi-npu.md)) and encodes frame t on the CPU, then
waits for the LM, as `pipeline.py` overlaps rustymimi threads with the LM. An LM step's audio is
decoded during a later one, so it comes out after its text; `flush` returns the audio and text
left when the input ends (the CLI's `translate` and `Bench` call it). A step can take k frames
([chunks](#chunks)).

- One thread drives both devices: mlx-swift 0.32 holds a global lock through every `eval`,
  waiting included, so MLX work on a second thread takes turns with the first.
- MLX's `eval` blocks while more than 10 tasks are in flight; a task is a Metal command buffer,
  which MLX closes every 20 ops or 40 MB on phones (40–50 on Macs), or 10 CPU dispatches (~5
  ops). With those defaults the LM step ran almost entirely inside `asyncEval` (34 of 38 ms on
  the A15) and the CPU work followed it. `Translator.init` sets `MLX_MAX_OPS_PER_BUFFER=400` and
  `MLX_MAX_MB_PER_BUFFER=1000000` unless already set: LM start 4.5 ms on the A15. MLX reads them
  once, when the first array or `Memory` setting creates its Metal device (`MLXRandom.seed`
  too), so the translator must come before anything else touches MLX; the CLI seeds after
  creating it. The 16-codebook 3B (~2,600 ops a step) does not block either: on the Mac its LM
  start was 14–16 ms at 400, 1,000 and 3,000 ops per buffer. The CPU side cannot be split the
  same way: `asyncEval` of a ~300-op Mimi graph waits for most of it.
- MLX's CPU backend needed three changes to Mimi's encoder (busy M4 Pro: its transformer 23 →
  14 ms a frame):
  - `ComposedLayerNorm`: MLXFast.layerNorm's CPU fallback is ~8x slower. Only Mimi's
    transformers use `Norm.layerNorm`; the LM's depformer slice norms keep MLXFast.
  - Mimi's transformer linears are stored column-major, so `Linear`'s x · wᵀ hands sgemm a
    row-major operand: ~7x faster at two rows.
  - ELU and the tanh GELU are written out: MLXNN's activations are compiled functions, and iOS
    cannot compile CPU kernels (`[Compiled::eval_cpu] CPU compilation not supported on the
    platform`). Never `compile`, or call a compiled MLXNN activation, in code that runs on the
    CPU.
- On the A15's CPU, Mimi takes 20 ms to encode a frame alone, 28 beside the GPU LM and 35–45
  once the phone is hot; 17–19 at 2 frames per call.
- Tried and dropped: rustymimi is no faster (59 / 53 ms per frame on the same busy Mac);
  decoding on the GPU puts 16 ms more on the GPU, which bounds the LM; decoding on the CPU in
  MLX (a matmul-plus-overlap-add transposed conv, as MLX's CPU one is a naive loop at 35–65 ms a
  layer) took 19–25 ms a frame on the A15, 15–17 at 2 frames per call, and is replaced by the
  Neural Engine.

### Chunks

`Translator.step` takes k whole frames: k LM steps one after another, with Mimi encoding the k
frames in one call beside the first and decoding the audio of the LM steps finished so far
beside the last, one Neural Engine call per frame (the step's own last LM step is decoded in
the next step). A frame is a few rows per op, so
Mimi's per-op overhead dominates and its cost per call grows much slower than its length; a
chunk cuts its CPU time per frame, for up to (2k − 2) × 80 ms more latency: the first frame of a chunk waits for the rest,
and the step may take k × 80 ms.

- Busy M4 Pro CPU, Mimi encode per frame: k = 1 26 ms, 2 19, 4 16, 8 11.
- `WindowKVCache` keeps the last `context − 1` (249) steps and masks each query to the 250 steps
  ending at itself, as Mimi was trained and `mimi_mlx.py` encodes. moshi-swift's ring of 250
  steps let a k-step call overwrite keys its first queries still needed: once the ring was full
  (frame 125), 2-frame calls changed 16% of frames' codes. With the window cache, codes for
  k = 1, 2 and 4 are identical over 800 frames of `leon.wav`.
- `translate` at `--chunk` 1, 2 and 3 gives the same text word for word up to where k = 1 stops
  (a run stops only at the end of a step, so k ≥ 2 can add up to k − 1 frames: one more word on
  `leon.wav`); so did a review's mixed chunk sizes (3, 1, 2, 5, …).

## Speed and memory

The M4 Pro numbers need an idle machine: with the cache build running (`cache_codes`), CPU Mimi
took ~55 ms per frame there, more than on the A15, and the all-GPU runtime had dropped from 25
to 80 ms/frame. Without compilation the LM was 3.4x slower: compile the per-layer regions and the
per-slice depformer steps as `Decode.swift` does, each over fixed shapes
([mlx-inference](mlx-inference.md#memory)).

Footprint after load for the 3B: LM 1.96 GB, Mimi's encoder 0.18 GB (fp32, MLX) and its
decoder 0.09 GB (fp16, Core ML; the MLX decoder had been another 0.15 GB), KV cache 0.34 GB (28
layers, 3,000 frames, bf16, allocated in full at the first frame), runtime buffers. Warmup runs
silent frames until Mimi decode has run once, so the first timed frames pay no kernel builds.
`Translator.Timing` splits a step as the driving thread spends it: LM start (graph build and
Metal encoding), Mimi decode, Mimi encode, LM wait. LM wait near 0 means the CPU codec is the
bottleneck.

Leaks: none in the runtime. Over 10 minutes on the phone, MLX active memory stayed at 1,628 MB
and its buffer cache at 86 MB; 6,000 frames of CPU Mimi on the Mac kept a flat footprint, and an
autorelease pool per frame changed nothing. The app's footprint did grow ~1.6 MB a minute,
always on the frame after a progress line: its on-screen log, a text view one line taller every
10 s. Showing only the newest 10 lines, the footprint stayed at 1,766 MB from minute 2 to 5.

## Device test

- Phone: iPhone 13 Pro Max (A15, 6 GB, iOS 27.0.1), wired to the Mac, developer mode on.
  Commands are in the [README](../../inference/swift/README.md#iphone).
- Signing: team `6R79BB49DG` with automatic provisioning (`-allowProvisioningUpdates`) signs the
  app with the increased-memory-limit entitlement. A new developer certificate must be trusted on
  the phone once (Settings > General > VPN & Device Management) before `devicectl` can launch it.
- `devicectl device copy to` puts the model into the app's `Documents/model/` (2.6 GB in about a
  minute over USB). Results are `Documents/bench-<time>.{jsonl,wav,txt}`.
- Result (2026-11-01, all on the GPU, before Mimi moved to the CPU), upstream 3B q4 FR→EN,
  `leon.wav` looped, 10 minutes back to back
  (`--unpaced`; the phone cannot keep pace, so pacing changes nothing), wired and charging,
  thermal state "serious" from the first frame (warm from a 30 s run before):

  | | min 1 | min 4–10 |
  |---|---|---|
  | frame p50 | 135 ms | 193–203 ms |
  | Mimi encode / main / depformer / Mimi decode | 17 / 52 / 48 / 18 ms | 23 / 79–82 / 67–71 / 25 ms |

  10 minutes of audio took 23.5 minutes (0.43x real time; 0.59x cold). Footprint 2.78 GB with
  1.3 GB still available (increased memory limit), growing 14 MB over the run. Every stage
  slows ~1.5x under throttling. The 80 ms budget fails on the LM alone.
- Shrunk copies of the 3B for timing (2026-11-01; scratch converter, not kept; no retraining, so
  only the 28-layer one translates): 8 codebooks per stream (`dep_q` 8, `n_q` 24, delays
  `d[0] + d[1:9] + d[17:]`, schedule `[0..7]`, depformer slices 8–15 dropped, the padding rows
  of `audio_embs` 8–15 folded into every `text_emb` row, `audio_embs` 16–31 renamed 8–23) and
  the first N main layers, q4. Paced at 12.5 frames/s on `leon.wav`, wired and charging, from
  cool; thermal state "serious" within 30–80 s in every run; frame p50 in ms:

  | Model (q4, 8 codebooks) | Runtime | min 1 | min 3 | min 5 | Lag at the end |
  |---|---|---|---|---|---|
  | 10 layers (1.14 GB) | all on the GPU | 78 | 82 | | 2 s at 3 min |
  | 10 layers | Mimi on the CPU | 58 | 73 | 85 | real time for ~3 min; 45 s at 10 min |
  | 10 layers | Mimi on the CPU, 2-frame steps | 50 | 53 | 81 | real time for ~4 min; 19 s at 10 min |
  | 10 layers, 3-layer depformer (0.89 GB) | Mimi on the CPU, 2-frame steps | 42 | 44 | 46 | real time through 10 min (79 ms in min 10) |
  | 10 layers, 3-layer depformer | Mimi encoding on the CPU, decoding on the Neural Engine, 2-frame steps | 41 | 43 | 44 | real time through 10 min (44.5 ms in min 10) |
  | 14 layers (1.30 GB) | all on the GPU | 84 | 98 | | 27 s at 3 min |
  | 14 layers | Mimi on the CPU | 64 | 94 | 98 | 41 s at 5 min |
  | 28 layers (1.85 GB) | all on the GPU | 105 | 123 | | 76 s at 3 min |
  | 28 layers | Mimi on the CPU | 84 | 137 | | 87 s at 3 min |

  With Mimi on the CPU the 10-layer step is CPU-bound: LM start / decode / encode / LM wait went
  5 / 25 / 28 / 0 ms in minute 1 and 12 / 37 / 44 / 0 in minutes 8–10, so once hot the codec alone
  exceeds 80 ms. The 28-layer one is GPU-bound (LM wait 24 → 73 ms). At 10 layers the 8 depformer
  slices (64 MB each at q4) read more than the main transformer (39 MB a layer).
- 2-frame steps (2026-11-02, `--chunk 2`) halve Mimi's CPU time per frame: decode / encode 15 /
  17 ms cold and 17 / 19 hot. The 10-layer step then turns GPU-bound: LM wait 13–16 ms a frame
  for 4 minutes, then at ~250 s the GPU throttles and it jumps to 39–45 ms, so the hot 10-layer
  LM alone takes ~75–80 ms a step. MLX's buffer cache fills its 256 MB cap (86 MB a frame at a
  time), so the footprint is 174 MB higher (1,940 MB), and flat.
- A thinner depformer (each slice keeps its first 3 of 6 layers; `depformer_num_layers` 3)
  takes the 10-layer cut from 1.14 to 0.89 GB. With 2-frame steps it held real time for all 10
  minutes (worst lag 0.54 s): 42 ms a frame cold, 52 by minute 8 with the LM wait at 4–6 ms a
  frame, then the GPU throttled at ~8.5 minutes instead of ~4 and minute 10 ran at 79 ms.
- With Mimi's decoder on the Neural Engine (2026-11-02), one frame per call, the same cut ran
  41.3 → 44.5 ms a frame for all 10 minutes, worst lag 0.11 s, with no throttle step. Per frame,
  LM start / decode / encode / LM wait: 5.6 / 6.9 / 20.0 / 8.7 ms in minute 1 and 5.8 / 6.9 /
  20.2 / 11.4 in minute 10. The decoder holds its time hot, and the CPU encode no longer slows
  either, so the step is GPU-bound throughout. Thermal state "serious" came at 223 s; an
  earlier run with 2-frame decoder calls reached it at ~110 s and throttled at ~9.3 minutes
  (75 ms in minute 10), so starting temperature matters as much as the setup. Footprint 1,492
  MB (200 MB less: the MLX decoder is gone), flat all run once each Core ML call runs in an
  autorelease pool; without one it grew 1.6 MB a minute.
- Quality of the shrinks, teacher-forced against bf16 on `leon.wav` (share of text / codebook-0
  tokens equal to bf16's): q4 g32 93 / 88%, q4 g64 92 / 85%, q3 g32 82 / 78%, q3 g64 82 / 71%,
  q2 g32 55 / 7%; q4 g64 main with a q3 g64 depformer 92 / 78%; 8 codebooks in bf16 95 / 87%,
  at q4 92 / 83%. q3 is out anywhere; dropping codebooks 8–15 costs about as much as q4. Sampled
  free runs of bf16, q4 and q4 with 8 codebooks all translate `leon.wav` well.
- Long input breaks the model, not the port: on a 64 s clip looped for 6 minutes, Python and
  Swift both start repeating after ~1.6 min and emit garbage after ~2.4 min; the iPhone run's
  text ends in garbage too.
