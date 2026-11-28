# Swift runtime

Read when touching `inference/swift/` (the Swift runtime, the `hibiki` CLI, the HibikiBench iOS
app) or when measuring on a phone. Its LM is the phone student's in Core ML (`NeuralLM`):
[lm-npu](lm-npu.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Layout](#layout) · [Parity](#parity) ·
[Mimi's encoder](#mimis-encoder) · [Speed and memory](#speed-and-memory) ·
[Device test](#device-test)

## Layout

- `inference/swift/` is a SwiftPM package pinned to mlx-swift 0.32.3 (MLX core 0.32.2; the Python
  side runs mlx 0.32.4). Build commands are in its [README](../../inference/swift/README.md).
  `swift build` cannot compile MLX's Metal shaders; build with `xcodebuild`.
- `Sources/HibikiCore/` is the library:
  - Mimi's encoder and its transformer are vendored from
    [moshi-swift](https://github.com/kyutai-labs/moshi-swift) (MIT, commit `df64ffd`,
    `LICENSE-moshi-swift`), cut to Mimi's streaming encoder (`encodeStep`) in MLX on the GPU.
    `MimiQuantizer.swift` is Mimi's residual vector quantizer. Its ring KV cache is replaced by
    `WindowKVCache` ([Mimi's encoder](#mimis-encoder)).
  - `NeuralDecoder.swift` runs Mimi's decoder on the Neural Engine, one frame per call, from the
    compiled Core ML model `mimi-decoder.mlmodelc` in the model directory
    ([mimi-npu](mimi-npu.md)).
  - `HibikiConfig.swift` (`HibikiConfig`, the model directory's `config.json`; `HibikiError`;
    `URL.modelFile`, every check that a directory holds a file), `NeuralLM.swift` (the Core ML LM,
    [lm-npu](lm-npu.md); its host side, the embedding sums, input rows, Gumbel noise and output
    reads, in `NeuralLM+Host.swift`), `Generator.swift` (the delayed streams over it),
    `Sampler.swift` (top-k on the host, `SplitMix64`), `Translator.swift` (one 80 ms frame per step: the LM on its
    own thread, on the Neural Engine by default, beside it Mimi encoding on the GPU and decoding
    on the Neural Engine), `Bench.swift` (sustained-load test), `Vocabulary.swift` (reads the
    SentencePiece `.model` protobuf directly, and `TextDecoder`) and `Audio.swift` (file reading,
    the WAV writer, and `Leveller`, the input level stage `Translator.step` runs before Mimi:
    Python's `hibiki_mlx/level.py`, equal sample for sample; `reset` starts its stream again;
    [sft-evaluation](sft-evaluation.md#input-level)).
- Only a directory `export_lm_coreml.py` wrote runs: `HibikiConfig.validate` refuses a
  `config.json` without `lm_coreml`, or whose `audio_heads` is not `parallel` or
  `depformer_parallel`. `config.json` must also name `tokenizer_name` and `lm_gen_config`, which
  sets the samplers. MLX LMs (upstream's 3B, the two-way model) run in Python
  ([mlx-inference](mlx-inference.md)): on the A15's GPU the 3B misses the 80 ms frame even cut
  down ([Device test](#device-test)).
- `Translator.Output.text` is what the frame's token completes: `TextDecoder` assembles UTF-8
  from byte-fallback pieces (`<0xE1>`, type BYTE) across frames, as `TextDecoder` in
  `hibiki_mlx/text.py` does, an invalid sequence as U+FFFD; `flush` returns the rest. Fed the same
  token streams (2026-11-14, scratch harness compiling `Vocabulary.swift`: 364 streams on both
  tokenizers, PADs between pieces), it returns Python's text call for call, and the same whole
  text at 2, 3 and 5 tokens per call.
- `Translator.endInput()` ends a file's input: one LM step whose 16 source codes are all the
  source EOS (`card`) while Mimi codes a silent frame, as `file_mode.translate` does.
- `Sources/hibiki/` is the macOS CLI: `translate` mirrors `inference/main.py` (the clip
  zero-padded to whole frames, `endInput`, silence to 250 frames after the clip, stop at the
  first frame after `endInput` with the text EOS, seed 299792458; it has no quiet stop, so it
  runs the whole tail where main.py stops after 5 s without a word once the source has ended,
  and its speech ends `max_delay` − 1 frames before main.py's, which steps on past the EOS for
  the frames before it ([mlx-inference](mlx-inference.md#runtime)); `--model` defaults to
  `weights/student_slices14/pal6_snap6_ane6f`, the slices release); `bench` runs `Bench`, which
  streams `<base>.jsonl` (one sample per frame) and `<base>.wav` to disk as it runs and writes
  `<base>.txt` (summary, text) at the end. `Audio.Writer` writes 16-bit WAV clipped like
  `sphn.write_wav` and rewrites the header after every step: a run killed at frame 153 left 153
  samples and a valid 12.08 s WAV.
- `HibikiBench/` is the iOS app: a hand-written Xcode project (synchronized folder `App/`) that
  depends on the local package, with the increased-memory-limit entitlement. `--model NAME` reads
  `Documents/NAME/`, so several models can stay on the phone; `--clip NAME` reads Documents/NAME
  instead of `clip.wav`, and `--lm-units` places the LM, as the CLI's does. The team ID is
  passed on the command line (`DEVELOPMENT_TEAM=`), not stored in the project. It caps MLX's
  buffer cache at 256 MB through `Translator(cacheLimit:)` (freed buffers count toward the
  app's memory limit) and runs the bench on a user-interactive thread. iOS refuses GPU work
  from the background, so the app must stay in the foreground; locking the phone ends a run.
- The public API is `Translator`, `Bench`, `Audio`, `Sampler`, `Vocabulary`, `HibikiConfig` and
  `HibikiError`; the vendored and LM types are internal (`@testable import` reaches them).
- Hibiki Edge (An Quách's 1B Swift app) has no licence file, so none of its code is used here.
- Speech Live, the iPhone app in [onevoice-ai-challenge](https://github.com/anthoai97/onevoice-ai-challenge)
  (`apps/ios/`), copies `HibikiCore` by hand, adding `Translator.mlxMemoryMB()` and keeping the
  −20 dBFS level stage (+14 dB cost the phone EN→VI). It downloads the heads student's Core ML
  `release/student_heads_coreml6` or the slices student's `release/student_slices_coreml6`
  ([slices-phone](slices-phone.md)) and samples as `lm_gen_config` says
  ([student-palettize](student-palettize.md#release-and-phone)). Its screen shares the GPU with
  Mimi's encode, which a hot phone runs short of ([lm-npu](lm-npu.md#speed)).

## Parity

- Mimi: Swift MLX codes equal rustymimi's in 98% of frames on codebook 0, falling to 87% on
  codebook 15 (`inference/assets/leon.wav`, 2026-11-01). Against the training caches' encoder
  (`training/cache/mimi_mlx.py`, whole clip) on the same 800 frames, they are equal on 100% of
  frames in codebooks 0–9 and 99% in 10–15 (2026-11-02), both measured on MLX's CPU; the GPU's
  codes equal the CPU's on every frame (760 frames, all 16 codebooks; 2026-11-20).
- The Core ML LM's agreement with the MLX runtime is in [lm-npu](lm-npu.md#precision).

## Mimi's encoder

`Translator.step` queues LM step t on the LM's thread, submits frame t's encode to the GPU
(`asyncEval`), decodes the audio of step t − 1 on the Neural Engine ([mimi-npu](mimi-npu.md)),
collects the codes (`Timing.encodeWait`), then waits for the LM, as `file_mode.py` overlaps
rustymimi threads with the LM. The encode goes first: the Core ML LM holds the Neural Engine at
the start of its frame, and a decode queued behind it left the CPU idle (the Mac's frame 26 →
18.5 ms, [lm-npu](lm-npu.md#runtime)). An LM step's audio is decoded during the next one, so it
comes out after its text; `flush` returns the audio and text left when the input ends (the CLI's
`translate` and `Bench` call it).

- One thread drives MLX: mlx-swift 0.32 holds a global lock through every `eval`, waiting
  included, so MLX work on a second thread takes turns with the first.
- MLX's `eval` blocks while more than 10 tasks are in flight; a task is a Metal command buffer,
  which MLX closes every 20 ops or 40 MB on phones (40–50 on Macs). With those defaults an MLX LM
  step ran almost entirely inside `asyncEval` (34 of 38 ms on the A15). `Translator.init` sets
  `MLX_MAX_OPS_PER_BUFFER=400` and `MLX_MAX_MB_PER_BUFFER=1000000` unless already set. MLX reads
  them once, when the first array or `Memory` setting creates its Metal device, so the translator
  must come before anything else touches MLX; the CLI seeds after creating it.
- GPU kernels: MLXFast's fused layer norm, MLXNN's compiled GELU and ELU, and the transformer
  linears as the checkpoint stores them (the GPU's gemv wants them so).
- `Mimi.encodeStep` compiles the encode over the stream's state (`streamState`: each conv's held
  input and each layer's keys and values) after a session's first call, which pads and runs
  eagerly; one trace, since every call is one frame. So `WindowKVCache` holds its 249 steps in
  fixed buffers from the first call (zeros, masked), and RoPE's offset and the mask are inputs.
  On the A15 it cut the driving thread's build and submit 11.8 → 9.5–10.6 ms a frame and the
  footprint 725 → 541 MB (MLX's buffer cache 256 → 68 MB).
- `WindowKVCache` keeps the last `context − 1` (249) steps and masks each query to the 250 steps
  ending at itself, as Mimi was trained and `mimi_mlx.py` encodes; a frame is two steps of the
  transformer (25 Hz before the downsample). moshi-swift's ring of 250 steps let a call of several
  steps overwrite keys its first queries still needed: once the ring was full (frame 125),
  2-frame calls changed 16% of frames' codes.
- **Mimi on the CPU is no option beside the Core ML LM**: on the A15 its encode took 37 ms a frame
  cold and 64 ms by minute 10, so the frame reached 74.6 ms (past 80 started hot), where the GPU
  holds 53 ms for 10 minutes ([lm-npu](lm-npu.md#speed)). It also needed CPU-only kernels: iOS
  cannot compile CPU kernels (`[Compiled::eval_cpu] CPU compilation not supported on the
  platform`), so never `compile`, or call a compiled MLXNN activation, in code that runs on the CPU.
- **Several frames per step did worse**: 2-frame steps cut the CPU encode per frame by about a
  third, but beside the Core ML LM (Mimi on the CPU, started hot) the frame went 78–96 ms, a
  step's two decodes queuing behind the LM on the Neural Engine ([lm-npu](lm-npu.md#speed)).
- **fp16 / mixed GPU encode failed the chrF gate** (2026-11-27, M4 Pro, `pal6c4000_ane6`,
  `/Volumes/data/hibiki-zero-evals/mac/mimi-fp16/`): full fp16 (653 vs 797 MB, frame p50 14.0 vs
  16.7 ms) and mixed (`encoder_transformer` + `downsample` fp16: 737 MB, 14.4 ms) both dropped
  20×2 FLEURS dev chrF and EN→VI EOS (fp32 **43.51**, EOS 7/6 vs mixed 41.67, EOS 7/2 and fp16
  41.81, EOS 5/**0**); kept fp32.
- Tried and dropped: rustymimi is no faster (59 / 53 ms per frame on the same busy Mac); Mimi's
  decoder in MLX, on the GPU or the CPU, is replaced by the Neural Engine
  ([mimi-npu](mimi-npu.md#core-ml)).

## Speed and memory

The M4 Pro numbers need an idle machine: with the cache build running (`cache.en_speech`), Mimi's
encode took ~55 ms per frame there, more than on the A15.

Mimi's encoder takes 0.18 GB (fp32, MLX) and its decoder 0.09 GB (fp16, Core ML; the MLX decoder
had been another 0.15 GB); the heads release's whole app footprint on the A15 is 541 MB
([lm-npu](lm-npu.md#speed)). Warmup runs silent frames until Mimi decode has run once, so the
first timed frames pay no kernel builds. `Translator.Timing` splits a step as the driving thread
spends it: LM start (queueing the frame), Mimi decode, Mimi encode (the level stage, building
and submitting), encode wait, LM wait; `lmTrunk`, `lmHeads` and `lmHost` are the LM's own time
on its thread. Encode wait near 0 means the GPU's encode fits beside the decode.

- One session has no length limit: `Bench` runs its 10 minutes (7,500 frames) through one
  session, where rustymimi stops at 8,192 positions ([mlx-inference](mlx-inference.md#runtime)).
- Leaks: none in the runtime; each Core ML call runs in an autorelease pool (without one the
  footprint grew 1.6 MB a minute). The app's footprint did grow ~1.6 MB a minute, always on the
  frame after a progress line: its on-screen log, a text view one line taller every 10 s. It shows
  only the newest 10 lines.

## Device test

- Phone: iPhone 13 Pro Max (A15, 6 GB, iOS 27.0.1), wired to the Mac, developer mode on.
  Commands are in the [README](../../inference/swift/README.md#iphone).
- Signing: team `6R79BB49DG` with automatic provisioning (`-allowProvisioningUpdates`) signs the
  app with the increased-memory-limit entitlement. A new developer certificate must be trusted on
  the phone once (Settings > General > VPN & Device Management) before `devicectl` can launch it.
- `devicectl device copy to` puts the model into the app's `Documents/model/` (2.6 GB in about a
  minute over USB). Results are `Documents/bench-<time>.{jsonl,wav,txt}`.
- The phone students' results are in [lm-npu](lm-npu.md#speed).
- **An MLX LM on the A15's GPU is too slow** (2026-11-01 and 02, `leon.wav` looped, wired): upstream's
  3B q4 ran 0.43x real time (frame p50 135 ms, 193–203 hot; Mimi encode / main / depformer /
  Mimi decode 23 / 79–82 / 67–71 / 25 ms hot). Shrunk copies (no retraining: the first 10, 14 or
  28 main layers, 8 codebooks, q4) missed the 80 ms frame at 14 and 28 layers; only 10 layers with
  a 3-layer depformer held real time for 10 minutes (41–44 ms, decoder on the Neural Engine). Hence
  the Core ML phone students ([phone-student](phone-student.md)).
- Quantizing those shrinks, teacher-forced against bf16 on `leon.wav` (share of text /
  codebook-0 tokens equal to bf16's): q4 g32 93 / 88%, q3 g32 82 / 78%, q2 g32 55 / 7%; 8
  codebooks in bf16 95 / 87%. q3 is out anywhere; dropping codebooks 8–15 costs about as much as
  q4.
- Long input breaks the model, not the port: on a 64 s clip looped for 6 minutes, Python and
  Swift both started repeating after ~1.6 min and emitted garbage after ~2.4 min.
