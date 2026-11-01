# Swift runtime

Read when touching `inference/swift/` (the MLX Swift runtime, the `hibiki` CLI, the HibikiBench
iOS app) or when measuring on a phone.
Back to [CONTEXT.md](../../CONTEXT.md) · [Layout](#layout) · [Hibiki-Zero deltas](#hibiki-zero-deltas) ·
[Parity](#parity) · [Speed and memory](#speed-and-memory) · [Device test](#device-test)

## Layout

- `inference/swift/` is a SwiftPM package pinned to mlx-swift 0.32.3 (MLX core 0.32.2; the Python
  side runs mlx 0.32.4). Build commands are in its [README](../../inference/swift/README.md).
  `swift build` cannot compile MLX's Metal shaders; build with `xcodebuild`.
- `Sources/HibikiCore/` is the library:
  - Mimi, the transformer and the rotating KV cache are vendored from
    [moshi-swift](https://github.com/kyutai-labs/moshi-swift) (MIT, commit `df64ffd`,
    `LICENSE-moshi-swift`), cut to what Mimi and the LM use.
  - `LM.swift` (config, weights, LM, depformer), `Decode.swift` (compiled one-token paths),
    `Generator.swift` (delayed streams), `Translator.swift` (one 80 ms frame through Mimi and the
    LM), `Bench.swift` (sustained-load test), `Vocabulary.swift` (reads the SentencePiece
    `.model` protobuf directly) and `Audio.swift`.
- `Sources/hibiki/` is the macOS CLI: `translate` mirrors `inference/main.py` (8 s silence tail,
  stop on text EOS or 12 special frames, seed 299792458); `bench` runs `Bench`.
- `HibikiBench/` is the iOS app: a hand-written Xcode project (synchronized folder `App/`) that
  depends on the local package, with the increased-memory-limit entitlement. The team ID is
  passed on the command line (`DEVELOPMENT_TEAM=`), not stored in the project.
- Hibiki Edge (An Quách's 1B Swift app) has no licence file, so none of its code is used here.

## Hibiki-Zero deltas

Stock moshi-swift runs none of the four; this runtime implements them, reading `config.json` and
failing on any setting it does not implement:

| Delta | Where |
|---|---|
| FFN width `hidden_scale × dim` | `HibikiConfig.transformer` / `.depformer` |
| Grouped-query attention, `kv_repeat` 2 | `Attention.projections` (Transformer.swift) |
| `rope_concat` = MLX RoPE `traditional: false` | `PositionalEmbedding.ropeConcat` |
| Learned per-slice depformer LayerNorm | `DepformerSlice.norm` |

- Depformer weights per step: the q4 file holds a full copy for each of the 16 codebook steps,
  but steps 8–15 share one weight set (`depformer_weights_per_step_schedule`). `Depformer.remap`
  loads each set once into `depformer.cores.N`: 1.96 GB of LM weights instead of 2.41 GB. MLX
  loads safetensors lazily, so the dropped copies are never read.
- mlx-swift 0.32 treats every `MLXArray` property of a module as a parameter unless its name
  starts with `_`, so Mimi's derived codebook arrays are `_embedding`, `_c2`, `_expandedWeight`.
  The Mimi checkpoint holds 32 codebooks; `Mimi.load` keeps the 16 in use.

## Parity

Against `inference/hibiki_mlx` on `inference/assets/leon.wav` (2026-11-01, scratch scripts, not
kept):

- Mimi: Swift MLX codes equal rustymimi's in 98% of frames on codebook 0, falling to 87% on
  codebook 15.
- LM, teacher-forced on the Python run's greedy tokens and rustymimi codes: 99.0% of text
  tokens and 98.6% of codebook-0 tokens equal; each further codebook matches 93–96% of the time
  given the previous ones, with no step at codebook 8 where the shared weights start. Free
  running, the first 25 frames match token for token, then a near-tie flip diverges the streams.
  Python's own compiled and eager paths are bit-identical, so the residue is the MLX version and
  kernel difference.
- Sampled translations of leon.wav read as well as the Python runtime's.

## Speed and memory

M4 Pro, 3B q4, leon.wav (`hibiki translate`): 2.9x real time, 24.7 ms/frame = Mimi encode 4.2
+ main transformer 8.0 + depformer 8.8 + Mimi decode 3.7, all on the GPU in sequence. Without
compilation it was 84 ms/frame (main 28.6, depformer 38.1): compile the per-layer regions and
the per-slice depformer steps as `Decode.swift` does, each over fixed shapes
([mlx-inference](mlx-inference.md#memory)).

Footprint after load is 2.8 GB: LM 1.96 GB, Mimi 0.37 GB (fp32), KV cache 0.34 GB (28 layers,
3,000 frames, bf16, allocated in full at the first frame), runtime buffers.

## Device test

- Phone: iPhone 13 Pro Max (A15, 6 GB, iOS 27.0.1), wired to the Mac, developer mode on.
  Commands are in the [README](../../inference/swift/README.md#iphone).
- Signing: team `6R79BB49DG` with automatic provisioning (`-allowProvisioningUpdates`) signs the
  app with the increased-memory-limit entitlement. A new developer certificate must be trusted on
  the phone once (Settings > General > VPN & Device Management) before `devicectl` can launch it.
- `devicectl device copy to` puts the model into the app's `Documents/model/` (2.6 GB in about a
  minute over USB). Results are `Documents/bench-<time>.{json,txt,wav}`.
- No result yet.
