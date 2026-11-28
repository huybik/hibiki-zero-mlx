# Swift runtime

Hibiki-Zero's phone students in Swift, for macOS and iOS: the LM runs as Core ML, on the Neural
Engine by default, beside it Mimi encodes in MLX on the GPU and decodes on the Neural Engine. It
runs the model directories `inference/tools/export_lm_coreml.py` writes (`lm_coreml` in
`config.json`: the parallel-heads student, or a `depformer_parallel` slices student), plus Mimi's
decoder as a compiled Core ML model (`mimi-decoder.mlmodelc`, below). MLX models (upstream's 3B,
the two-way model) run in Python ([mlx-inference](../../docs/agent-memory/mlx-inference.md)).

```text
Package.swift          SwiftPM package, mlx-swift pinned
Sources/HibikiCore/    the library: Mimi's encoder, its Core ML decoder, the Core ML LM, the frame loop, Bench
Sources/hibiki/        macOS CLI: translate, bench
HibikiBench/           iOS app that runs Bench on a phone
```

## Core ML LM

The LM is exported from the bf16 staging: for the heads student two models a frame, the text
sampled between them; for a `depformer_parallel` model the trunk and one depformer model
(`lm-depformer.mlmodelc`, every slice with in-graph top-k Gumbel-max sampling)
([lm-npu](../../docs/agent-memory/lm-npu.md)). From the repo root:

```sh
# ~15 min of k-means on one core at 6 bits
python inference/tools/export_lm_coreml.py weights/student_heads14_cb16/s16k_bf16 \
  weights/student_heads14_cb16/s16k_ane6 --bits 6
# Mimi's decoder for this model's dep_q (or fewer codebooks, which the runtime accepts)
python inference/tools/export_mimi_coreml.py \
  weights/student_heads14_cb16/s16k_ane6/mimi-pytorch-e351c8d8@125.safetensors \
  weights/student_heads14_cb16/s16k_ane6 --decoder-codebooks 16
```

## macOS

`swift build` cannot compile MLX's Metal shaders, so build with `xcodebuild`, from this directory:

```sh
xcodebuild -scheme hibiki -destination 'platform=macOS' -derivedDataPath .build/xcode \
  -configuration Release -skipPackagePluginValidation build
cd ../..   # repo root, where weights/ lives
inference/swift/.build/xcode/Build/Products/Release/hibiki translate CLIP.wav
inference/swift/.build/xcode/Build/Products/Release/hibiki bench CLIP.wav --minutes 10
```

Both take `--model DIR` (default `weights/student_slices14/pal6_snap6_ane6f`, the slices
release) and sample as its `config.json` says (`lm_gen_config`). `translate` feeds the file as
`inference/main.py` does (the source EOS after the clip, then silence, stopping at the first text
EOS after the source EOS) and writes `translations/<stem>_swift.{wav,txt}`. `bench` loops the
clip for the given minutes, paced at 12.5 frames/s like a microphone unless `--unpaced`. It
writes `translations/<stem>_bench.{jsonl,wav}` as it runs and `.txt` (summary and text) at the
end (`--out BASE` to change), and prints frame times, lag, thermal state and memory per minute.

`--lm-units ane|gpu|cpu|all` (CLI and app) places the LM: the Neural Engine by default.
`translate` prints its own time per frame (main transformer, audio head, host) beside the
step's; `bench` prints their sum per minute and logs each in its `.jsonl`. Mimi's encode shows as
`encode` (the level stage, building and submitting the GPU's work) and `encode wait`
(`encodeWaitMs`, waiting for its codes after the decode)
([swift-runtime](../../docs/agent-memory/swift-runtime.md#mimis-encoder)).

## iPhone

Build and install with your own team, then copy the model and a clip into the app's Documents
folder (Finder file sharing works too) and launch it:

```sh
DEVICE=<udid from: xcrun devicectl list devices>
APP=com.huybik.hibikibench
cd HibikiBench
xcodebuild -project HibikiBench.xcodeproj -scheme HibikiBench -destination "id=$DEVICE" \
  -configuration Release -derivedDataPath ../.build/ios -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<team id> -skipPackagePluginValidation build
xcrun devicectl device install app --device $DEVICE \
  ../.build/ios/Build/Products/Release-iphoneos/HibikiBench.app
cd ../../..   # repo root
MODEL=weights/student_slices14/pal6_snap6_ane6f
(cd $MODEL && find . -type f) | sed 's|^\./||' | while read f; do
  xcrun devicectl device copy to --device $DEVICE --domain-type appDataContainer \
    --domain-identifier $APP --source $MODEL/$f --destination Documents/model/$f
done
xcrun devicectl device copy to --device $DEVICE --domain-type appDataContainer \
  --domain-identifier $APP --source CLIP.wav --destination Documents/clip.wav
xcrun devicectl device process launch --device $DEVICE --console --terminate-existing $APP \
  --minutes 10
```

This copies every file of the slices release's directory; set `MODEL` for another (only its
`config.json`, the files it names, the `.bin` tables and the `.mlmodelc` directories are read).

`--model NAME` reads `Documents/NAME/` instead of `Documents/model/`, to keep several models
on the phone, and `--clip NAME` reads `Documents/NAME` instead of `clip.wav`; `--unpaced` and
`--lm-units` work as in `bench`. The phone must be unlocked, and a new developer certificate trusted once under
Settings > General > VPN & Device Management. Keep the app in the foreground: iOS stops GPU
work in the background. The app keeps the phone awake and writes
`Documents/bench-<time>.{jsonl,wav}` as it runs and `.txt` at the end; copy them back with
`devicectl device copy from`. The app stays open after a run, so the console does too; stopping
the console ends the app, and the phone may lock.
