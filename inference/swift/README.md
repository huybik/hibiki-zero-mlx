# Swift runtime

Hibiki-Zero in MLX Swift, for macOS and iOS: the LM runs on the GPU, beside it Mimi encodes on
the CPU and decodes on the Neural Engine. The parallel-heads student's LM runs as Core ML
instead, on the Neural Engine by default ([below](#core-ml-lm)). It runs the same staged model directories as the
Python runtime (`convert_mlx.py` outputs, quantized as their `config.json` records, such as
upstream's `weights/hibiki-zero-3b-q4`), plus Mimi's decoder as a compiled Core ML model
(`mimi-decoder.mlmodelc`, made once per model directory, below). Fed the same tokens, it picks the same text token as the
Python runtime 99% of the time; sampled runs differ token by token but translate as well ([swift-runtime](../../docs/agent-memory/swift-runtime.md#parity)).

```text
Package.swift          SwiftPM package, mlx-swift pinned
Sources/HibikiCore/    the library: Mimi's encoder, its Core ML decoder, the Hibiki-Zero LM, the frame loop, Bench
Sources/hibiki/        macOS CLI: translate, bench
HibikiBench/           iOS app that runs Bench on a phone
```

## macOS

`swift build` cannot compile MLX's Metal shaders, so build with `xcodebuild`, from this directory:

```sh
xcodebuild -scheme hibiki -destination 'platform=macOS' -derivedDataPath .build/xcode \
  -configuration Release -skipPackagePluginValidation build
cd ../..   # repo root, where weights/ lives
# Mimi's decoder for this model's dep_q (16 for upstream Hibiki-Zero, 8 for the 8-codebook cuts)
python inference/tools/export_mimi_coreml.py weights/mimi-pytorch-e351c8d8@125.safetensors \
  weights/hibiki-zero-3b-q4 --decoder-codebooks 16
inference/swift/.build/xcode/Build/Products/Release/hibiki translate inference/assets/leon.wav
inference/swift/.build/xcode/Build/Products/Release/hibiki bench inference/assets/leon.wav --minutes 10
```

Both take `--model DIR` (default `weights/hibiki-zero-3b-q4`) and sample as its `config.json`
says (`lm_gen_config`). `translate` feeds the file as `inference/main.py` does (the source EOS
after the clip, then silence, stopping at the first text EOS after the source EOS) and writes
`translations/<stem>_swift.{wav,txt}`. `bench` loops the clip for the given
minutes, paced at 12.5 frames/s like a microphone unless `--unpaced`. It writes
`translations/<stem>_bench.{jsonl,wav}` as it runs and `.txt` (summary and text) at the end
(`--out BASE` to change), and prints frame times, lag, thermal state and memory per minute.

Both take `--chunk K` (default 1): K frames per step, so Mimi encodes K frames per call. That
costs less CPU time per frame and adds up to (2K − 2) × 80 ms of latency. The translation is
the same, but `translate` stops only at the end of a step, up to K − 1 frames later than with
K = 1. The decoder runs one frame per call on the Neural Engine, whatever K is.

## Core ML LM

The parallel-heads student (`"audio_heads": "parallel"`) runs only as Core ML: its LM exported from
the bf16 staging, two models a frame with the text sampled between them
([lm-npu](../../docs/agent-memory/lm-npu.md)). From the repo root:

```sh
# ~15 min of k-means on one core at 6 bits
python inference/tools/export_lm_coreml.py weights/student_heads14_cb16/s16k_bf16 \
  weights/student_heads14_cb16/s16k_ane6 --bits 6
python inference/tools/export_mimi_coreml.py \
  weights/student_heads14_cb16/s16k_ane6/mimi-pytorch-e351c8d8@125.safetensors \
  weights/student_heads14_cb16/s16k_ane6 --decoder-codebooks 16
inference/swift/.build/xcode/Build/Products/Release/hibiki translate CLIP.wav \
  --model weights/student_heads14_cb16/s16k_ane6
```

`--lm-units ane|gpu|cpu|all` (CLI and app) places the Core ML LM: the Neural Engine by default.
`translate` prints its own time per frame (main transformer, heads, host) beside the step's;
`bench` prints their sum per minute and logs each in its `.jsonl`.

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
MODEL=weights/hibiki-zero-3b-q4
(cd $MODEL && find . -type f) | sed 's|^\./||' | while read f; do
  xcrun devicectl device copy to --device $DEVICE --domain-type appDataContainer \
    --domain-identifier $APP --source $MODEL/$f --destination Documents/model/$f
done
xcrun devicectl device copy to --device $DEVICE --domain-type appDataContainer \
  --domain-identifier $APP --source inference/assets/leon.wav --destination Documents/clip.wav
xcrun devicectl device process launch --device $DEVICE --console --terminate-existing $APP \
  --minutes 10
```

This copies every file of upstream's q4 model directory; set `MODEL` for another (only its
`config.json`, the files it names, `lm-embeddings.bin` and the `.mlmodelc` directories are
read).

`--model NAME` reads `Documents/NAME/` instead of `Documents/model/`, to keep several models
on the phone, and `--clip NAME` reads `Documents/NAME` instead of `clip.wav`; `--chunk K`,
`--unpaced` and `--lm-units` work as in `bench`. The phone must be unlocked, and a new developer certificate trusted once under
Settings > General > VPN & Device Management. Keep the app in the foreground: iOS stops GPU
work in the background. The app keeps the phone awake and writes
`Documents/bench-<time>.{jsonl,wav}` as it runs and `.txt` at the end; copy them back with
`devicectl device copy from`. The app stays open after a run, so the console does too; stopping
the console ends the app, and the phone may lock.
