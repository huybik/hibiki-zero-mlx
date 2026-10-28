# MLX inference

Read when touching anything under `inference/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Entry points](#entry-points) · [Runtime](#runtime) ·
[Model directories](#model-directories) · [Vendored moshi-mlx](#vendored-moshi-mlx) · [Gates](#gates) ·
[Memory](#memory)

## Entry points

- `python inference/main.py INPUT` writes `translations/<stem>_translated.wav` (repo root) and
  a matching `.txt`; `--mic` plays the translation live, capped by `--minutes` (default 30).
- `--model` takes `3b` (the `weights/` symlink to `/Volumes/data/models/hibiki/weights`) or a
  staged model directory. Text sampling is temperature 0.4, top-k 25; audio is 0.8, top-k 250;
  the seed is 299792458.
- `inference/macos/HibikiTestApp/` is a SwiftUI wrapper that runs `inference/main.py` with the
  conda Python path hard-coded. It finds the repo root by walking up to a directory holding
  `inference/main.py` and `inference/hibiki_mlx/`, and defaults to the model
  `weights/vi-step135000-mlx-bf16` when it exists. `build.sh` produces `build/Hibiki Test.app`.

## Runtime

`inference/hibiki_mlx/pipeline.py` owns load and run, plus the repo-root constants `W`
(`weights/`) and `SAMPLE` (`inference/assets/leon.wav`). `pip install -e .` installs
`hibiki_mlx` from `inference/`.

- `load()` returns the LM, its config, the SentencePiece tokenizer and two separate
  `rustymimi.Tokenizer`s. One instance cannot be borrowed by two threads, and a fresh pair
  resets streaming state, so batch callers create a new pair per file.
- `run()` overlaps a CPU Mimi encoder thread and decoder thread with the GPU LM on the main
  thread. Queues carry NumPy arrays because lazy MLX graphs are bound to their creating thread.
- One frame is 1920 samples at 24 kHz (80 ms, 12.5 Hz).
- File mode appends 8 s of silence (`tail_s`) and, once the input is consumed, stops on text
  EOS or 12 consecutive special-token frames (`PAD_STOP`).
- Mic mode keeps the LM and generator on the main thread; only Mimi encode/decode run in
  workers, which adds one frame of output latency.

## Model directories

- q4 (default): `moshi_name` ends in `.q4.safetensors` (else `hibiki.q4.safetensors`). Loading
  quantizes to 4 bits, group size 32, on layers whose input width is divisible by 32, then
  loads strictly.
- BF16: `weight_dtype: bfloat16` and `moshi_name` ending in `.bf16.safetensors`; no quantization.
- The tokenizer vocabulary must equal `text_card`; the Mimi file is `mimi_name`
  (default `mimi-pytorch-e351c8d8@125.safetensors`).
- `inference/tools/convert_mlx_q4.py` converts `weights/hibiki-pytorch-77f82164@110.safetensors`
  into `weights/hibiki.q4.safetensors`.
- `inference/tools/convert_mlx_bf16.py` stages an exact PyTorch checkpoint (for example an SFT
  `model_step*.safetensors`) into a new directory and refuses to overwrite.

## Vendored moshi-mlx

- `inference/moshi-mlx/` is a minimal LM, generator, transformer, KV cache and sampler subset,
  installed with `pip install -e ./inference/moshi-mlx`. It and the root `pyproject.toml` both
  require `mlx>=0.31`.
- Hibiki-Zero deltas: grouped-query attention via `kv_repeat`, FFN width `hidden_scale * dim`,
  `rope_concat` as non-traditional RoPE, and a learned per-slice depformer output LayerNorm.
- The depformer is the only audio head. `load_pytorch_weights` accepts both the upstream fused
  `in_proj_weight` layout and moshi 0.2.13's per-slice `in_projs.N` layout written by SFT
  checkpoints.

## Gates

- `inference/tools/verify_mlx_q4.py` translates `inference/assets/leon.wav` (French) into
  `translations/leon_mlx_q4.wav`.
- `inference/tools/bench.py --model M` reports per-stage milliseconds (Mimi encode, LM, audio head,
  Mimi decode) and a projected iPhone frame time from an assumed `--scale` (0.5). The
  projection is not a measurement. `--silence` gates rms < 0.10 and peak < 1.1.
- `inference/tools/check_swift_compat.py [dir]` replays the stock moshi-swift loader path: strict q4
  group-32 load, required config keys, tokenizer and Mimi sidecars.

## Memory

These rules hold for every MLX model here (the Hibiki LM, VieNeu, Mimi, wav2vec2), not only
`inference/`.

- Compile only over a small, fixed set of input shapes. A function under `mx.compile` is traced
  again for each new input shape, and memory grows with every one: MLX wav2vec2-base compiled and
  fed 252 clips of 252 different lengths reached a 61 GB footprint and filled 43 GB of swap
  (2026-10-27); uncompiled, the same run peaked at 3.6 GB. `mx.set_cache_limit` (1 GB in that
  run) does not bound it. Bucket lengths before compiling (`training/prep/mimi_mlx.py` pads to
  multiples of 16 frames: at most 18 shapes, 1.8 GB), or keep the function eager. A model whose
  output changes with padding needs the padding masked first (`training/prep/wav2vec2_mlx.py`
  masks wav2vec2-base's GroupNorm, positional convolution and attention keys).
- Every compiled function in the repo sees fixed shapes: the depformer step, one compile per
  slice (`moshi_mlx/models/lm.py`); the attention regions of a single-token step, with the RoPE
  offset passed as an array (`moshi_mlx/modules/transformer.py`); the samplers; the VieNeu
  frame (`training/prep/vieneu_mlx.py`); the bucketed Mimi encoder (`mimi_mlx.py`) and wav2vec2
  (`wav2vec2_mlx.py`). Hold any new compiled path to the same rule.
- MLX also keeps freed buffers across shapes; cap the cache with `mx.set_cache_limit` when shapes
  vary ([vi-tts](vi-tts.md#engine) caps the codec at 2 GB). Too small a cap costs speed instead:
  the cache builder cycles through ~36 bucketed shapes and ran 195x at 4 GB, 225x at 16 GB
  ([sft-data-caches](sft-data-caches.md#speed)). Unbucketed shapes (eager wav2vec2 over raw
  lengths) fill any cap with buffers that are never reused.
- PyTorch MPS keeps memory per input shape too: Mimi encode at batch 1 over per-clip lengths
  reached 28–30 GB; 16-frame length buckets held it at 8 GB.
- Watch a run with `footprint -p <pid>` (RSS hides compressed pages) and `sysctl vm.swapusage`.
