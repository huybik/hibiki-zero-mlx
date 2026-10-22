# MLX inference

Read when touching `main.py`, `hibiki_mlx/`, `moshi-mlx/`, `scripts/` or `macos/HibikiTestApp/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Entry points](#entry-points) · [Runtime](#runtime) ·
[Model directories](#model-directories) · [Vendored moshi-mlx](#vendored-moshi-mlx) · [Gates](#gates)

## Entry points

- `main.py INPUT` writes `translations/<stem>_translated.wav` and a matching `.txt`;
  `main.py --mic` plays the translation live, capped by `--minutes` (default 30).
- `--model` takes `3b` (the `weights/` symlink to `/Volumes/data/models/hibiki/weights`) or a
  staged model directory. Text sampling is temperature 0.4, top-k 25; audio is 0.8, top-k 250;
  the seed is 299792458.
- `macos/HibikiTestApp/` is a SwiftUI wrapper that runs `main.py` with the conda Python path
  hard-coded; `build.sh` produces `build/Hibiki Test.app`.

## Runtime

`hibiki_mlx/pipeline.py` owns load and run.

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
- `scripts/convert_mlx_q4.py` converts `weights/hibiki-pytorch-77f82164@110.safetensors` into
  `weights/hibiki.q4.safetensors`.
- `scripts/convert_mlx_bf16.py` stages a PyTorch checkpoint into a new directory and refuses to
  overwrite. For `head: parallel_v1` it accepts only a `hibiki_parallel_bf16_export_v1` file
  with all-BF16 tensors, as written by `student.export_parallel`.

## Vendored moshi-mlx

- A minimal LM, generator, transformer, KV cache and sampler subset, installed with
  `pip install -e ./moshi-mlx`. It requires `mlx>=0.31`, stricter than the root
  `pyproject.toml`.
- Hibiki-Zero deltas: grouped-query attention via `kv_repeat`, FFN width `hidden_scale * dim`,
  `rope_concat` as non-traditional RoPE, and a learned per-slice depformer output LayerNorm.
- `LmConfig` selects exactly one audio head: `ar` (depformer, one pass) or `parallel_v1`
  (shape `(512, 2, 8)`, `head_passes` 1 or 2, no classifier-free guidance, needs the previous
  raw head output). See `moshi-mlx/moshi_mlx/models/lm.py`.

## Gates

- `scripts/verify_mlx_q4.py` translates `assets/samples/leon.wav` into
  `translations/leon_mlx_q4.wav`.
- `scripts/bench.py --model M` reports per-stage milliseconds (Mimi encode, LM, audio head,
  Mimi decode) and a projected iPhone frame time from an assumed `--scale` (0.5). The
  projection is not a measurement. `--silence` gates rms < 0.10 and peak < 1.1.
- `scripts/check_swift_compat.py [dir]` replays the stock moshi-swift loader path: strict q4
  group-32 load, required config keys, tokenizer and Mimi sidecars. It reports `parallel_v1`
  packs as unsupported by stock Swift.
