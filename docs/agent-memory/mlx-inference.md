# MLX inference

Read when touching anything under `inference/` except `inference/swift/`
([swift-runtime](swift-runtime.md)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Entry points](#entry-points) · [Runtime](#runtime) ·
[Model directories](#model-directories) · [Vendored moshi-mlx](#vendored-moshi-mlx) · [Gates](#gates) ·
[Memory](#memory)

## Entry points

- `python inference/main.py INPUT` writes `translations/<stem>_translated.wav` (repo root) and
  a matching `.txt`; `--mic` plays the translation live, capped by `--minutes` (default 30).
- `--model` takes `3b` (`weights/hibiki-zero-3b-q4`, upstream's 3B staged by `convert_mlx.py
  --q4` from the files in `weights/`, the symlink to `/Volumes/data/models/hibiki/weights`) or
  any staged model directory. Sampling is the model's `config.json` `lm_gen_config`; the seed,
  299792458, is set after loading, so a file samples as the gate's row of that seed does.
- `inference/macos/HibikiTestApp/` is a SwiftUI wrapper that runs `inference/main.py` with the
  conda Python path hard-coded. It finds the repo root by walking up to a directory holding
  `inference/main.py` and `inference/hibiki_mlx/`, and defaults to the model
  `weights/vi-step135000-mlx-bf16` when it exists. `build.sh` produces `build/Hibiki Test.app`.

## Runtime

`inference/hibiki_mlx/pipeline.py` owns load and run, plus the repo-root constants `W`
(`weights/`), `UPSTREAM` (`weights/hibiki-zero-3b-q4`) and `SAMPLE` (`inference/assets/leon.wav`).
`pip install -e .` installs `hibiki_mlx` from `inference/`.

- `load()` returns a `Model`: the LM, its config, the SentencePiece tokenizer and the text and
  audio samplers `lm_gen_config` sets. `Model.mimi()` makes a fresh `rustymimi.Tokenizer`: one
  instance cannot be borrowed by two threads, and each stream needs a fresh one, since
  rustymimi's `reset()` leaves state behind (2026-11-14: after `reset()`, the next clip's first
  frame came out different in 14 of 16 codebooks from a fresh instance's).
- `translate()` overlaps a CPU Mimi encoder thread and decoder thread with the GPU LM on the main
  thread. Queues carry NumPy arrays because lazy MLX graphs are bound to their creating thread.
  `audio=False` skips the decoder (the gate scores text only); `run()` writes its WAV and text.
- One frame is 1920 samples at 24 kHz (80 ms, 12.5 Hz).
- The source goes through the input level stage (`hibiki_mlx/level.py`, `Leveller`) before
  Mimi: a file's first channel as one stream, a mic session as another, and `tools/bench.py`'s
  frames (timed in "level + mimi encode"). It raises quiet speech toward −20 dBFS; how and why
  in [sft-evaluation](sft-evaluation.md#input-level). The evaluation (`training/finetune/
  common.py`) imports it as `inference.hibiki_mlx.level`, so the module needs only NumPy.
- `hibiki_mlx` and `moshi_mlx` are installed editable from the main checkout, so in a worktree
  run the tools with `PYTHONPATH=<worktree>/inference:<worktree>/inference/moshi-mlx`: otherwise
  they import main's packages (`main.py` finds its own `hibiki_mlx` first, not `moshi_mlx`). With
  main's older `moshi_mlx` the source EOS was code 2047 and no clip ended.
- File mode feeds a file as training's free-running evaluation does
  (`common.generate_batch`), so its scores compare with the box's: the levelled first channel
  zero-padded to whole frames, then one frame whose 16 source codes are all the source EOS
  (`card`, 2048, `LmConfig.audio_eos_token`; Mimi still codes a silent frame there, as a
  whole-file encode codes the padding), then silence to 20 s after the clip (`TAIL_S`, training's
  `tail_s`), stopping at the first text EOS, before the clip ends too, and cutting the text
  there. Upstream's own offline inference sends the source EOS the same way. Step t hears source
  frames before t, as training's forward does; moshi's `LMGen` in the evaluation hears one silent
  frame more first, so its outputs come 80 ms later throughout, and the gate adds that frame to
  each End Offset. Mic mode never sends the EOS.
- Sampling is `lm_gen_config`: every training checkpoint's config and upstream's own say
  temperature 0.8, top-k 250 for text and audio, which training's evaluation and upstream's
  inference (`run.py`, `**lm_gen_config`) use. Until 2026-11-14 the runtimes sampled text at 0.4,
  top-k 25, which makes the fine-tuned model lag and seldom end
  ([analysis](../analysis/eos_and_lag.md)); upstream's 3B now samples as upstream does.
- Text comes from `TextDecoder` (both modes): a byte-fallback piece (`<0xE1>`) carries one
  UTF-8 byte, so a character the tokenizer lacks spans frames; the decoder (an incremental UTF-8
  codec with `errors="replace"`) prints it once complete, skips `<unk>`/`</s>`/`<pad>` without
  breaking a sequence, and `flush` ends the stream. For valid UTF-8 its stream equals
  SentencePiece's `decode` of the non-special tokens, as the evaluation decodes
  (`decode_text_batch`; 2026-11-14, scratch check: the repo's ~6,500 lines of docs and
  Vietnamese sentences, on both tokenizers, PADs between pieces); an invalid run of bytes is one
  U+FFFD where SentencePiece writes one per byte. Without byte pieces its text equals the old
  piece-by-piece output. `<s>` is printed as before. Mic mode prints the flushed tail before
  `[stopping]`.
- Mic mode keeps the LM and generator on the main thread; only Mimi encode/decode run in
  workers, which adds one frame of output latency. A worker's error ends the session and is
  raised once it has stopped. Use headphones: the level stage raises the speakers' echo of the
  translation toward speech level ([input level](sft-evaluation.md#input-level)).
- rustymimi fails on long sessions: encoding a 6-minute file raised `narrow invalid args
  start + len > dim_len: [8192, 32]` at frame 2,920 (3.9 min). The Swift runtime's Mimi has no
  such limit ([swift-runtime](swift-runtime.md)).

## Model directories

- `config.json` names the MLX weights (`moshi_name`, required), the sampling (`lm_gen_config`,
  required) and, for quantized weights, `quantization`: `group_size` and `bits` for every Linear
  and Embedding, plus an entry `{"group_size", "bits"}` of its own for each layer quantized
  otherwise, keyed by its module path (mlx-lm's layout). Both runtimes quantize the bf16 model
  by it (`pipeline.quantize`, MLX's per-layer `class_predicate`; Swift `quantize(model:filter:)`
  with a per-layer tuple, mapping its depformer cores to the checkpoint's slice paths), then load
  strictly, shapes included: a record that disagrees with the weights fails to load. Without
  `quantization` the weights are bf16. Older keys (`weight_dtype`, `artifact_format`) are ignored.
- The q4 recipe (`convert_mlx.py --q4`, plan stage 6): 4 bits on every transformer linear (main
  and depformer layers), 8 bits on the 34 embeddings, the text head (`text_linear`) and each
  depformer slice's `linear_in` and `linear_out` (82 entries), group size 32 throughout. Those
  are where fewer bits hurt most: q3 on the heads made the 3B babble
  ([inference_matrix](../vision/reports/inference_matrix.md)). Quantized from bf16, so scales and
  biases are bf16. The 3B is 2.65 GB on disk (uniform q4 was 2.41 GB, BF16 7.7 GB); the 8-bit
  layers are ~0.48B of its 3.85B parameters (with its 16 depformer copies). Teacher-forced on
  bf16's sampled tokens (2026-11-14, the two-way GRPO model, 6 FLEURS dev clips both ways, 927
  frames; scratch script), the share of argmaxes equal to bf16's, text / text where bf16 writes a
  word / codebook 0 / all codebooks: the recipe 96.7 / 85.4 / 88.9 / 72.2%, uniform q4 g32 95.6 /
  82.8 / 85.5 / 66.5%, 4-bit linears with bf16 heads 96.5 / 86.1 / 89.0 / 72.1%: 8 bits on the
  heads is as good as bf16 there, and the rest of q4's loss is in the 4-bit linears.
- The tokenizer is the file `tokenizer_name` names (required, no default: the Vietnamese swap
  and the base share `text_card`, so a wrong one loads and prints the wrong words,
  [vi-tokenizer](vi-tokenizer.md#runtimes)); its vocabulary must equal `text_card`. The Mimi
  file is `mimi_name` (default `mimi-pytorch-e351c8d8@125.safetensors`).
- `inference/tools/convert_mlx.py --checkpoint --config --mimi [--tokenizer] [--q4] --out-dir`
  stages an exact PyTorch checkpoint (upstream's, or an SFT or GRPO `*_step*.safetensors`, fp32
  or bf16) into a new directory, q4 or BF16, and refuses to overwrite. It ships the tokenizer
  the checkpoint is on: its safetensors metadata `text_tokenizer` (else the base, as in
  training), from `weights/` or `--tokenizer`, whose file name must match; the directory's
  `config.json` names it. A two-way checkpoint stages with the swap and no flag. The 12.5 GB fp32
  GRPO checkpoint staged as q4 in under a minute with a 6.3 GB peak footprint (2026-11-14).
- Upstream's 3B stages into `weights/hibiki-zero-3b-q4` (README). A `weights/` from before
  2026-11-14 holds `hibiki.q4.safetensors` (uniform q4) beside upstream's PyTorch `config.json`,
  which names no MLX weights or quantization, so it no longer loads: stage it again, then delete
  that file.

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

- `inference/tools/gate.py` (`python -m inference.tools.gate --model DIR --pairs …
  [--limit N] [--seed S] --out-dir DIR`, from the repo root, since it imports `training`) scores
  a model directory as training's free-running evaluation scores a checkpoint, to gate q4 against
  BF16 (plan stage 6). It translates each row alone in file mode, the MLX seed set to `--seed` + the
  row's index, and scores each pairs file as its own set with training's `common.score_sets`
  (corpus chrF and BLEU, length ratio, empty and ended rows, the health gates). It writes
  `hypotheses.jsonl` (a row each, as it finishes) and `scores.json`. Its scores compare with the
  box's in sampling and input, not in batching: training samples rows 64 at a time. The pairs are
  `data/eval/` files ([sft-evaluation](sft-evaluation.md#evaluation-sets)), their audio paths
  relative to the repo root.
- `inference/tools/verify_mlx_q4.py` translates `inference/assets/leon.wav` (French) with
  `weights/hibiki-zero-3b-q4` into `translations/leon_mlx_q4.wav`.
- `inference/tools/bench.py --model M` reports per-stage milliseconds (Mimi encode, LM, audio head,
  Mimi decode) and a projected iPhone frame time from an assumed `--scale` (0.5). The
  projection is not a measurement. `--silence` gates rms < 0.10 and peak < 1.1.

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
  ([cache-build](cache-build.md#speed-on-the-mac)). Unbucketed shapes (eager wav2vec2 over raw
  lengths) fill any cap with buffers that are never reused.
- PyTorch MPS keeps memory per input shape too: Mimi encode at batch 1 over per-clip lengths
  reached 28–30 GB; 16-frame length buckets held it at 8 GB.
- Watch a run with `top -l 1 -pid <pid> -stats mem` (the physical footprint; RSS hides
  compressed pages) and `sysctl vm.swapusage`. `footprint -p <pid>` reports the same number but
  can hang system-wide, even on a shell (2026-11-02, for over an hour), so a watchdog that calls
  it needs a timeout.
