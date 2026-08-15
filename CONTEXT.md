# Hibiki MLX — project context

This repository maintains two paths only:

1. q4 MLX inference for Hibiki-Zero/Hibiki-M on Apple Silicon;
2. base-start, full-model Vietnamese-to-English SFT on CUDA.

Obsolete experiments and generated research artifacts were removed from the
active tree in August 2026. Their history remains available in Git.

## Inference

- `main.py` is the CLI for file and microphone translation.
- `hibiki_mlx/pipeline.py` owns model loading and the three-thread
  Mimi-encode → LM → Mimi-decode pipeline.
- `moshi-mlx/` is the minimal vendored MLX language-model runtime. Its required
  Hibiki deltas are GQA (`kv_repeat`), configurable `hidden_scale`,
  `rope_concat`, and per-slice depformer output LayerNorm.
- Supported weights are q4 with `group_size=32`. `3b` resolves to `weights/`;
  `1b` resolves to `weights/hibiki-m-mlx-q4/`.
- Each codec thread owns a separate `rustymimi.Tokenizer`. Queues carry NumPy
  arrays because lazy MLX graphs cannot move across creating threads.
- File inference adds an 8-second silence tail and stops after 12 sustained PAD
  frames to flush translation lag without extended hallucination.
- Runtime gates are `scripts/verify_mlx_q4.py` and
  `scripts/bench.py --model {3b,1b} --silence`.
- `scripts/check_swift_compat.py` requires strict q4 group-size-32 reload plus
  valid config, tokenizer, and Mimi sidecars.

## Training

- `finetune/train.py` always trains every model parameter. There is no LoRA,
  warm-start adapter, replay sampler, or alternate batch scheduler.
- Training starts from `weights/hibiki-pytorch-77f82164@110.safetensors`.
  `--resume-checkpoint` is only for interruption recovery within the same run.
- CUDA uses fp32 master weights, bf16 autocast, fused AdamW, causal SDPA, fixed
  length-sorted batches, and `--max-frames 280` in the current recipe.
- Text prefix PAD weight defaults to 0.5. Content/EOS remain weight 1.0.
- `finetune/common.py` owns cached data, losses, schedules, exact full-model
  checkpoint I/O, greedy generation, and metrics. Checkpoint loading rejects
  missing and unexpected tensor keys.
- `finetune/cache_phomt_stream.py` builds the published PhoMT cache directly
  from parquet. `remote_dataset/download_fleurs_vi_en.py` →
  `finetune/build_pairs.py` → `finetune/cache_codes.py` builds FLEURS inputs.
- `finetune/validate.py` is teacher-forced diagnostics only.
  `finetune/eval.py` free-running chrF plus nonempty/EOS/loop/length gates select
  checkpoints; `sacrebleu` is required.
- `finetune/hf_sync.py` uploads the latest complete model/trainer pair without
  deleting remote files or rewriting Hub history.

The exact launch command is in `docs/training_plan.md`; qualification thresholds
are in `docs/validation_plan.md`.

## Environment

- Local Python work uses `/opt/homebrew/Caskroom/miniconda/base/bin/python`.
- Inference requires MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and
  sounddevice for microphone mode.
- Training additionally requires a CUDA-compatible PyTorch build, `moshi`
  0.2.13, safetensors, sacrebleu, datasets, soundfile, pyarrow, and
  huggingface-hub.
