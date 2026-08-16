# Hibiki MLX — project context

This repository maintains two paths only:

1. q4 MLX inference for Hibiki-Zero on Apple Silicon;
2. base-start, full-model Vietnamese-to-English SFT on CUDA.

The mobile target is a future distilled 1B Hibiki-Zero student with parallel
target-codebook heads. It is not implemented yet.

Historical design, vision, and report documentation is retained under `docs/`.
Generated FLEURS data is excluded from both the active tree and Git history.

## Inference

- `main.py` is the CLI for file and microphone translation.
- `hibiki_mlx/pipeline.py` owns model loading and the three-thread
  Mimi-encode → LM → Mimi-decode pipeline.
- `moshi-mlx/` is the minimal vendored MLX language-model runtime. Its required
  Hibiki deltas are GQA (`kv_repeat`), configurable `hidden_scale`,
  `rope_concat`, and per-slice depformer output LayerNorm.
- Supported weights are q4 with `group_size=32`. `3b` resolves to `weights/`;
  custom staged Hibiki-Zero directories can be passed explicitly.
- Each codec thread owns a separate `rustymimi.Tokenizer`. Queues carry NumPy
  arrays because lazy MLX graphs cannot move across creating threads.
- File inference adds an 8-second silence tail and stops after 12 sustained PAD
  frames to flush translation lag without extended hallucination.
- Runtime gates are `scripts/verify_mlx_q4.py` and
  `scripts/bench.py --model 3b --silence`.
- `scripts/check_swift_compat.py` requires strict q4 group-size-32 reload plus
  valid config, tokenizer, and Mimi sidecars.

## Training

- Start every new-agent or new-pod training session with
  `docs/finetune.md`. It is the copy-paste handoff for fresh setup, artifact
  restore, preflight/smoke, the stop-before-launch boundary, and exact recovery.
- `finetune/train.py` always trains every model parameter. There is no LoRA,
  warm-start adapter, replay sampler, or alternate batch scheduler.
- Training starts from `weights/hibiki-pytorch-77f82164@110.safetensors`.
  `--resume-checkpoint` is only for interruption recovery within the same run.
- CUDA uses fp32 master weights, bf16 autocast, fused AdamW, causal SDPA, fixed
  length-sorted 16-frame buckets, and `--max-frames 280`; 80 GB H100s run batch
  8 with two accumulation steps.
- `finetune/h100.sh` pins the pod environment, verifies staged artifacts and
  evaluation audio, selects the 80/94 GB batch recipe, and gates training on a
  save/eval/resume smoke.
- `HIBIKI_RECIPE=grounded-v2` selects isolated CTC word-timed caches, 95/5
  PhoMT/FLEURS sampling, conservative cosine SFT, eligibility-only best saves,
  and a shuffled-source dependency control. PhoMT is pinned, later timbre-matched
  source ranges are labeled without excluding earlier gender-matched pairs, and
  CTC rows below the calibrated 0.5 score are rejected. Legacy behavior remains
  the default. Full weighted runs cover every PhoMT row before repeating data.
- Text prefix PAD weight defaults to 0.5. Content/EOS remain weight 1.0.
- `finetune/common.py` owns cached data, losses, schedules, exact full-model
  checkpoint I/O, greedy generation, and metrics. Complete checkpoint pairs are
  published atomically and pre-rotated to avoid transient disk spikes; loading
  rejects missing and unexpected tensor keys.
- `finetune/cache_phomt_stream.py` builds the published PhoMT cache directly
  from parquet with bounded download prefetch and Hugging Face Xet. MPS runs
  release each CTC batch and can bound concurrent workers by row and audio-sample
  budgets. `finetune/publish_grounded_cache.py` validates and checksum-publishes
  the complete cache under an isolated dataset prefix. `remote_dataset/download_fleurs_vi_en.py` →
  `finetune/build_pairs.py` → `finetune/cache_codes.py` builds FLEURS inputs.
- `finetune/validate.py` is teacher-forced diagnostics only.
  `finetune/eval.py` free-running chrF plus nonempty/EOS/loop/length gates select
  checkpoints; `sacrebleu` is required.
- `finetune/hf_sync.py` maintains two recovery pairs plus the best model under
  `full_run/` in the public `huybik/hibiki-zero-vi-full-sft` model repo;
  `h100.sh` verifies a shared run identity, supervises sync, and protects the
  local resume point before training restarts.

After the `docs/finetune.md` handoff, use `docs/training_plan.md` for the exact
recipe and `docs/validation_plan.md` for qualification thresholds.

## Environment

- Local Python work uses `/opt/homebrew/Caskroom/miniconda/base/bin/python`.
- Inference requires MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and
  sounddevice for microphone mode.
- Training additionally requires a CUDA-compatible PyTorch build, `moshi`
  0.2.13, safetensors, sacrebleu, datasets, soundfile, pyarrow, and
  huggingface-hub.
