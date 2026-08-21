# Hibiki MLX — project context

This repository maintains two paths:

1. q4 MLX inference for Hibiki-Zero on Apple Silicon;
2. direct full-model Vietnamese-to-English voice-preserving simultaneous
   translation SFT on CUDA.

The mobile model track now has the frozen 12-layer contract, strict CUDA AR and
parallel-head distillation, and a BF16 MLX listening path. Teacher inference and
student forward/backward use BF16; optimizer masters, Adam state, and recovery
checkpoints stay FP32. The final parallel export is cast to BF16. Automatic
quality validation, benchmarking, qualification receipts, and student q4 are
deferred; sample translation and manual listening are the current endpoint.

Historical design, experiment, and report documentation remains under `docs/`.
The active training handoff is `docs/finetune.md`.

## Inference

- `main.py` is the file and microphone translation CLI.
- `macos/HibikiTestApp/` is the native SwiftUI file/microphone test app. Its
  local build script produces `build/Hibiki Test.app` and invokes `main.py`.
- `hibiki_mlx/pipeline.py` owns the Mimi-encode → LM → Mimi-decode pipeline.
- `moshi-mlx/` is the minimal vendored MLX runtime. Its required Hibiki deltas
  are GQA (`kv_repeat`), configurable `hidden_scale`, `rope_concat`,
  per-slice depformer output LayerNorm, and the compact one/two-pass
  `parallel_v1` head.
- `3b` resolves to the existing q4 `weights/` directory. Custom directories may
  contain existing q4 weights or staged BF16 weights with
  `weight_dtype=bfloat16`.
- Each codec thread owns a separate `rustymimi.Tokenizer`. Queues carry NumPy
  arrays because lazy MLX graphs cannot cross their creating threads.
- File inference adds an 8-second silence tail and stops on text EOS or after
  12 sustained PAD frames.
- The student listening path is BF16 export → `scripts/convert_mlx_bf16.py` →
  `main.py INPUT --model STAGED_DIR`.
- `scripts/check_swift_compat.py` requires strict q4 group-size-32 reload plus
  valid config, tokenizer, and Mimi sidecars, and deliberately rejects
  `parallel_v1` as unsupported by stock Swift.

## Training

- The only launcher recipe is direct full-data SFT from the upstream
  Hibiki-Zero weight. ASCII-ASR and all pilot, replay, post-source, contrastive,
  anti-repetition, and six-epoch curricula are obsolete and rejected.
- Cached streams are used unchanged: English target Mimi audio, CTC-timed
  English text ending in tokenizer EOS, and Vietnamese source Mimi audio ending
  in explicit codec-card EOS. There is no dataset transform.
- English target audio is teacher-forced and trained with audio CE. Every model
  parameter is trainable.
- The frozen train receipt has 719,120 rows: 683,164 PhoMT and 35,956 FLEURS.
  Raw train and validation are capped at 280 frames. Validation retains 138
  rows, has observed maximum 277, uses batch 4, and never shuffles.
- The run uses physical batch 16, accumulation 1, five epochs / 224,725 steps,
  fixed LR `1e-6`, and fused AdamW with betas `(0.9, 0.95)` and weight decay
  `0.1`. CUDA uses fp32 master weights, bf16 autocast, compile, and 16-frame
  buckets.
- `finetune/h100.sh` pins Torch `2.8.0+cu128` and Moshi `0.2.13`, verifies one
  H100 with at least 90 GiB VRAM and driver 570+, freezes the receipt, and gates
  launch on a longest-row save/resume smoke.
- `finetune/train.py` validates and saves every 9,000 steps plus the final step,
  promoting the lowest teacher-forced validation loss. `finetune/eval.py` is an optional
  correct-source-only free-running check; shuffled-source evaluation is not part
  of this receipt.
- `finetune/hf_sync.py` keeps the newest two complete recovery pairs, the latest
  promoted best model, run metadata, and logs under the public model prefix
  `grounded_v2_full_direct_voice_5epoch`. Loading requires an exact same-run model,
  optimizer, manifest, receipt, run identity, and code commit.
- Published caches live under the dataset prefix `grounded-v2/`. The launcher
  expects `finetune/cache/phomt_grounded_v2`,
  `finetune/cache/train_grounded_v2`, and
  `finetune/cache/validation_grounded_v2`.

Use `docs/finetune.md` for pod setup and recovery, `docs/training_plan.md` for
the frozen recipe, and `docs/validation_plan.md` for monitoring and optional
free-running evaluation.

## Canonical resources

- Caches: https://huggingface.co/datasets/huybik/hibiki-zero-vi-full-sft/tree/main/grounded-v2
- Recovery: https://huggingface.co/huybik/hibiki-zero-vi-full-sft/tree/main/grounded_v2_full_direct_voice_5epoch
- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
- PhoMT speech data: https://huggingface.co/datasets/anquachdev/PhoMT-en-vi-speech

## Environment

- Local Python work uses `/opt/homebrew/Caskroom/miniconda/base/bin/python`.
- The ignored `.env` may contain the HF credential. Never print, log, or commit
  its value.
- Inference requires MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and
  sounddevice for microphone mode.
- Training uses Python 3.10–3.13, Torch `2.8.0+cu128`, Moshi `0.2.13`, and the
  exact dependencies installed by `./finetune/h100.sh setup`.
