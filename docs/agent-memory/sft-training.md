# Full-model SFT training

Read when touching `finetune/train.py`, `finetune/common.py`, `finetune/h100.sh`,
`finetune/hf_sync.py`, `finetune/freeze_full_data_receipt.py`, `finetune/validate.py` or
`finetune/eval.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Recipe](#recipe) · [Objective](#objective) ·
[Launcher](#launcher) · [Checkpoints and recovery](#checkpoints-and-recovery) ·
[Side tools](#side-tools)

The pod procedure is in [docs/finetune.md](../finetune.md), the recipe rationale in
[docs/training_plan.md](../training_plan.md) and monitoring in
[docs/validation_plan.md](../validation_plan.md). Data and caches are covered in
[sft-data-caches](sft-data-caches.md).

## Recipe

- Direct full-model Vietnamese→English voice-preserving SFT, starting from
  `kyutai/hibiki-zero-3b-pytorch-bf16`. Every parameter is trainable.
- The constants live in `finetune/freeze_full_data_receipt.py`, and `train.py` `validate_args`
  rejects any deviation:
  - 719,120 rows; batch 16; accumulation 1; 280 frames;
  - 5 epochs × 44,945 steps = 224,725 steps;
  - constant LR 1e-6; fused AdamW with betas (0.9, 0.95) and weight decay 0.1;
  - seed 42; validation and save cadence of 9,000 steps;
  - validation set: 138 rows, batch 4, observed maximum 277 frames;
  - no gradient checkpointing.
- `--max-steps` is allowed only for the ≤11-step smoke. `grad_clip` 1.0 comes from `h100.sh`
  and is checked on resume.
- Every epoch replays the same frozen batch order; there is no reshuffle.
- Precision: fp32 master weights, bf16 autocast, TF32 matmul "high", and a causal-SDPA patch
  on CUDA. Training requires 16-frame buckets (`HIBIKI_FRAME_BUCKET=16`; the library default is
  32) and moshi's internal compile, meaning `NO_TORCH_COMPILE` must be unset.
- `--cache-dir` order is PhoMT, then FLEURS, because manifest keys are `(cache_index, id)`.

## Objective

`common.compute_batch_losses`; the derivation is in
[docs/analysis/loss_function.md](../analysis/loss_function.md).

- Loss = 1.0 × audio CE over the 16 English target codebooks (teacher-forced)
  + 1.0 × text CE.
- Text uses `prefix` pad mode: only PADs before the first content token are supervised, at
  weight 0.05, giving (content + 0.05 · pad) / 1.05. PADs between words are unsupervised.

## Launcher

`finetune/h100.sh`; see [docs/finetune.md](../finetune.md) for the order.

- `HIBIKI_RECIPE` must be `grounded-v2`. Ten obsolete curriculum flags, plus
  `HIBIKI_MAX_SAMPLES`, `HIBIKI_MAX_STEPS`, `HIBIKI_CACHE_SAMPLE_SHARDS` and
  `HIBIKI_HF_PREFIX_OVERRIDE`, are rejected.
- `setup` needs Linux, driver 570+, exactly one H100, Python 3.10–3.13 and no existing
  `.venv`. It installs Torch `2.8.0+cu128` (CUDA 12.8), pinned dependencies, and
  `moshi==0.2.13 --no-deps`.
- `cache-grounded` builds the caches ([sft-data-caches](sft-data-caches.md)).
- `preflight` checks the following, then freezes the receipt:
  - a clean tracked tree;
  - exact Torch and Moshi versions; one H100 with compute capability 9.0;
  - ≥90 GiB VRAM (a standard 80 GB H100 fails), ≥110 GiB host RAM and ≥190 GiB free disk;
  - four weight hashes; cache shard counts 1377 / 46 / 5.
- `smoke` trains 10 longest-first steps, resumes to step 11, checks outputs, promotion and
  ≥2 GiB VRAM headroom, and writes `SMOKE_OK` with the commit.
- `train` requires `SMOKE_OK` = HEAD, an empty run directory, and a public `HIBIKI_HF_REPO`
  with an empty prefix. It writes `run_id.json` and runs the `hf_sync` watcher; training is
  killed if sync dies.
- `resume <trainer_step.pt>` requires the current smoke, a checkpoint inside the run directory,
  a remote no newer than local, and a `run_id.json` commit equal to HEAD.

## Checkpoints and recovery

- A complete pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt`; two are kept
  locally. `best_step*.safetensors` is a hard link promoted only on a strict improvement in
  teacher-forced validation loss.
- `train.py` resumes only from the newest complete pair, and only if the receipt bytes,
  manifest bytes, `run_config`, total steps and artifact hashes match. `best.json` is required
  from step 9,000 on. Run identity and commit are checked by `h100.sh`, not `train.py`.
- `hf_sync.py` uploads under `HIBIKI_HF_PREFIX` (default `full_run`; `h100.sh` sets
  `grounded_v2_full_direct_voice_5epoch`):
  - `checkpoints/`: the newest two pairs, at multiples of 9,000 plus the final step;
  - `best/`: the newest promoted model only;
  - `metadata/`;
  - logs, on the final sync only.
  It requires a public repo and a remote `run.json` byte-equal to the local `run_id.json`. It
  polls every 60 s and aborts after three failures.
- `hf_sync` never restores anything; restoring is the manual procedure in
  [docs/finetune.md](../finetune.md).

## Side tools

- `eval.py` is the optional correct-source free-running check. Despite its name,
  `run_greedy_eval` samples (text 0.4, audio 0.8).
- `validate.py` does not reproduce in-training validation: it defaults to the legacy
  `cache/validation`, applies no 280-frame filter, and uses BF16 weights, batch 8 and MPS.
- `common.py` is shared by `train.py`, `validate.py`, `eval.py`, `freeze_full_data_receipt.py`
  and `grpo_rollout.py`; `student/` does not import it.
- Defaults that can't work for production: `utils.DEFAULT_RUN_DIR` (`runs/vi_base_full`), and
  `train.py`'s single default `--cache-dir`, which fails the two-cache check. `h100.sh` passes
  the real values.
