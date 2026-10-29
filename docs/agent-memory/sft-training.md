# Full-model SFT training

Read when touching `training/h100.sh` or `training/finetune/`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Trainer](#trainer) · [Objective](#objective) ·
[Launcher](#launcher) · [Checkpoints and recovery](#checkpoints-and-recovery) ·
[Side tools](#side-tools)

The pod procedure is in [docs/finetune.md](../finetune.md). Caches are covered in
[sft-data-caches](sft-data-caches.md).

## Trainer

`training/finetune/train.py`:

- Full-model SFT from `kyutai/hibiki-zero-3b-pytorch-bf16`; every parameter is trainable.
- Inputs are one or more `--cache-dir` grounded caches plus a `--val-cache-dir`.
- Rows over `--max-frames` (280) are dropped. Training rows are length-sorted, then
  batch-shuffled once with `--seed` (42), and every epoch replays that order. Validation
  rows are length-sorted and never shuffled.
- Defaults are the last production recipe:
  - batch 16, 5 epochs (`--max-steps` overrides);
  - constant LR 1e-6; fused AdamW with betas (0.9, 0.95) and weight decay 0.1; grad clip 1.0;
  - validation (batch 4) and checkpoints every 9,000 steps; `--val-every` must be a multiple
    of `--save-every`, because promotion hard-links the checkpoint saved at that step.
- Precision: fp32 master weights, bf16 autocast, TF32 matmul "high", and a causal-SDPA patch.
  Batches pad to 16-frame buckets (`common.FRAME_BUCKET`), and training requires moshi's
  internal compile, meaning `NO_TORCH_COMPILE` must be unset. `training.finetune` is
  CUDA-only (`train.py` has no device flag); the `training.prep` steps, the cache build
  included ([sft-data-caches](sft-data-caches.md)), run on the Mac.

## Objective

`common.compute_batch_losses`; the derivation is in
[docs/analysis/loss_function.md](../analysis/loss_function.md).

- Loss = 1.0 × audio CE over the 16 English target codebooks (teacher-forced)
  + 1.0 × text CE.
- Text uses `prefix` pad mode: only PADs before the first content token are supervised, at
  weight 0.05, giving (content + 0.05 · pad) / 1.05. PADs between words are unsupervised.

## Launcher

`training/h100.sh` runs from the repo root and invokes modules as
`python -m training.<package>.<module>`.

- `setup` needs Linux, driver 570+, exactly one H100, Python 3.10–3.13 and no existing
  `.venv`. It installs Torch `2.8.0+cu128` (CUDA 12.8), pinned dependencies, and
  `moshi==0.2.13 --no-deps`.
- `preflight` checks:
  - a clean tracked tree;
  - exact Torch and Moshi versions; one H100 with compute capability 9.0;
  - ≥90 GiB VRAM (a standard 80 GB H100 fails), ≥110 GiB host RAM and ≥190 GiB free disk;
  - the SHA-256 of the four upstream files in `weights/`.
- `train RUN_NAME ARGS…` runs preflight and requires an empty `runs/RUN_NAME/`. It needs a
  public `HIBIKI_HF_REPO` whose `RUN_NAME/` prefix is empty. It writes `run_id.json` (commit
  plus random id), uploads it as `RUN_NAME/run.json`, and runs `train.py --out-dir
  runs/RUN_NAME ARGS…` beside the `hf_sync` watcher. Training is killed if sync dies.
- `resume runs/RUN_NAME/trainer_stepNNNNNN.pt ARGS…` requires:
  - a checkpoint directly under `runs/`;
  - a remote no newer than local;
  - a `run_id.json` commit equal to HEAD.
  It syncs once, then resumes with the same ARGS.

## Checkpoints and recovery

- A complete pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt`; two are kept
  locally. `best_step*.safetensors` is a hard link promoted only on a strict improvement in
  teacher-forced validation loss (`best.json`, metric `teacher_forced_loss`).
- `train.py` resumes only from the newest complete pair, and only if `run_config.json`
  matches on `RESUME_KEYS`: data dirs, weights, hyperparameters, cadences, seed, train row
  count and total steps. From the first validation step on, `best.json` is required.
- `hf_sync.py` uploads under `HIBIKI_HF_PREFIX` (the run name):
  - `checkpoints/`: on each poll, the newest complete pair if newer than the remote, keeping
    the newest two;
  - `best/`: the newest promoted model only;
  - `metadata/run_config.json`;
  - logs, on the final sync only.
  It requires a public repo and a remote `run.json` byte-equal to the local `run_id.json`. It
  polls every 60 s and aborts after three failures.
- `hf_sync` never restores anything; restoring is the manual procedure in
  [docs/finetune.md](../finetune.md).

## Side tools

- `eval.py` is the correct-source free-running check on a pair file (`id`, `vi_audio`,
  `text_en`). Despite its name, `run_greedy_eval` samples (text 0.4, audio 0.8).
- `validate.py` reports teacher-forced losses on one cache with BF16 weights and batch 8; it
  applies no frame filter.
- `common.py` is shared by `train.py`, `validate.py` and `eval.py`.
- `train.py` and `eval.py` take a required `--out-dir`.
