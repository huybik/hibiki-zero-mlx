# Vietnamese base-start training recipe

Train the full Hibiki-Zero 3B model from the upstream base weight on the frozen
PhoMT and FLEURS caches. Do not initialize from another finetuned checkpoint.
Use `--resume-checkpoint` only to recover this same run after interruption.

## Preflight

Stage the repository, weights, caches, pair manifests, and FLEURS validation/test
audio on a pod with one visible H100, NVIDIA driver 580 or newer, at least 128 GB
host RAM, and 300 GB disk. CUDA 13.x minor-version compatibility supports the
pinned CUDA 13.2 wheel on the 580 driver series. Then run:

```bash
./finetune/h100.sh setup
./finetune/h100.sh preflight
./finetune/h100.sh smoke
```

The launcher pins Torch 2.13.0/CUDA 13.2 and Moshi 0.2.13, verifies artifact
hashes and evaluation audio, and chooses batch 16 on a 94 GB H100 NVL or batch 8
with two accumulation steps on an 80 GB H100. The smoke must complete finite
training, save, standalone free-running evaluation, exact resume, and a VRAM
headroom check before it writes `finetune/runs/h100_smoke/SMOKE_OK`.

## Run

Create one private, empty, dedicated Storage Bucket and authenticate the pod once:

```bash
./.venv/bin/hf auth login
./.venv/bin/hf buckets create YOUR_USER/hibiki-vi-checkpoints --private
export HIBIKI_HF_BUCKET=YOUR_USER/hibiki-vi-checkpoints
```

```bash
./finetune/h100.sh train
```

Recover an interrupted run in its original directory with:

```bash
./finetune/h100.sh resume finetune/runs/vi_base_full/trainer_stepNNNNNN.pt
```

If the pod volume was lost, list `checkpoints/` in the bucket and copy the
newest matching model/trainer filenames into `finetune/runs/vi_base_full/`
before running the same resume command. Restore the run identity and current
best files too; resume verifies them before it uploads the newest local pair:

```bash
./.venv/bin/hf buckets list "$HIBIKI_HF_BUCKET/checkpoints"
./.venv/bin/hf buckets cp \
  "hf://buckets/$HIBIKI_HF_BUCKET/run.json" \
  finetune/runs/vi_base_full/run_id.json
./.venv/bin/hf buckets cp \
  "hf://buckets/$HIBIKI_HF_BUCKET/checkpoints/model_stepNNNNNN.safetensors" \
  finetune/runs/vi_base_full/model_stepNNNNNN.safetensors
./.venv/bin/hf buckets cp \
  "hf://buckets/$HIBIKI_HF_BUCKET/checkpoints/trainer_stepNNNNNN.pt" \
  finetune/runs/vi_base_full/trainer_stepNNNNNN.pt
./.venv/bin/hf buckets list "$HIBIKI_HF_BUCKET/best"
./.venv/bin/hf buckets cp \
  "hf://buckets/$HIBIKI_HF_BUCKET/best/best_stepBBBBBB.safetensors" \
  finetune/runs/vi_base_full/best_stepBBBBBB.safetensors
./.venv/bin/hf buckets cp \
  "hf://buckets/$HIBIKI_HF_BUCKET/best/best_stepBBBBBB.json" \
  finetune/runs/vi_base_full/best.json
```

`h100.sh train` expands to the recipe below using the hardware-selected batch
size and accumulation factor.

```text
epochs=2, max_frames=280, effective_batch=16
lr=1e-4@0,3e-5@0.5, warmup_steps=500
text_weight=5@0,2@0.6, text_prefix_pad_weight=0.5
val_every=2000, eval_every=9000, save_every=3000, keep_checkpoints=2
```

Checkpoint rotation keeps the newest two complete model/trainer pairs. Before a
new save it removes the oldest pair while retaining one valid recovery point,
so checkpoint storage peaks at two pairs (about 70 GiB) rather than three. Model
and trainer files are published atomically; interrupted writes are ignored and
cleaned at the next save. The setup also disables pip's package cache.

The launcher runs `hf_sync.py` beside training. It uploads 9,000-step evaluation
checkpoints, the best model, and the newest pair when training exits to the
private mutable bucket. The bucket keeps two recovery pairs plus one best model;
its steady checkpoint footprint is about 82 GiB, and old objects are actually
deleted instead of retained in Git history. Hard-link
staging prevents local rotation from invalidating an active upload without
copying checkpoint bytes, though a slow or interrupted upload can temporarily
pin one additional pair on disk. Training waits for a verified final sync before
the launcher exits. If sync fails three consecutive times, the launcher stops
training instead of continuing without disaster recovery.

## Stop conditions

Stop immediately on non-finite loss, checkpoint mismatch, corrupt output, or a
logged LR that differs from `run_config.json`. Teacher-forced loss cannot select
a model. Apply the free-running eligibility gates in
[validation_plan.md](validation_plan.md) and preserve the best eligible
model/trainer pair before rotation. The rolling bucket preserves the raw-chrF
best model but not its matching trainer state.
