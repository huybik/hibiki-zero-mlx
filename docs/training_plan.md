# Vietnamese base-start training recipe

Train the full Hibiki-Zero 3B model from the upstream base weight on the frozen
PhoMT and FLEURS caches. Do not initialize from another finetuned checkpoint.
Use `--resume-checkpoint` only to recover this same run after interruption.

## Preflight

Before launch, verify the base-model, cache, and pair-manifest hashes, then run a
10-step smoke with the final arguments. Confirm finite loss, the requested LR,
an exact checkpoint reload, and free-running validation artifacts.

## Run

```bash
python finetune/train.py \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --cache-dir finetune/cache/phomt_stream finetune/cache/train \
  --val-cache-dir finetune/cache/validation \
  --batch-size 16 --max-frames 280 --sort-by-length \
  --epochs 2 \
  --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
  --text-weight-schedule "5@0,2@0.6" \
  --text-prefix-pad-weight 0.5 \
  --seed 42 \
  --val-every 2000 --val-batch-size 8 \
  --eval-every 9000 \
  --eval-pairs finetune/pairs/val128.jsonl \
  --eval-limit 128 --eval-batch-size 8 --eval-text-temp 0 \
  --save-every 3000 --keep-checkpoints 3 --log-every 10 \
  --out-dir finetune/runs/vi_base_full
```

Optionally run `python finetune/hf_sync.py <run-dir> <model-repo>` alongside
training. It uploads complete checkpoint pairs without deleting Hub history.

## Stop conditions

Stop immediately on non-finite loss, checkpoint mismatch, corrupt output, or a
logged LR that differs from `run_config.json`. Teacher-forced loss cannot select
a model. Apply the free-running eligibility gates in
[validation_plan.md](validation_plan.md) and preserve the best eligible
model/trainer pair before rotation.
