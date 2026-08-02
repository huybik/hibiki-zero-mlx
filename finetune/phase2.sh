#!/bin/bash
# Phase-2 full-model SFT launcher (H100 pod). Usage: phase2.sh restore|train
# restore: pull full Mimi cache (8 chunks), phase-1 FLEURS caches+pairs tarball,
#          warm-start checkpoint, FLEURS validation wavs (greedy val128 eval).
# train:   2-epoch warm-start run at the benched best config (batch 16 +
#          causal-SDPA + torch.compile), hf_sync alongside.
set -e
cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
set -a; source .env; set +a
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120

case "$1" in
restore)
  mkdir -p finetune/cache/phomt_stream
  if [ "$(ls finetune/cache/phomt_stream | wc -l)" -lt 1377 ]; then
    hf download huybik/hibiki-zero-vi-full-sft --repo-type dataset \
      --include 'cache_chunk_*.tar.zst' --local-dir /workspace/cache_dl --max-workers 16
    for f in /workspace/cache_dl/cache_chunk_*.tar.zst; do
      tar --zstd -xf "$f" -C finetune/cache/phomt_stream
    done
  fi
  if [ ! -d finetune/cache/train ]; then
    hf download huybik/hibiki-zero-vi-full-sft fleurs_cache.tar.zst --repo-type dataset \
      --local-dir /workspace/cache_dl
    tar --zstd -xf /workspace/cache_dl/fleurs_cache.tar.zst -C finetune
  fi
  if [ ! -f finetune/runs/init/model_step055284.safetensors ]; then
    hf download huybik/hibiki-zero-vi-full-sft model_step055284.safetensors \
      --local-dir finetune/runs/init
  fi
  if [ ! -d remote_dataset/fleurs_vi_en/validation ]; then
    python remote_dataset/download_fleurs_vi_en.py --split validation
  fi
  echo RESTORE_DONE
  ;;
train)
  # causal-SDPA + torch.compile are default-on for CUDA in finetune/common.py
  python finetune/hf_sync.py finetune/runs/vi_full_p2 huybik/hibiki-zero-vi-full-sft &
  SYNC_PID=$!
  python finetune/train_lora.py \
    --device cuda --dtype float32 --full-finetune \
    --cache-dir finetune/cache/phomt_stream finetune/cache/train \
    --val-cache-dir finetune/cache/validation \
    --init-adapter finetune/runs/init/model_step055284.safetensors \
    --batch-size 16 --max-frames 280 --epochs 2 \
    --lr-schedule "5e-5@0,2e-5@0.5,1e-5@0.8" --warmup-steps 500 \
    --text-weight-schedule "3@0,2@0.5" \
    --val-every 2000 --eval-every 9000 --save-every 5000 --keep-checkpoints 2 \
    --log-every 10 \
    --out-dir finetune/runs/vi_full_p2
  kill $SYNC_PID 2>/dev/null || true
  echo TRAIN_DONE
  ;;
*)
  echo "usage: phase2.sh restore|train" >&2; exit 1
  ;;
esac
