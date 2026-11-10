#!/bin/bash
# The rounds' test sets (plan "Order", sft-evaluation.md): one eval.py call a model over eight
# sets, batch 64, --seed 43, text only; ~11 min a model on the H200. Scores vary run to run, so
# compare rounds evaluated in one session. Run it alone on the GPU, from anywhere:
#   /workspace/test_evals.sh MODEL_WEIGHT OUT_DIR    (OUT_DIR: e.g. runs/<run>/eval/cmp_step<N>)
# Writes predictions.csv and metrics.json to OUT_DIR (local only; record the chrF in the plan).
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 MODEL_WEIGHT OUT_DIR" >&2; exit 1; }
cd /workspace/hibiki-zero
E=data/eval
.venv/bin/python -m training.finetune.eval --model-weight "$1" --out-dir "$2" \
  --pairs $E/fleurs-vi/test.jsonl $E/multimed-vi/select.jsonl $E/multimed-vi/final.jsonl \
  $E/giga-heldout/test.jsonl $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl \
  $E/en-vi-val-noisy/cv27.jsonl $E/en-vi-val-noisy/libriheavy.jsonl \
  --batch-size 64 --seed 43 --text-only
