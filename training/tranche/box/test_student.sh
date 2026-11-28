#!/bin/bash
# A phone student's test call: the 13 test sets in one eval.py call, seed 43, batch 64, LAAL, as
# both_heads_z's and the heads 6-bit finish's tests ran (heads-on-policy-runs.md, student-palettize.md),
# with the voice scored and no speech written. Run it alone on the GPU; a finished OUT_DIR (its
# metrics.json) is skipped, so a rerun continues.
#   test_student.sh CONFIG MODEL_WEIGHT OUT_DIR
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 CONFIG MODEL_WEIGHT OUT_DIR" >&2; exit 1; }
cd /workspace/hibiki-zero
[[ -f $3/metrics.json ]] && { echo "have $3"; exit 0; }
read -r q per < /sys/fs/cgroup/cpu.max; if [[ $q == max ]]; then N=$(nproc); else N=$(( q / per )); fi
export OMP_NUM_THREADS=$N
E=data/eval
.venv/bin/python -m training.finetune.eval --config-path "$1" --model-weight "$2" --out-dir "$3" \
  --pairs $E/fleurs-vi/test.jsonl $E/fleurs-en/test.jsonl $E/multimed-vi/select.jsonl $E/multimed-vi/final.jsonl \
  $E/giga-heldout/test.jsonl $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl \
  $E/en-vi-val-noisy/cv27.jsonl $E/en-vi-val-noisy/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl \
  $E/en-vi-val-en/libriheavy.jsonl $E/en-vi-val-en-noisy/cv27.jsonl $E/en-vi-val-en-noisy/libriheavy.jsonl \
  --batch-size 64 --seed 43 --laal --no-write-audio
