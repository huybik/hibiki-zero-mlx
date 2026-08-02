#!/bin/bash
# One-off phase-2 speed bench (H100, full-model SFT config): short runs over a
# 20-shard subset of the phomt_stream cache, mean s/step from train_log.jsonl.
# Run dirs are deleted after metric extraction (full saves are ~37 GB each).
set -e
cd "$(dirname "$0")/.."
BENCH_CACHE=finetune/cache/bench
mkdir -p "$BENCH_CACHE"
ls finetune/cache/phomt_stream/shard_*.pt | head -20 | while read -r f; do
  cp -n "$f" "$BENCH_CACHE/"
done
mkdir -p finetune/runs

run() {
  name=$1; min_step=$2; shift 2
  out=finetune/runs/bench_$name
  rm -rf "$out"
  if ! python finetune/train_lora.py --device cuda --dtype float32 --full-finetune \
      --cache-dir "$BENCH_CACHE" --max-frames 280 --save-every 0 \
      "$@" --out-dir "$out" > "$out.log" 2>&1; then
    echo "$name FAILED"; tail -5 "$out.log"; rm -rf "$out"; return 0
  fi
  python - "$out" "$name" "$min_step" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/train_log.jsonl")]
rows = [r for r in rows if r["step"] > int(sys.argv[3])]
sps = sum(r["sec_per_step"] * r["log_steps"] for r in rows) / sum(r["log_steps"] for r in rows)
print(f"RESULT {sys.argv[2]}: {sps:.3f} s/step, last loss {rows[-1]['loss']:.3f}")
PY
  rm -rf "$out"
}

run base_b8_log1 10 --batch-size 8 --log-every 1 --max-steps 40
run b8_log10 10 --batch-size 8 --log-every 10 --max-steps 40
HIBIKI_SDPA_CAUSAL=1 run b8_causal 10 --batch-size 8 --log-every 10 --max-steps 40
HIBIKI_SDPA_CAUSAL=1 HIBIKI_FRAME_BUCKET=1 run b8_causal_bucket1 10 --batch-size 8 --log-every 10 --max-steps 40
HIBIKI_SDPA_CAUSAL=1 run b16_causal 10 --batch-size 16 --log-every 10 --max-steps 40
HIBIKI_SDPA_CAUSAL=1 run b24_causal 10 --batch-size 24 --log-every 10 --max-steps 40
HIBIKI_SDPA_CAUSAL=1 NO_TORCH_COMPILE= run b8_causal_compile 20 --batch-size 8 --log-every 10 --max-steps 60
echo BENCH_DONE
