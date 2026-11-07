#!/usr/bin/env bash
# restart.sh asr|mt|voice|filters|cache|sft: stop the job by its PID file (its whole process group), start it again.
# Logs: /workspace/logs/<job>.log, PID files /workspace/logs/<job>.pid. Never pkill -f: the jobs spawn
# workers that look like vLLM processes.
set -euo pipefail
job=$1; L=/workspace/logs; mkdir -p $L
cd /workspace/hibiki-zero
set -a; . /workspace/.env; set +a  # HF_TOKEN_AN: the stages push to the anquachdev repo
if [[ -f $L/$job.pid ]] && kill -0 "$(cat $L/$job.pid)" 2>/dev/null; then
  kill -- -"$(cat $L/$job.pid)" || true
  for _ in $(seq 60); do kill -0 "$(cat $L/$job.pid)" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$(cat $L/$job.pid)" 2>/dev/null || true
fi
# The container's CPU quota (cgroup cpu.max), not nproc (the host's cores): threads and pools follow it.
read -r q per < /sys/fs/cgroup/cpu.max; if [[ $q == max ]]; then N=$(nproc); else N=$(( q / per )); fi
ASR_SNAP=/workspace/.hf_home/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
case $job in
  # Capped: uncapped, its 8 API processes held 508 threads each, most of the container's 6,656.
  asr) export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
       cmd=(/workspace/envs/asr/bin/qwen-asr-serve $ASR_SNAP --served-model-name asr --gpu-memory-utilization 0.2
            --port 8100 --max-num-seqs 256 --api-server-count $(( N / 3 > 0 ? N / 3 : 1 ))) ;;
  # The VI->EN server translate.py calls; it needs the GPU memory voice holds, so the two take turns.
  mt) read -ra cmd <<< "$(/workspace/envs/translate/bin/python -m training.tranche.translate serve)" ;;
  # The scale-up: each reruns 30 s after a run that succeeds, as the ingest and translation bring
  # shards (a run takes the shards ready at its start); the first error stops the loop.
  # Voice reruns after a failure too (an ASR request timing out at 60 s ends a run about once a
  # day: 2026-11-06, 2026-11-07 17:30 UTC+7), but stops after 3 in a row.
  voice) cmd=(bash -c 'f=0; while (( f < 3 )); do if /workspace/envs/nanovox/bin/python -m training.tranche.voice run giga; then f=0; else f=$((f+1)); echo "$(date -u "+%F %T") voice run failed ($f in a row)"; fi; sleep 30; done') ;;
  filters) cmd=(bash -c 'while /workspace/envs/nanovox/bin/python -m training.tranche.filters asr giga; do sleep 120; done') ;;
  cache) cmd=(.venv/bin/python -m training.tranche.cache build pho vivoice giga) ;;
  # Round 4 SFT (2026-11-08): every tranche cache plus 150 seeded shards (~720 h) of the noisy en_speech
  # cache, from round 3's final model. Rerun with the same args resumes it (h100.sh run). Round 3
  # (tranche_epoch1b) ran from run 2's final model on the first tranche, --seed 45 --val-every 8000.
  sft) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo E=data/eval
       cmd=(./training/h100.sh run full_epoch4
            --cache-dir $T/cache/pho $T/cache/vivoice $T/cache/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation
            --model-weight runs/tranche_epoch1b/model_step026176.safetensors
            --lr 3e-6 --seed 46 --val-every 15000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/multimed-vi/select.jsonl $E/en-vi-val/cv27.jsonl
            $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-noisy/cv27.jsonl $E/en-vi-val-noisy/libriheavy.jsonl) ;;
  *) echo "usage: $0 asr|mt|voice|filters|cache|sft" >&2; exit 1 ;;
esac
echo "$(date -u "+%F %T") restart $job" >> $L/$job.log
setsid "${cmd[@]}" >> $L/$job.log 2>&1 < /dev/null &
echo $! > $L/$job.pid
echo "$job pid $(cat $L/$job.pid)"
