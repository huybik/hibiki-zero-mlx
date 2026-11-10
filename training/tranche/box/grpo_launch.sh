#!/usr/bin/env bash
# GRPO after round 4, run grpo_r4 on the tranche caches, from the box's checkout (no commit or
# tracked edit there while it runs: a resume needs the launch commit at HEAD).
#   grpo_launch.sh START_WEIGHT   start it (START_WEIGHT under weights/: every resume reloads it)
#   grpo_launch.sh                relaunch, which resumes it
# Stop: kill -TERM "$(cat /workspace/logs/grpo.pid)" ALONE and wait for the final sync; a group
# kill (restart.sh style) kills hf_sync, and the newest pair and the logs never reach the Hub.
set -euo pipefail
cd /workspace/hibiki-zero
read -r q per < /sys/fs/cgroup/cpu.max; if [[ $q == max ]]; then N=$(nproc); else N=$(( q / per )); fi  # the container's quota
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
T=/workspace/tranche/repo/cache
args=()
[[ $# -ge 1 ]] && args=(--cache-dir $T/pho $T/vivoice $T/giga --model-weight "$1" --gen-batch 512 --micro-batch 4 --updates 800)  # the H200 profile, grpo.md
echo "$(date -u '+%F %T') grpo_r4 ${args[*]}" >> /workspace/logs/grpo.log
setsid ./training/h100.sh grpo grpo_r4 "${args[@]}" >> /workspace/logs/grpo.log 2>&1 < /dev/null &
echo $! > /workspace/logs/grpo.pid
echo "grpo pid $(cat /workspace/logs/grpo.pid)"
