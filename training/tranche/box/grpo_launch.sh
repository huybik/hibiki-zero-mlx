#!/usr/bin/env bash
# GRPO on a two-way model, both directions (user, 2026-11-10), from the box's checkout (no commit
# or tracked edit there while it runs: a resume needs the launch commit at HEAD).
#   grpo_launch.sh RUN START_WEIGHT [ARGS...]   start RUN (START_WEIGHT under weights/: every
#                                               resume reloads it)
#   grpo_launch.sh RUN                          relaunch RUN, which resumes it
# ARGS follow the 3B's sizes and win over them (argparse keeps an option's last value): a phone
# student passes its --config-path, --text-weight 50 and its own profile's sizes (grpo.md).
# Stop: kill -TERM "$(cat /workspace/logs/grpo.pid)" ALONE and wait for the final sync; a group
# kill (restart.sh style) kills hf_sync, and the newest pair and the logs never reach the Hub.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 RUN [START_WEIGHT [ARGS...]]" >&2; exit 2; }
run=$1
shift
cd /workspace/hibiki-zero
# grpo.pid is how a run is stopped: never overwrite a live launcher's.
if [[ -f /workspace/logs/grpo.pid ]] && kill -0 "$(cat /workspace/logs/grpo.pid)" 2>/dev/null; then
  echo "a GRPO launcher is running (pid $(cat /workspace/logs/grpo.pid))" >&2
  exit 1
fi
read -r q per < /sys/fs/cgroup/cpu.max; if [[ $q == max ]]; then N=$(nproc); else N=$(( q / per )); fi  # the container's quota
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
T=/workspace/tranche/repo/cache
# Real Vietnamese speech to English (the tranche), English to Vietnamese from the tranche's
# voice-cloned English and from en_speech's real recordings. The 3B's H100 profile's sizes
# (grpo.md); 2,000 updates, judged at 800 (user, 2026-11-13: 800, more while it still improves;
# the rate is constant, so a longer --updates is the same run up to there).
args=()
[[ $# -ge 1 ]] && args=(--cache-dir $T/pho $T/vivoice $T/giga $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga
                         data/cache/en-vi/en-vi-grounded-v2/train
                         --model-weight "$1" --gen-frames 614400 --micro-batch 2 --updates 2000 "${@:2}")
echo "$(date -u '+%F %T') $run ${args[*]}" >> /workspace/logs/grpo.log
setsid ./training/h100.sh grpo "$run" "${args[@]}" >> /workspace/logs/grpo.log 2>&1 < /dev/null &
echo $! > /workspace/logs/grpo.pid
echo "grpo pid $(cat /workspace/logs/grpo.pid)"
