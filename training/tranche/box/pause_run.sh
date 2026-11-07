#!/usr/bin/env bash
# A benchmark pause: at the next shard boundary stop voice and filters, run the command given
# ("$@", from /workspace/hibiki-zero), then restart filters and voice on main.
set -uo pipefail
L=/workspace/logs
say() { echo "$(TZ=Asia/Bangkok date '+%T') $*"; }
start=$(grep -a -c "still broken" $L/voice.log)
until (( $(grep -a -c "still broken" $L/voice.log) > start )); do sleep 10; done
say "boundary: $(grep -a "still broken" $L/voice.log | tail -1 | cut -c1-200)"
for j in voice filters; do kill -- -"$(cat $L/$j.pid)" 2>/dev/null; done
for j in voice filters; do for _ in $(seq 60); do kill -0 -- -"$(cat $L/$j.pid)" 2>/dev/null || break; sleep 2; done; kill -9 -- -"$(cat $L/$j.pid)" 2>/dev/null; done
echo "$(date -u '+%F %T') stopped for a benchmark pause" | tee -a $L/voice.log >> $L/filters.log
sleep 10; say "paused; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
cd /workspace/hibiki-zero && "$@"
/workspace/restart.sh filters; /workspace/restart.sh voice; say "filters and voice restarted"
