#!/usr/bin/env bash
# end_rental.sh HH:MM HH:MM: before the box expires. From the first time (UTC+7) wait for voice's next
# shard boundary (an open shard's voicing is lost when voice stops), or until the second, then stop
# voice, filters and the ASR server, push every voiced shard and check the Hub holds each one.
set -uo pipefail
L=/workspace/logs
say() { echo "$(TZ=Asia/Bangkok date '+%F %T') $*"; }
at() { TZ=Asia/Bangkok date -d "$1" +%s; }
from=$(at "$1"); until_=$(at "$2")
while (( $(date +%s) < from )); do sleep 30; done
start=$(grep -a -c "still broken" $L/voice.log)
until (( $(grep -a -c "still broken" $L/voice.log) > start )) || (( $(date +%s) >= until_ )); do sleep 10; done
say "stopping after: $(grep -a "still broken" $L/voice.log | tail -1 | cut -c1-120)"
for j in voice filters asr; do kill -- -"$(cat $L/$j.pid)" 2>/dev/null; done
for j in voice filters asr; do
  for _ in $(seq 60); do kill -0 -- -"$(cat $L/$j.pid)" 2>/dev/null || break; sleep 2; done
  kill -9 -- -"$(cat $L/$j.pid)" 2>/dev/null
done
echo "$(date -u '+%F %T') stopped: the rental ends" | tee -a $L/voice.log $L/filters.log >> $L/asr.log
cd /workspace/hibiki-zero && set -a && . /workspace/.env && set +a
/workspace/envs/nanovox/bin/python -m training.tranche.ingest push giga --part en_audio || { say "push FAILED"; exit 1; }
/workspace/envs/nanovox/bin/python - <<'EOF'
from pathlib import Path
from huggingface_hub import HfApi
from training.tranche import ingest
files = HfApi().list_repo_files(ingest.REPO, repo_type="dataset")
hub = {f.rsplit("/", 1)[1] for f in files if f.startswith("giga/en_audio/")}
local = {p.name for p in (ingest.ROOT / "giga" / "en_audio").glob("*.parquet")}
left = sorted(f.rsplit("/", 1)[1] for f in files if f.startswith("giga/segments/") and f.rsplit("/", 1)[1] not in hub)
print(f"giga en_audio on the Hub: {len(hub)} of 240; local not on the Hub: {sorted(local - hub) or 'none'}")
print(f"left to voice ({len(left)}): {' '.join(n.split('.')[0] for n in left)}")
EOF
