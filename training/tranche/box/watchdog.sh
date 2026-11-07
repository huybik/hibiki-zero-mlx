#!/usr/bin/env bash
# Quiet watchdog for the all-voice-first plan (2026-11-07; docs/agent-memory/tranche-box.md): checks every 10 min and exits at the first
# failure or event worth a decision, printing it, so the session hears of it early without progress
# chatter. Generation must stay up: asr, filters, voice. Alerts: a job down, a new "voice run
# failed", voice or filters silent 30 min, the ASR health failing twice (20 min), and voice finding
# 0 shards to voice (stalled on asr_vi, or every shard voiced: the cache build next).
L=/workspace/logs
say() { echo "$(TZ=Asia/Bangkok date '+%F %T UTC+7') $*"; exit 1; }
up() { kill -0 "$(cat $L/$1.pid 2>/dev/null)" 2>/dev/null; }
silent() { (( $(date +%s) - $(stat -c %Y $L/$1.log) > ${2:-1800} )); }
fails0=$(grep -a -c "voice run failed" $L/voice.log); asr_bad=0
while :; do
  sleep 600
  for j in asr filters voice; do up $j || say "$j is down"; done
  (( $(grep -a -c "voice run failed" $L/voice.log) > fails0 )) && say "voice: $(grep -a 'voice run failed' $L/voice.log | tail -1)"
  for j in filters voice; do silent $j && say "$j.log silent 30 min"; done
  last=$(grep -a "shards to voice" $L/voice.log | tail -1)
  [[ $last == *" 0 shards to voice"* ]] && say "voice: $last"
  if curl -sf -m 20 localhost:8100/health >/dev/null; then asr_bad=0; else asr_bad=$((asr_bad + 1)); fi
  (( asr_bad >= 2 )) && say "ASR /health failing"
done
