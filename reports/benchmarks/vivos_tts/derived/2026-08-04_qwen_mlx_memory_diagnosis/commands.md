# Memory diagnosis commands

All commands in the first block were read-only and executed on 2026-08-04 ICT. They did not signal either live process.

```bash
date '+%Y-%m-%dT%H:%M:%S%z %Z'
/opt/homebrew/bin/tmux list-sessions
/opt/homebrew/bin/tmux list-panes -a -F '#{session_name}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead}\t#{pane_start_command}'
ps -p 72367 -o pid=,ppid=,state=,etime=,%cpu=,%mem=,rss=,vsz=,command=
footprint 72367
vmmap -summary 72367
lsof -p 72367
sysctl vm.swapusage
memory_pressure
vm_stat
tail -n 30 /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/generation_attempt0_20260804.log
find /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/attempt0_t08/groups -name group.json -type f
find /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/attempt0_t08/groups -name '*.wav' -type f -size 0
rg -in 'error|traceback|exception|abort|killed|oom' /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/generation_attempt0_20260804.log
```

The exploration repeated the process/footprint/system sample after 15 seconds while observing group progress. The all-group distribution was derived by reading the first 581 immutable `group.json` files and code-array headers with conda base Python; generation-step counts come from `lane_accounting.active_widths`. No model was imported.

## Proposed repair operations — not executed

These are a procedural record, not a shell script to run blindly. PIDs and hashes must be re-resolved at execution time.

```bash
# 1. Stop and archive the postprocess supervisor first.
/opt/homebrew/bin/tmux kill-session -t hibiki_vivos_qwen_v6_postprocess_r1_20260804

# 2. Observe the next committed [n/1391] line, then interrupt the generator.
tail -F /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/generation_attempt0_20260804.log
/opt/homebrew/bin/tmux send-keys -t hibiki_vivos_qwen_v6_attempt0_20260804 C-c

# 3. Validate the completed prefix and inspect any hidden temporary group before removal.
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/validate_vivos_qwen_production_v6.py production /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan.json --attempt 0
find /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/attempt0_t08/groups -mindepth 1 -maxdepth 1 -type d -name '.*' -print

# 4. After patch, new immutable supersession attestation, and retest, resume the same scope.
/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python training-data/run_vivos_qwen_production_v6.py run /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full/production_plan.json --round 0
```

The original launch and supervisor commands remain authoritative in their immutable external records. No command in the proposed block was run during this diagnosis.
