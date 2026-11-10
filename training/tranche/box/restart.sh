#!/usr/bin/env bash
# restart.sh asr|mt|voice|filters|cache|sft|envi-cache|pilot: stop the job by its PID file (its whole process group), start it again.
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
  cache) export HF_HUB_DISABLE_XET=1  # plain HTTP at 143 MB/s; Xet throttles to ~4 MB/s
         cmd=(.venv/bin/python -m training.tranche.cache build pho vivoice giga) ;;
  # Round 4 SFT again (2026-11-10, full_epoch4e): every tranche cache plus 500 seeded shards (~2,480 h, 29% of the hours;
  # full_epoch4b had 150, 11%) of the noisy en_speech cache, from round 3's final model, in round 3's 4,608-frame batches:
  # 131,659 steps. A pair every 4,000 steps: each save idles the GPU ~6 s (sft-speed.md). Validation batches of 64 rows fit
  # an 80 GB card (round 4b's 128 ran on 141 GB). Rerun with the same args resumes it (h100.sh run). Round 3
  # (tranche_epoch1b) ran from run 2's final model on the first tranche, --seed 45 --val-every 8000.
  sft) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo E=data/eval
       cmd=(./training/h100.sh run full_epoch4e
            --cache-dir $T/cache/pho $T/cache/vivoice $T/cache/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation
            --model-weight runs/tranche_epoch1b/model_step026176.safetensors --batch-frames 4608 --eval-batch-size 64 --keep-checkpoints 1
            --lr 3e-6 --seed 46 --save-every 4000 --val-every 28000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/multimed-vi/select.jsonl $E/en-vi-val/cv27.jsonl
            $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-noisy/cv27.jsonl $E/en-vi-val-noisy/libriheavy.jsonl) ;;
  # The two-way pilot's English-to-Vietnamese inputs (two-way.md#pilot): the English evaluation sets, the en_speech
  # validation shards and the 40 train shards of box/en_speech_pilot.txt (of sft's 500, random.Random(0)) by cache_codes
  # from the dataset's copy in the tranche repo (en_speech/, fetched to /workspace/en_speech_src), and the first 6 giga,
  # 14 pho and 20 vivoice shards by the tranche builder, which pushes them to its cache repo's en-vi/.
  envi-cache) export HF_HUB_DISABLE_XET=1
       cmd=(bash -c 'set -e; D=/workspace/en_vi_speech/data S=/workspace/en_speech_src/en_speech; mkdir -p $D
            ln -sfn $S/train $D/train; ln -sfn $S/validation $D/validation
            .venv/bin/python -m training.prep.eval_sets --sets fleurs-en en-vi-val-en
            .venv/bin/python -m training.prep.cache_codes --device cuda --direction en-vi --dataset-dir ${D%/data} \
              --out-dir data/cache/en-vi/en-vi-grounded-v2 --shards "validation/*.parquet" $(sed "s|.*|train/&.parquet|" training/tranche/box/en_speech_pilot.txt)
            for limit in giga:6 pho:14 vivoice:20; do
              .venv/bin/python -m training.tranche.cache build ${limit%:*} --direction en-vi --limit ${limit#*:}; done') ;;
  # The pilot (two-way.md#pilot): 10k steps on both directions of the same rows, envi-cache's English-to-Vietnamese
  # shards and the Vietnamese-to-English shards of the same names (linked into data/cache/pilot-vi-en; en_speech's from
  # the noisy cache, as sft trains it), from the VI->EN start moved onto the Vietnamese swap (init_weights writes
  # weights/twoway_start_vi6k.safetensors). Validation every 2,500 steps, both directions teacher-forced and free-running.
  pilot) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2 P=data/cache/pilot-vi-en
       for pair in giga:$T/giga pho:$T/pho vivoice:$T/vivoice en_speech:data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train; do
         name=${pair%%:*}; from=$(realpath ${pair#*:}); mkdir -p $P/$name
         if [[ $name == en_speech ]]; then built=$V/train; else built=$T/en-vi/$name; fi
         compgen -G "$built/shard_*.pt" > /dev/null || { echo "no shards in $built: run envi-cache" >&2; exit 1; }
         for f in $built/shard_*.pt; do
           [[ -f $from/$(basename $f) ]] || { echo "$from has no $(basename $f)" >&2; exit 1; }
           ln -sfn $from/$(basename $f) $P/$name/
         done
       done
       cmd=(./training/h100.sh run twoway_pilot
            --cache-dir $T/en-vi/giga $T/en-vi/pho $T/en-vi/vivoice $V/train $P/giga $P/pho $P/vivoice $P/en_speech
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --model-weight weights/twoway_start_vi6k.safetensors --tokenizer weights/tokenizer_spm_48k_multi6_2_vi6k.model
            --max-steps 10000 --batch-frames 4608 --eval-batch-size 64 --keep-checkpoints 1
            --lr 3e-6 --seed 47 --save-every 2500 --val-every 2500
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl) ;;
  *) echo "usage: $0 asr|mt|voice|filters|cache|sft|envi-cache|pilot" >&2; exit 1 ;;
esac
echo "$(date -u "+%F %T") restart $job" >> $L/$job.log
setsid "${cmd[@]}" >> $L/$job.log 2>&1 < /dev/null &
echo $! > $L/$job.pid
echo "$job pid $(cat $L/$job.pid)"
