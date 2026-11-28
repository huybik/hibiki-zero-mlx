#!/usr/bin/env bash
# restart.sh asr|voice|voice-en-speech|cache|student-slices-5090|student-slices-decay-5090|student-slices-refit-5090|student-slices-pal6-5090|student-slices-wide-5090|student-slices-wide-decay-5090|student-slices-wide-warm-5090|student-slices-wide-warm-decay-5090: stop the job by its PID file (its whole process group), start it again.
# Logs: /workspace/logs/<job>.log, PID files /workspace/logs/<job>.pid. Never pkill -f: the jobs spawn
# workers that look like vLLM processes.
set -euo pipefail
job=$1; L=/workspace/logs; mkdir -p $L
cd /workspace/hibiki-zero
set -a; . /workspace/.env; set +a  # HF_TOKEN_AN: the stages push to the anquachdev repo
# Hub progress bars (the stages' threaded pushes and downloads, hf_sync) would glue onto the jobs' log lines.
export HF_HUB_DISABLE_PROGRESS_BARS=1
# The whole group, its leader gone or not: a run killed out of memory leaves its spawned workers alive in it.
if [[ -f $L/$job.pid ]] && kill -0 -- -"$(cat $L/$job.pid)" 2>/dev/null; then
  kill -- -"$(cat $L/$job.pid)" || true
  for _ in $(seq 60); do kill -0 -- -"$(cat $L/$job.pid)" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$(cat $L/$job.pid)" 2>/dev/null || true
fi
# The container's CPU quota (cgroup cpu.max), not nproc (the host's cores): threads and pools follow it.
read -r q per < /sys/fs/cgroup/cpu.max; if [[ $q == max ]]; then N=$(nproc); else N=$(( q / per )); fi
ASR_SNAP=/workspace/.hf_home/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
case $job in
  # One thread each: uncapped, its 8 API processes held 508 threads each, most of the container's 6,656;
  # at 4 the server spent 2.4× the CPU of 1 for the same rows/s (170 against 71 CPU-s per 3,000 YODAS rows, 5090).
  asr) export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
       # A check is under ~1.2k tokens (a clip of at most 22.4 s, 512 tokens out). At 0.2 of the 5090's 32 GB the KV
       # cache (0.91 GiB) held less than one sequence of the model's own maximum length, and the server did not start.
       # No processor cache: every clip is new, and vLLM's 4 GiB a process (7 of them) only held RAM (31.6 GB in all,
       # 2026-11-26). FP8 (vLLM's online quantization): 1.39x bf16's rows/s, voicing's checks and YODAS's gate as
       # bf16's but for sung Vietnamese it tags Chinese (tranche-tts-scaleup.md#in-production, 2026-11-28).
       cmd=(/workspace/envs/asr/bin/qwen-asr-serve $ASR_SNAP --served-model-name asr --gpu-memory-utilization 0.25
            --max-model-len 4096 --mm-processor-cache-gb 0 --port 8100 --max-num-seqs 256 --quantization fp8
            --api-server-count $(( N / 3 > 0 ? N / 3 : 1 ))) ;;
  # The scale-up: each reruns 30 s after a run that succeeds, as the ingest and translation bring
  # shards (a run takes the shards ready at its start); the first error stops the loop.
  # Voice reruns after a failure too (an ASR request timing out at 60 s ends a run about once a
  # day: 2026-11-06, 2026-11-07 17:30 UTC+7), but stops after 3 in a row.
  voice) cmd=(bash -c 'f=0; while (( f < 3 )); do if /workspace/envs/vieneu/bin/python -m training.tranche.voice run yodas; then f=0; else f=$((f+1)); echo "$(date -u "+%F %T") voice run failed ($f in a row)"; fi; sleep 30; done') ;;
  # The EN-VI dataset's Vietnamese heard, its broken rows voiced again (voice.py run en_speech), to the end: its parts go
  # to en_speech/vi_check/ of the tranche repo as it goes, and a rerun continues after a failure, but after 3 in a row
  # (a run that lasted 10 min counts as progress). Plain HTTP: each of its ~447 GB of shards downloads on the loader's
  # path, ~140 MB/s where Xet throttles to ~5.
  voice-en-speech) export HF_HUB_DISABLE_XET=1
       cmd=(bash -c 'f=0; while true; do s=$SECONDS; /workspace/envs/vieneu/bin/python -m training.tranche.voice run en_speech && exit 0; (( SECONDS - s < 600 )) || f=0; f=$((f+1)); echo "$(date -u "+%F %T") voice run failed ($f in a row)"; (( f < 3 )) || exit 1; sleep 30; done') ;;
  # Single-pass dual-direction CUDA build (vi-en + en-vi): reads parquets & decodes clips ONCE,
  # runs Triton fused SEANet Mimi + FP16 wav2vec2 CTC + C Viterbi + /dev/shm slabs. Plain HTTP
  # for the Hub: 143 MB/s, where Xet throttles to ~4 MB/s.
  cache) export HF_HUB_DISABLE_XET=1
         cmd=(/workspace/envs/cache/bin/python -m training.tranche.cache build pho vivoice giga yodas
              --direction both --drop-inputs) ;;
  # Stage 1 of the depformer-slices phone student (training/student/slices.py,
  # docs/agent-memory/student-slices.md): warm up the 9 random 512x2 slices on the frozen
  # both_heads_z_best1200 trunk with chunked KL + cached frozen trunk FP8 + pre-quantized FP8
  # teacher (~0.43 s/update, 16.1 GiB peak on RTX 5090) and rebalanced AR acoustic loss (cb 1..7).
  student-slices-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       name=student_slices14 S=weights/$name
       [[ -f $S/config.json ]] || .venv/bin/python -m training.student.slices \
         --weight weights/both_heads_z_best1200.safetensors --config-path weights/student_heads14_cb16/config.json \
         --dim 512 --num-heads 8 --layers 2 --parallel 8 --seed 47 --out-dir $S
       cmd=(./training/h100.sh run ${name}_stage1
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $S/model.safetensors
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --freeze-trunk --fp8 --fp8-teacher all --batch-frames 12288 --park-states
            --lr 2e-4 --first-codebook-weight 10 --acoustic-ar-weight 3
            --schedule constant --warmup-fraction 0.05 --max-steps 16000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 47 --save-every 4000 --val-every 8000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # Stage 1's decay: a --schedule decay branch from Stage 1's update-16,000 pair (hard-linked under
  # its own names into weights/student_slices14_stage1_step016000/), the trunk still frozen, the
  # rate 2e-4 to 0 as 1 - sqrt over 8,000 updates (~0.43 s/update). Unfreezing the trunk on this
  # data (the joint stage, the trunk at 1e-5) lost chrF on every set by 8,000:
  # docs/agent-memory/student-slices.md.
  student-slices-decay-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       name=student_slices14 S=weights/$name P=weights/student_slices14_stage1_step016000
       cmd=(./training/h100.sh run ${name}_stage1_decay
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $P/model_step016000.safetensors
            --optimizer-state $P/trainer_step016000.pt
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --freeze-trunk --fp8 --fp8-teacher all --batch-frames 12288 --park-states
            --lr 2e-4 --first-codebook-weight 10 --acoustic-ar-weight 3
            --schedule decay --warmup-fraction 0 --max-steps 8000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 49 --save-every 4000 --val-every 4000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The slices' refit on the on-policy trunk (slices-on-policy.md#refit): on-policy updates raised the
  # text but lowered the speech's clarity (HNR), so the slices are distilled again from the 3B on the
  # SFT data, the trunk frozen, from the on-policy run's best (runs/both_slices_z/best.json), with a
  # fresh optimizer at 5e-5, cosine over 2,000 updates (~0.43 s/update).
  student-slices-refit-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       u=$(.venv/bin/python -c 'import json; print(json.load(open("runs/both_slices_z/best.json"))["step"])')
       cmd=(./training/h100.sh run slices_z${u}_refit
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path weights/student_slices14/config.json
            --model-weight runs/both_slices_z/$(printf best_step%06d.safetensors "$u")
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --freeze-trunk --fp8 --fp8-teacher all --batch-frames 12288 --park-states
            --lr 5e-5 --first-codebook-weight 10 --acoustic-ar-weight 3
            --schedule cosine --warmup-fraction 0.05 --max-steps 2000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 51 --save-every 1000 --val-every 1000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The slices student's 6-bit finish (slices-on-policy.md#phone-finish, the heads student's recipe in
  # student-palettize.md): the refit's final model on the on-policy trunk (hard-linked to
  # weights/both_slices_z_refit.safetensors) distilled from its own bf16 weights with the Core ML export's 6-bit tables in the step, at 5e-7
  # (a usual rate walks the weights from the teacher), 3,072-frame batches, --park-states for the
  # validation on 32 GB.
  student-slices-pal6-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       W=weights/both_slices_z_refit.safetensors C=weights/student_slices14/config.json
       cmd=(./training/h100.sh run pal6_slices_z
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $C --model-weight $W --teacher-weight $W --teacher-config $C
            --palettize-bits 6 --palettize-refit 100 --park-states
            --batch-frames 3072 --lr 5e-7 --max-steps 4000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 50 --save-every 1000 --val-every 1000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The wide slices (student-slices.md#wide-slices): 9 slices of 1024x3 for the 512x2 ones, whose voice is hoarse, on
  # the shipping trunk (the 6-bit finish's snap, frozen, so the text stays the release's), with Stage 1's recipe and
  # then its decay. Two starts: random slices at Stage 1's rate (2e-4), or the teacher's depformer at its layers
  # 0,2,5 (slices.py --audio-from, which the wide slices' shape takes whole) at half the rate, not to scramble it.
  student-slices-wide-5090|student-slices-wide-warm-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       name=student_slices14_wide start=(--seed 47) lr=2e-4
       [[ $job == *warm* ]] && name=student_slices14_wide_warm lr=1e-4 \
         start=(--audio-from weights/grpo_twoway_best1400.safetensors --audio-layers 0,2,5)
       S=weights/$name
       [[ -f $S/config.json ]] || .venv/bin/python -m training.student.slices \
         --weight weights/pal6_slices_z4000_snap6.safetensors --config-path weights/student_slices14/config.json \
         --dim 1024 --num-heads 16 --layers 3 --parallel 8 "${start[@]}" --out-dir $S
       cmd=(./training/h100.sh run ${name}_stage1
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $S/model.safetensors
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --freeze-trunk --fp8 --fp8-teacher all --batch-frames 12288 --park-states
            --lr $lr --first-codebook-weight 10 --acoustic-ar-weight 3
            --schedule constant --warmup-fraction 0.05 --max-steps 16000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 47 --save-every 4000 --val-every 4000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # Its decay, as Stage 1's: a --schedule decay branch from its update-16,000 pair, hard-linked here first so the
  # branch's own checkpoints never rotate it away.
  student-slices-wide-decay-5090|student-slices-wide-warm-decay-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       name=student_slices14_wide lr=2e-4
       [[ $job == *warm* ]] && name=student_slices14_wide_warm lr=1e-4
       S=weights/$name P=weights/${name}_stage1_step016000
       [[ -d $P ]] || { mkdir -p $P; ln runs/${name}_stage1/{model_step016000.safetensors,trainer_step016000.pt} $P/; }
       cmd=(./training/h100.sh run ${name}_stage1_decay
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $P/model_step016000.safetensors
            --optimizer-state $P/trainer_step016000.pt
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --freeze-trunk --fp8 --fp8-teacher all --batch-frames 12288 --park-states
            --lr $lr --first-codebook-weight 10 --acoustic-ar-weight 3
            --schedule decay --warmup-fraction 0 --max-steps 8000
            --eval-batch-size 64 --keep-checkpoints 1 --seed 49 --save-every 4000 --val-every 4000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  *) echo "usage: $0 asr|voice|voice-en-speech|cache|student-slices-5090|student-slices-decay-5090|student-slices-refit-5090|student-slices-pal6-5090|student-slices-wide-5090|student-slices-wide-decay-5090|student-slices-wide-warm-5090|student-slices-wide-warm-decay-5090" >&2; exit 1 ;;
esac
echo "$(date -u "+%F %T") restart $job" >> $L/$job.log
setsid "${cmd[@]}" >> $L/$job.log 2>&1 < /dev/null &
echo $! > $L/$job.pid
echo "$job pid $(cat $L/$job.pid)"
