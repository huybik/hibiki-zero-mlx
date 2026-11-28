#!/usr/bin/env bash
# restart.sh asr|mt|voice|filters|cache|envi-cache|two-way|two-way-5090|student|student-heads|student-heads-5090|student-slices-5090|student-slices-decay-5090|student-slices-refit-5090|student-slices-pal6-5090|student-distill|yodas-pilot|yodas-control: stop the job by its PID file (its whole process group), start it again.
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
  # The VI->EN server translate.py calls; it needs the GPU memory voice holds, so the two take turns.
  mt) read -ra cmd <<< "$(/workspace/envs/translate/bin/python -m training.tranche.translate serve)" ;;
  # The scale-up: each reruns 30 s after a run that succeeds, as the ingest and translation bring
  # shards (a run takes the shards ready at its start); the first error stops the loop.
  # Voice reruns after a failure too (an ASR request timing out at 60 s ends a run about once a
  # day: 2026-11-06, 2026-11-07 17:30 UTC+7), but stops after 3 in a row.
  voice) cmd=(bash -c 'f=0; while (( f < 3 )); do if /workspace/envs/vieneu/bin/python -m training.tranche.voice run yodas; then f=0; else f=$((f+1)); echo "$(date -u "+%F %T") voice run failed ($f in a row)"; fi; sleep 30; done') ;;
  filters) cmd=(bash -c 'while /workspace/envs/nanovox/bin/python -m training.tranche.filters asr giga; do sleep 120; done') ;;
  # The cache builds run in the free-threaded env (envs-min.sh), Mimi's rope eager there (no Triton). Plain HTTP for the
  # Hub: 143 MB/s, where Xet throttles to ~4 MB/s.
  # Both directions: each pass builds the shards its cache folder on the Hub lacks; the en-vi pass, second, deletes
  # each shard's parts from the HF cache once read.
  cache) export HF_HUB_DISABLE_XET=1 NO_TORCH_COMPILE=1
         cmd=(bash -c 'set -e; P=/workspace/envs/cache/bin/python
              $P -m training.tranche.cache build pho vivoice giga yodas
              $P -m training.tranche.cache build pho vivoice giga yodas --direction en-vi --drop-inputs') ;;
  # The two-way run's English-to-Vietnamese inputs (two-way.md#two-way-run): the English evaluation sets, round 4e's 500
  # en_speech shards (envi_en_speech.sh, which pushes them at its end), then every tranche shard by the tranche builder
  # (pushed to its cache repo's en-vi/ as it goes; each shard's parts deleted once read). ~8 h on the H100 (2,490x real
  # time on the Mimi audio).
  envi-cache) export HF_HUB_DISABLE_XET=1
       cmd=(bash -c 'set -e; .venv/bin/python -m training.prep.eval_sets --sets fleurs-en en-vi-val-en
            training/tranche/box/envi_en_speech.sh
            NO_TORCH_COMPILE=1 /workspace/envs/cache/bin/python -m training.tranche.cache build pho vivoice giga \
              --direction en-vi --drop-inputs') ;;
  # The two-way SFT (two-way.md#two-way-run): round 4e's rows both ways, from round 4's final model moved onto the
  # Vietnamese swap (init_weights writes weights/twoway_start_vi6k.safetensors; --tokenizer follows the weight), in
  # the trainer's whole 4,608-frame batches at its 2.25e-6 peak. Every cache keeps only its packs here (streams
  # --drop-shards). Validation every 20,000 updates on both directions. twoway_sft ran round 4's 8,192-frame batches
  # in slices at 3e-6, so this is a new run, twoway_sft_4608. Two epochs (user, 2026-11-11), but the next run ends
  # sooner (sft-training.md#trainer): decide its --max-steps with the user before launching it.
  two-way) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       # Shards beside their packs would fill the disk at the second checkpoint pair: drop them first (two-way.md).
       for d in $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train; do
         [[ -d $d ]] || { echo "$d is missing: run envi-cache first" >&2; exit 1; }
         if compgen -G "$d/shard_*.pt" > /dev/null; then echo "$d still holds shards: streams --drop-shards it first" >&2; exit 1; fi
       done
       cmd=(./training/h100.sh run twoway_sft_4608
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --model-weight weights/twoway_start_vi6k.safetensors
            --eval-batch-size 64 --keep-checkpoints 1
            --epochs 2 --seed 47 --save-every 4000 --val-every 20000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The phone students (phone-student.md#training-a-student), each cut from the shipping model, GRPO update 1,400
  # (weights/grpo_twoway_best1400.safetensors: the recovery repo's grpo_twoway/best/best_step001400.safetensors) when
  # its directory is missing: `student` the depformer student (10 layers, a 3-layer depformer, 8 codebooks),
  # `student-heads` the parallel heads (14 layers, 16 codebooks, 8-15 at one delay: student-heads.md), distilled from
  # that model from the start (--teacher-weight, phone-student.md#distillation). Both train on the two-way run's rows
  # both ways, the same streams (seed 47) for the same steps, in whole 12,288-frame batches (51,000 frames/s on the
  # depformer student, +32% over 4,608) at peak 5e-5 (the probe, phone-student.md#next).
  student|student-heads) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       if [[ $job == student ]]; then name=student_dep10 cut=() teach=()
       else name=student_heads14_cb16 cut=(--heads --num-layers 14 --codebooks 16) teach=(--teacher-weight weights/grpo_twoway_best1400.safetensors); fi
       S=weights/$name
       [[ -f $S/config.json ]] || .venv/bin/python -m training.student.cut \
         --weight weights/grpo_twoway_best1400.safetensors "${cut[@]}" --out-dir $S
       cmd=(./training/h100.sh run $name
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $S/model.safetensors "${teach[@]}"
            --batch-frames 12288 --lr 5e-5 --eval-batch-size 64 --keep-checkpoints 1
            --epochs 2 --seed 47 --save-every 4000 --val-every 16000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The depformer student distilled from its teacher, GRPO update 1,400 (phone-student.md#distillation): from
  # student_dep10_distill's model 32,000 (hard-linked to weights/student_dep10_anneal_start.safetensors when that run
  # stopped: its KL stalled at peak 3e-5), the same rows on new streams (seed 49), each token's loss its KL from the
  # teacher's distribution, at peak 1e-5. --max-steps ends it with time left on the box for a GRPO run after it.
  student-distill) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2 S=weights/student_dep10
       cmd=(./training/h100.sh run student_dep10_anneal
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight weights/student_dep10_anneal_start.safetensors
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --batch-frames 12288 --lr 1e-5 --warmup-fraction 0.02 --max-steps 110000 --eval-batch-size 64
            --keep-checkpoints 1 --seed 49 --save-every 4000 --val-every 8000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # 5090 route for 3B full-model two-way SFT: all-GPU bf16 weights/grads/moments with fused Triton
  # stochastic rounding (--bf16-adamw: 23.3 GiB static, +0.00001 loss parity vs fp32 AdamW),
  # whole-loop compiled selective checkpointing (first 14 main + depformer layers) and whole
  # 4,608-frame batches (6,554 fr/s, 28.3 GiB reserved on real variable batches; or add --fp8 with
  # --grad-checkpoint-layers 8 for 7,242 fr/s at 28.1 GiB reserved).
  two-way-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       for d in $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train; do
         [[ -d $d ]] || { echo "$d is missing: run envi-cache first" >&2; exit 1; }
         if compgen -G "$d/shard_*.pt" > /dev/null; then echo "$d still holds shards: streams --drop-shards it first" >&2; exit 1; fi
       done
       cmd=(./training/h100.sh run twoway_sft_5090
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --model-weight weights/twoway_start_vi6k.safetensors
            --bf16-adamw --grad-checkpoint --grad-checkpoint-layers 14 --park-states --batch-frames 4608
            --eval-batch-size 64 --keep-checkpoints 1
            --epochs 2 --seed 47 --save-every 4000 --val-every 20000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # 5090 route for 14-layer 16-codebook parallel-heads student distillation from the 3B teacher:
  # --bf16-adamw + 6,144-frame slices (12,362 fr/s, 26.0 GiB peak, no checkpointing needed).
  # For exact fp32 AdamW instead, drop --bf16-adamw and use --grad-checkpoint --micro-frames 3072 (10,065 fr/s, 27.35 GiB).
  student-heads-5090) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       name=student_heads14_cb16 S=weights/$name
       [[ -f $S/config.json ]] || .venv/bin/python -m training.student.cut \
         --weight weights/grpo_twoway_best1400.safetensors --heads --num-layers 14 --codebooks 16 --out-dir $S
       cmd=(./training/h100.sh run ${name}_5090
            --cache-dir $T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
            $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --config-path $S/config.json --model-weight $S/model.safetensors
            --teacher-weight weights/grpo_twoway_best1400.safetensors
            --bf16-adamw --batch-frames 12288 --micro-frames 6144 --park-states
            --lr 5e-5 --eval-batch-size 64 --keep-checkpoints 1
            --epochs 2 --seed 47 --save-every 4000 --val-every 16000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
  # The YODAS pilot (tranche-yodas.md#pilot): 20,000 updates from the two-way SFT's final model on two-way-5090's
  # rows and route, the cosine over them; yodas-pilot adds YODAS's caches both ways, yodas-control does not. One
  # seed, validated at the end; the test sets then compare the two.
  yodas-pilot|yodas-control) export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft OMP_NUM_THREADS=$N MAX_JOBS=$N TORCHINDUCTOR_COMPILE_THREADS=$N
       T=/workspace/tranche/repo/cache E=data/eval V=data/cache/en-vi/en-vi-grounded-v2
       dirs=($T/pho $T/vivoice $T/giga data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train
             $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga $V/train)
       if [[ $job == yodas-pilot ]]; then dirs+=($T/yodas $T/en-vi/yodas); fi
       for d in "${dirs[@]}"; do
         [[ -d $d ]] || { echo "$d is missing" >&2; exit 1; }
         if compgen -G "$d/shard_*.pt" > /dev/null; then echo "$d still holds shards: streams --drop-shards it first" >&2; exit 1; fi
       done
       cmd=(./training/h100.sh run ${job/-/_}
            --cache-dir "${dirs[@]}"
            --val-cache-dir data/cache/en-vi-grounded-v2/validation $V/validation
            --model-weight weights/twoway_final.safetensors
            --bf16-adamw --grad-checkpoint --grad-checkpoint-layers 14 --park-states --batch-frames 4608
            --max-steps 20000 --eval-batch-size 64 --keep-checkpoints 1
            --seed 47 --save-every 4000 --val-every 20000
            --eval-pairs $E/fleurs-vi/dev.jsonl $E/fleurs-en/dev.jsonl $E/multimed-vi/select.jsonl
            $E/en-vi-val/cv27.jsonl $E/en-vi-val/libriheavy.jsonl $E/en-vi-val-en/cv27.jsonl $E/en-vi-val-en/libriheavy.jsonl) ;;
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
  # data (the joint stage, --trunk-lr 1e-5) lost chrF on every set by 8,000:
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
  *) echo "usage: $0 asr|mt|voice|filters|cache|envi-cache|two-way|two-way-5090|student|student-heads|student-heads-5090|student-slices-5090|student-slices-decay-5090|student-slices-refit-5090|student-slices-pal6-5090|student-distill|yodas-pilot|yodas-control" >&2; exit 1 ;;
esac
echo "$(date -u "+%F %T") restart $job" >> $L/$job.log
setsid "${cmd[@]}" >> $L/$job.log 2>&1 < /dev/null &
echo $! > $L/$job.pid
echo "$job pid $(cat $L/$job.pid)"
