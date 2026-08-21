# Mobile student model

The active path is deliberately short:

```text
official 1B init -> BF16 AR distillation -> BF16 parallel-head distillation
-> BF16 export -> MLX BF16 sample translation
```

There is no automatic quality validation, benchmark, qualification receipt, or
student q4 stage. Listen to generated translations before deciding what to
change. Structural checks remain strict: configs, tensor shapes, input hashes,
cache schemas, checkpoint pairs, and base/head lineage must still match.

## Precision contract

- The frozen 3B teacher runs in BF16 inference.
- Student forward and backward compute uses BF16 autocast.
- Loss reduction, optimizer state, and trainable master weights remain FP32.
- Recovery checkpoints remain FP32 for exact resume.
- `student.export_parallel` casts the final listening checkpoint to BF16.

Training the optimizer masters directly in BF16 is not supported. It saves
little H100 memory and weakens small parameter updates.

## 1. Measure and initialize

The AR intermediate keeps the official Hibiki-M width and ordinary depformer.
The listening candidate replaces that large head with `parallel_v1`.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.contract receipt \
  --config student/configs/hibiki_m_12l_ar.json

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.initialize \
  --config student/configs/hibiki_m_12l_ar.json \
  --parent-config PATH/TO/OFFICIAL/config.json \
  --parent-weights PATH/TO/OFFICIAL/hibikim-pytorch.safetensors \
  --output RUN/init.safetensors \
  --receipt RUN/initialization_receipt.json
```

## 2. Build the aligned caches

Input rows contain `id`, `split`, `source_audio`, `target_audio`,
`target_text`, and sorted `text_frames` locating every SentencePiece token,
including EOS, on the 12.5 Hz timeline.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.cache build \
  --pairs PAIRS.jsonl --out-dir CACHE

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.cache build \
  --pairs PAIRS.jsonl --out-dir TEACHER_CACHE --role teacher_context \
  --config TEACHER/config.json --weights TEACHER/model.safetensors \
  --repo kyutai/hibiki-zero-3b-pytorch-bf16 \
  --revision 73175ce6243f8ad66b2138b0264a80044b35c1bd \
  --mimi TEACHER/mimi.safetensors --tokenizer TEACHER/tokenizer.model

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.dump_teacher \
  --teacher-cache TEACHER_CACHE --student-cache CACHE \
  --teacher-config TEACHER/config.json --teacher-weights TEACHER/model.safetensors \
  --teacher-repo kyutai/hibiki-zero-3b-pytorch-bf16 \
  --teacher-revision 73175ce6243f8ad66b2138b0264a80044b35c1bd \
  --out-dir DISTILL_CACHE --top-k 32
```

This materializes teacher text distributions only. Teacher audio sequence
distillation is disabled by default because this repository does not generate
teacher waveforms. If `--teacher-sequence-weight` is made positive later, every
selected sample must contain real teacher speech re-encoded by the student Mimi.

## 3. Distill the AR student

```bash
INIT_SHA=$(sha256sum RUN/init.safetensors | cut -d' ' -f1)

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.train train \
  --cache-dir DISTILL_CACHE \
  --init-checkpoint RUN/init.safetensors --init-sha256 "$INIT_SHA" \
  --out-dir RUN/ar_distill --steps 10000 \
  --batch-size 4 --grad-accum-steps 4 \
  --hard-audio-weight 1 --hard-text-weight 1 \
  --teacher-sequence-weight 0 --teacher-text-weight 1 \
  --rollout-start 0.8 --rollout-fraction 0.25
```

Every save is an FP32 model plus optimizer recovery pair. Resume only from the
newest complete optimizer checkpoint in the same run directory; the trainer
rejects changed data, settings, initialization, or incomplete pairs.

## 4. Distill the parallel head

Use the exact final AR checkpoint. The capture records the normalized hidden
state, sampled text token, prior raw pre-undelay frame, hard targets, and compact
AR-head distributions.

```bash
AR=RUN/ar_distill/model_step010000.safetensors
AR_SHA=$(sha256sum "$AR" | cut -d' ' -f1)

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.capture_parallel \
  --cache-dir DISTILL_CACHE --ar-checkpoint "$AR" --ar-sha256 "$AR_SHA" \
  --out-dir RUN/parallel_cache

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.train_parallel train \
  --cache-dir RUN/parallel_cache \
  --ar-checkpoint "$AR" --ar-sha256 "$AR_SHA" \
  --out-dir RUN/parallel_head --steps 10000
```

Only the 7,346,176-parameter head is trained. The AR backbone and
`text_emb.weight` stay frozen and are bound to the capture and head checkpoints
by their hashes.

## 5. Export and listen in BF16

```bash
HEAD=RUN/parallel_head/head_step010000.safetensors
HEAD_SHA=$(sha256sum "$HEAD" | cut -d' ' -f1)

/opt/homebrew/Caskroom/miniconda/base/bin/python -m student.export_parallel \
  --base-checkpoint "$AR" --base-sha256 "$AR_SHA" \
  --head-checkpoint "$HEAD" --head-sha256 "$HEAD_SHA" \
  --output-weights RUN/parallel/model.bf16.safetensors \
  --output-config RUN/parallel/config.json

/opt/homebrew/Caskroom/miniconda/base/bin/python scripts/convert_mlx_bf16.py \
  --checkpoint RUN/parallel/model.bf16.safetensors \
  --config RUN/parallel/config.json \
  --mimi weights/mimi-pytorch-e351c8d8@125.safetensors \
  --tokenizer weights/tokenizer_spm_48k_multi6_2.model \
  --out-dir RUN/parallel_mlx_bf16

/opt/homebrew/Caskroom/miniconda/base/bin/python main.py assets/samples/leon.wav \
  --model RUN/parallel_mlx_bf16 \
  --out RUN/listen/leon.wav \
  --text-out RUN/listen/leon.txt
```

The last command is the current decision gate: listen to the WAV and inspect
the text. Keep the exact checkpoint and sample outputs you chose; no receipt is
required.
