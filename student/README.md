# Mobile student model track

This directory owns the CUDA-to-MLX model path described in
[`docs/mobile_student_plan.md`](../docs/mobile_student_plan.md). The immutable
starting shapes are:

- `hibiki_m_12l_ar.json`: the full-model CUDA distillation intermediate;
- `hibiki_m_12l_parallel_v1.json`: the deployable frozen-backbone shape.

Both retain the official Hibiki-M 1B width, tokenizer, Mimi contract, and eight
source plus eight target codebooks. The AR checkpoint is deliberately larger
than one billion parameters; deleting its large depformer when installing
`parallel_v1` is what takes the deployable model below one billion parameters.

Measure either shape without allocating weights:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py receipt \
  --config student/configs/hibiki_m_12l_ar.json
```

Initialize the AR student from an explicitly downloaded official 1B config and
checkpoint. The command rejects every missing, extra, or shape-mismatched tensor
before renaming the frozen parent-layer selection:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/initialize.py \
  --config student/configs/hibiki_m_12l_ar.json \
  --parent-config PATH/TO/OFFICIAL/config.json \
  --parent-weights PATH/TO/OFFICIAL/hibikim-pytorch.safetensors \
  --output RUN/init.safetensors \
  --receipt RUN/initialization_receipt.json
```

A release model pack is accepted only when all config-selected files, the parity
fixture, and both receipts exist and match `manifest.json`:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py manifest PACK_DIR
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py validate PACK_DIR
```

Build strict hard-pair caches from JSONL or CSV rows containing `id`, `split`,
`source_audio`, `target_audio`, `target_text`, and `text_frames`. `text_frames`
is a sorted JSON list locating every SentencePiece token, including EOS, on the
12.5 Hz aligned timeline; the builder rejects unaligned text. Cache construction
and model distillation are CUDA-only; MLX is reserved for quantization and
inference.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/cache.py build \
  --pairs PAIRS.jsonl --out-dir CACHE
/opt/homebrew/Caskroom/miniconda/base/bin/python student/cache.py validate \
  CACHE --role student_hard
```

Build the matching 32-stream teacher context from the same aligned pairs. This
cache exists only to run the 3B text head; its audio logits are never exported:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/cache.py build \
  --pairs PAIRS.jsonl --out-dir TEACHER_CACHE --role teacher_context \
  --config TEACHER/config.json --weights TEACHER/model.safetensors \
  --repo kyutai/hibiki-zero-3b-pytorch-bf16 \
  --revision 73175ce6243f8ad66b2138b0264a80044b35c1bd \
  --mimi TEACHER/mimi.safetensors --tokenizer TEACHER/tokenizer.model
```

Every shard repeats one exact metadata object: the full model config and its
SHA-256, Mimi and tokenizer SHA-256 values, the 24 kHz/12.5 Hz contract, and the
17-row layout. The validator checks every shard and sample; legacy 32-stream 3B
caches fail. A teacher cache uses the same format with role `teacher_context`,
33 rows, full teacher config, and a teacher weights SHA-256.

Materialize only compact teacher text distributions; teacher audio heads are
never mapped to student heads:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/dump_teacher.py \
  --teacher-cache TEACHER_CACHE --student-cache CACHE \
  --teacher-config TEACHER/config.json --teacher-weights TEACHER/model.safetensors \
  --teacher-repo kyutai/hibiki-zero-3b-pytorch-bf16 \
  --teacher-revision 73175ce6243f8ad66b2138b0264a80044b35c1bd \
  --out-dir DISTILL_CACHE --top-k 32
```

`teacher_sequence_codes`, when present, is a separately produced `[8, T]`
integer field containing teacher-waveform audio re-encoded by the student Mimi.
It is preserved by the dumper and strictly validated, but this phase does not
generate teacher speech or mislabel hard target audio as teacher audio.

Train every parameter of the 12-layer AR student on CUDA. The trainer accepts
only `student_text_distillation`, re-hashes every cache shard, verifies the
embedded config/Mimi/tokenizer identities, and requires the initialized (or
qualified) checkpoint SHA explicitly. It keeps fp32 master parameters under
bf16 autocast and uses fused AdamW with fixed-size batches and gradient
accumulation:

```bash
INIT_SHA=$(sha256sum RUN/init.safetensors | cut -d' ' -f1)
/opt/homebrew/Caskroom/miniconda/base/bin/python student/train.py train \
  --cache-dir DISTILL_CACHE \
  --init-checkpoint RUN/init.safetensors --init-sha256 "$INIT_SHA" \
  --out-dir RUN/ar_distill --steps 10000 \
  --batch-size 4 --grad-accum-steps 4 \
  --hard-audio-weight 1 --hard-text-weight 1 \
  --teacher-sequence-weight 1 --teacher-text-weight 1 \
  --rollout-start 0.8 --rollout-fraction 0.25
```

The Hugging Face token in `.env` is for staging the pinned inputs; the trainer
does not print it or download artifacts implicitly. `teacher_sequence_codes`
contributes only when actually present. It always means teacher speech decoded
and re-encoded with the student Mimi, never the hard target audio.

Check the loss and rollout mechanics without CUDA or model allocation:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/train.py self-check
```

Gradient checkpointing is on by default. Every save is an exact
`model_stepNNNNNN.safetensors` and `optimizer_stepNNNNNN.pt` pair; the newest
two pairs are retained by default. Resume only from the newest optimizer file
in the same run directory; changed cache shards, artifact identities,
initialization SHA, training settings, missing tensors, or incomplete pairs are
rejected:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/train.py train \
  --cache-dir DISTILL_CACHE \
  --init-checkpoint RUN/init.safetensors --init-sha256 "$INIT_SHA" \
  --out-dir RUN/ar_distill --steps 10000 \
  --batch-size 4 --grad-accum-steps 4 \
  --hard-audio-weight 1 --hard-text-weight 1 \
  --teacher-sequence-weight 1 --teacher-text-weight 1 \
  --rollout-start 0.8 --rollout-fraction 0.25 \
  --resume-optimizer RUN/ar_distill/optimizer_step001000.pt
```

Training and teacher materialization are CUDA-only. Quantization and inference
belong to the MLX phase after the BF16 AR checkpoint qualifies; this trainer
does not generate evaluation audio, quantize weights, or run experiment grids.

After the AR student qualifies, freeze its exact SHA in a receipt with format
`hibiki_student_ar_qualification_v1`, architecture `hibiki_m_12l`, head `ar`,
decision `pass`, and the exact config/checkpoint SHA-256 values. Capture the AR
depformer distributions on its pre-undelay pattern timeline:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/capture_parallel.py \
  --cache-dir DISTILL_CACHE --ar-checkpoint RUN/qualified.safetensors \
  --ar-sha256 "$AR_SHA" --qualification-receipt RUN/qualification_receipt.json \
  --out-dir RUN/parallel_cache
```

Train only the 7,346,176-parameter `parallel_v1` head. `text_emb.weight` is read
from that exact AR checkpoint, kept frozen, and excluded from checkpoints and
the optimizer. The supplied config fixes either one or two passes.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/train_parallel.py self-check
/opt/homebrew/Caskroom/miniconda/base/bin/python student/train_parallel.py train \
  --cache-dir RUN/parallel_cache --ar-checkpoint RUN/qualified.safetensors \
  --ar-sha256 "$AR_SHA" --qualification-receipt RUN/qualification_receipt.json \
  --out-dir RUN/parallel_head --steps 10000
```

Merge the qualified backbone with an exact head only after both explicit hashes
and the head's embedded base/config lineage agree. The output remains BF16/fp32
PyTorch safetensors; MLX owns subsequent q4 quantization and inference.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/export_parallel.py \
  --base-checkpoint RUN/qualified.safetensors --base-sha256 "$AR_SHA" \
  --qualification-receipt RUN/qualification_receipt.json \
  --head-checkpoint RUN/parallel_head/head_step010000.safetensors \
  --head-sha256 "$HEAD_SHA" --output-weights RUN/parallel/model.safetensors \
  --output-config RUN/parallel/config.json
```
