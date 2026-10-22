# Mobile student model

Read when touching `student/`, `docs/mobile_student_plan.md` or the `parallel_v1` head.
Back to [CONTEXT.md](../../CONTEXT.md) · [Pipeline](#pipeline) · [Architecture](#architecture) ·
[Invariants](#invariants) · [Status](#status)

Commands live in [student/README.md](../../student/README.md); the design is in
[docs/mobile_student_plan.md](../mobile_student_plan.md).

## Pipeline

```text
official Hibiki-M init ─► AR distillation from the frozen 3B teacher ─► capture
  ─► parallel-head distillation ─► BF16 export ─► scripts/convert_mlx_bf16.py ─► main.py --model
```

Each stage is `python -m student.<module>`: `contract` (shape receipt), `initialize`, `cache`
(student and `teacher_context` caches), `dump_teacher` (teacher text distributions), `train`
(AR), `capture_parallel`, `train_parallel` and `export_parallel`. Shared trainer pieces are in
`harness.py`; the head is defined in `parallel.py`.

## Architecture

- `student/configs/hibiki_m_12l_{ar,parallel_v1}.json`: dim 2048, 16 heads, `hidden_scale` 4.125,
  8 generated codebooks, `text_card` 48000, and 12 layers taken from the parent
  (indices 0 1 3 4 5 7 8 10 11 12 14 15).
- The `parallel_v1` head has 7,346,176 parameters (`student/parallel.py`), the backbone
  880,400,896 and the total 887,747,072 (`student/receipts/hibiki_m_12l_parallel_v1.json`).

## Invariants

- Precision: the teacher runs BF16 inference and the student uses BF16 autocast. Loss
  reduction, optimizer state, master weights and recovery checkpoints stay FP32. Only
  `export_parallel` casts to BF16; BF16 master weights are unsupported.
- Structural checks stay strict: configs, tensor shapes, input hashes, cache schemas,
  checkpoint pairs and base/head lineage. Every stage takes the previous artifact plus its
  SHA-256.
- Resume only from the newest complete model and optimizer pair in the same run directory.
- The parallel stage trains only the head. The AR backbone and `text_emb.weight` are frozen
  and bound by hash.
- Teacher audio-sequence distillation is off (`--teacher-sequence-weight 0`) because the repo
  generates no teacher waveforms.
- `student/` imports PyTorch `moshi` but nothing from `finetune/`.
- Stock moshi-swift cannot run `parallel_v1` (see [mlx-inference](mlx-inference.md#gates)).

## Status

The code is complete, but no CUDA run or listening result is recorded. There is no automatic
quality validation, benchmark, qualification receipt or student q4 stage; listening to
`main.py` output is the decision gate.
