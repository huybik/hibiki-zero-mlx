# Slices student

Read when touching `training/student/slices.py` or `restart.sh student-slices-5090`, or training,
scoring or deciding on the phone student whose audio part is depformer slices. Back to
[CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Why this shape](#why-this-shape) ·
[Making it](#making-it) · [Training](#training) · [Runs](#runs) · [Not yet](#not-yet). The phone side
(export, Swift, A15 timings) is in [lm-npu](lm-npu.md#depformer-slices); the heads student it
replaces in [student-heads](student-heads.md).

## Design

The heads student's 14-layer trunk with a small depformer for its 16 parallel heads, which lost the
speaker's voice ([student-heads](student-heads.md#codebooks)):

```text
trunk out ─► slice 0 (reads text) ─► cb 0 ─► slice 1 ─► cb 1 ─► ... ─► slice 8 (reads cb 7) ─► cb 8..15
```

- 9 slices of 512 × 2 (8 heads, moshi's depformer layers: RMS norms, SiLU gating, no positional
  embedding), each with its own weights, input linear from the trunk, input embedding of the
  token before it and output LayerNorm; 16 output linears, the last slice's 8 read its one
  output. Slice k reads codebook k − 1, as moshi's steps do; codebooks 8–15 each read 0–7 of their
  frame, not each other.
- Delays are the teacher's: codebook 0 at the text's, 1–15 two frames later, so a step's codebooks
  1–15 are one frame and the slices follow its residual quantizer as the teacher's depformer does.
  The heads student's min(j, 8) delays go: a frame's speech completes 2 frames after its first
  codebook (160 ms), not 8 (640 ms). Distillation's targets line up codebook for codebook, but
  for 9–15, which the teacher reads 8..j − 1 for.
- Config: `"audio_heads": "depformer_parallel"`, `"parallel_codebooks": 8`, and moshi's own
  depformer keys, the step schedule codebook j → slice min(j, 8) (`slices.schedule`). moshi builds
  the slices' weights from them; `slices.install` drops the codebooks' input embeddings and norms
  past the slices' and runs the steps: `_forward_training` (teacher-forced, through
  `lm_training.run_depformer`) and `LMGen.depformer_step` (sampled slice by slice, the last
  slice's 8 at once). `common.load_checkpoint_info` takes both keys out before moshi reads the
  config; `common.build_lm` installs the slices.
- Checked on a tiny CPU model (2026-11-27, 4 codebooks, 3 slices): greedy `depformer_step`'s
  tokens are the argmax of the teacher-forced logits fed those tokens, on every codebook.

## Why this shape

The A15's Neural Engine runs a slice a Core ML call at ~1.8 ms a call whatever its size, so the
step count sets the cost: 9 slices of 512 × 2 held 67 ms a frame for 10 minutes on the heads
release's trunk, under the frame's 80; 1024 × 3, the depformer student's shape, missed it (78 ms,
p90 80) ([lm-npu](lm-npu.md#depformer-slices)). 16 sequential steps (the teacher's) would cost
~29 ms of calls alone. Fewer steps, grouping codebooks 3–7, or one call for all slices with the
sampling inside, are later levers if the frame needs room. The trunk stays the heads student's,
whose text matched the depformer student's before its own on-policy stage (test chrF 39.87 against
39.48 after GRPO, [student-heads](student-heads.md#tests)).

## Making it

`python -m training.student.slices --weight W --config-path C --out-dir D` writes `D/config.json`
and `D/model.safetensors`: the parent student's trunk, text and input embeddings copied whole, its
audio part (heads or depformer) dropped, its target delays set to its source stream's (the
teacher's), the slices initialized as moshi initializes a depformer (`--seed`); `--dim`,
`--num-heads`, `--layers`, `--parallel` (defaults 512, 8, 2, 8). The weight's metadata keeps the
parent's (its tokenizer among it) with `student_of` and a `student` record (the sizes, the seed
and the parent's own record), so the dir recuts exactly. Checked on a tiny heads parent: the
trunk comes back bit for bit and the student loads strictly.

## Training

`restart.sh student-slices-5090` cuts `weights/student_slices14` from `both_heads_z_best1200` (the
heads student after its on-policy stage) and runs the heads student's recipe on the 5090:
distilled from GRPO update 1,400 on the two-way run's rows both ways, 12,288-frame batches
(`--bf16-adamw --micro-frames 6144`), peak 5e-5 with 5% warmup and cosine, seed 47; one epoch
(95,063 updates, the trunk already trained), validation every 8,000 with the voice scores
([voice-scores](voice-scores.md)). The trunk trains too: its audio inputs move to the new delays.
The voice decides at 8,000 and 16,000: `mean_speaker_sim` and `mean_hnr_db` against the heads
student's and the depformer student's on the same sets.

## Runs

None yet.

## Not yet

- The MLX runtime and `convert_mlx.py` (moshi_mlx refuses `depformer_parallel`), so neither the Mac
  gate nor the Core ML export reads a trained slices checkpoint yet.
- Palettization against the export's tables (`train.py --palettize-bits` takes heads students only).
