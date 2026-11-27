# Slices student

Read when touching `training/student/slices.py`, `restart.sh student-slices-5090`, `train.py
--freeze-trunk` or moshi_mlx's
`parallel_codebooks`, or training,
scoring or deciding on the phone student whose audio part is depformer slices. Back to
[CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Why this shape](#why-this-shape) ·
[Making it](#making-it) · [Training](#training) · [Runs](#runs) · [On the Mac](#on-the-mac) ·
[Not yet](#not-yet). The phone side
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

`restart.sh student-slices-5090` (Stage 1) and `restart.sh student-slices-decay-5090` (its decay)
train `weights/student_slices14` (cut from `both_heads_z_best1200`) on the RTX 5090 with the trunk
frozen: a trunk trained on this data lost text (the joint stage below).

- **Stage 1 (`student_slices14_stage1`, `--freeze-trunk`)**: warms up the 9 random `512 × 2` slices
  (`slices.AUDIO_PARTS`, 134.7M params) while keeping the trunk, embeddings, and text head frozen
  bit-for-bit. Uses `--fp8 --fp8-teacher all --lr 2e-4 --first-codebook-weight 10
  --acoustic-ar-weight 3 --schedule constant --max-steps 16000` (giving AR codebooks `cb 1..7` 54%
  of audio loss instead of 6.1%, and skipping student trunk backward, `text_linear`, and `text_kl`).
  A frozen trunk reads the new `[0, 0, 2, ..., 2]` delays immediately (`text KL = 0.114` vs `0.088`).
- **Stage 1's decay (`student_slices14_stage1_decay`)**: a `--schedule decay` branch from Stage
  1's update-16,000 pair (both files hard-linked into `weights/student_slices14_stage1_step016000/`),
  the same recipe with the rate 2e-4 to 0 as 1 − sqrt over 8,000 updates, seed 49, validating at
  4,000 and 8,000.
- **The joint stage (`student_slices14_stage2`, stopped)**: from Stage 1's 16,000 with
  `--trunk-lr 1e-5 --lr 5e-5 --first-codebook-weight 20 --acoustic-ar-weight 2 --micro-frames 6144
  --schedule constant`, meant to let the trunk adapt to the new delays without text drift. It
  drifted: by 8,000 the voice rose and every set's chrF fell ([Runs](#runs)). Its job was
  `restart.sh student-slices-joint-5090` in 6afc776.
- **Diagnostics**: `train_log.jsonl` and `val_log.jsonl` always log `acoustic_ar_loss` (`cb 1..7`)
  and `acoustic_par_loss` (`cb 8..15`) separately, plus `trunk_grad_norm`, `audio_grad_norm`,
  `clipped_frac`, and `trunk_lr` (without extra per-step GPU syncs). Stage 1's training log has
  text losses of 0 and `loss` = `audio`: its frozen trunk's text is not computed (its
  validation's `loss` still includes text). `validate.py --acoustic-ar-weight` reproduces a
  stage's teacher-forced `loss`.

**Training optimizations (2026-11-27, branch `slices-opt`, box 62513802 RTX 5090, `pho` + `en-vi/pho`
12,288-frame batches; baseline `db0de25` vs optimized, warm updates 20–50):**

- **Exact Track A** (`common.py`, `lm_training.py`, `slices.py`, `train.py`): chunked fused
  linear+KL (`compute_distill_losses_chunked`, eliminating full-vocab logit materialization and
  `_undelay_sequence` NaN copies, saving ~5.5 GiB), batched contiguous schedule runs in `_per_step`
  (8× fewer teacher slice-8 GEMMs/HBM reads), fused `depformer_in` and `linears[8..15]` GEMMs,
  cached RoPE tables, compiled embeddings/norms, cuDNN SDPA for forward-only teacher depformer
  attention (1.57× faster on Blackwell), and freeing `trunk_grads`/`audio_grads` before `optimizer.step`
  (preventing a 2.69 GiB step-2 grad-list leak).
- **Blackwell `sm_120` FP8 Track B** (`fp8_linear.py`): fused Triton rowwise E4M3 quantization
  (1-pass for `K <= 8192`, 2-pass for wider rows, fused tiled `BLOCK_M=256, BLOCK_K=16`
  transpose+quantize 3.8× faster on 170 SMs) registered as `torch.library.custom_op` (zero graph
  breaks under whole-loop `torch.compile`), pre-quantized frozen teacher (`--fp8-teacher trunk|all`,
  halving teacher weight VRAM and avoiding runtime weight quantization), and plain-attribute FP8
  weight caching for frozen student trunk linears under `--freeze-trunk --fp8` (keeps `state_dict()`
  and bf16 eval intact).
- **Teacher FP8 target shift** (5 real variable-shape batches, `KL(bf16 || fp8)` / argmax flip):
  `--fp8-teacher trunk`: `text 0.00197 (0.89%)`, `cb 0 0.00280 (2.53%)`, `cb 1..7 0.00957 (7.57%)`,
  `cb 8..15 0.00745 (7.58%)`; `--fp8-teacher all`: `text 0.00197 (0.89%)`, `cb 0 0.00482 (3.37%)`,
  `cb 1..7 0.02270 (11.65%)`, `cb 8..15 0.03674 (16.56%)` — under ~0.3–1.1% of student acoustic KL.

| Recipe (12,288 frames/update, `--bf16-adamw`) | Micro-frames | s/update | frames/s | Peak (`nvidia-smi`) | Step-50 loss |
|---|---:|---:|---:|---:|---|
| Baseline joint (`db0de25`) | 4,096 (3×) | 0.985 | 12,475 | 25.9 GiB | `step40=2.6602` |
| Opt joint exact bf16 (Track A) | 6,144 (2×) | 0.946 | 12,989 | 27.8 GiB | `2.2784` (`step40=2.6599`) |
| Opt joint `--fp8 --fp8-teacher trunk` | 6,144 (2×) | 0.764 | 16,084 | ~27.0 GiB | `2.2793` (`text=0.1499`) |
| **Opt joint `--fp8 --fp8-teacher all`** | **6,144 (2×)** | **0.698** | **17,604** | **25.5 GiB** (`24.2` alloc) | **`2.2785`** (`+0.0001`) |
| Opt joint `--fp8 --fp8-teacher all --grad-checkpoint-layers 2` | whole (1×) | 0.690 | 17,809 | 31.2 GiB | `2.2826` |
| Baseline frozen (`db0de25 --freeze-trunk`) | whole (1×) | 0.699 | 17,579 | 25.2 GiB | `audio=3.7827` |
| Opt frozen exact bf16 (Track A + R1) | whole (1×) | 0.620 | 19,819 | 16.7 GiB | `audio=3.7823` |
| Opt frozen `--fp8-teacher trunk` | whole (1×) | 0.534 | 23,011 | 15.1 GiB | `audio=3.7771` |
| Opt frozen `--fp8-teacher all` | whole (1×) | 0.472 | 26,034 | 15.1 GiB | `audio=3.7763` |
| **Opt frozen `--fp8 --fp8-teacher all`** | **whole (1×)** | **0.428** | **28,710** | **15.8 GiB** (`13.7` alloc) | **`audio=3.7741`** |

`torch.profiler` (updates 26–28, `prof_train.py`): Stage 1 is 96.4% GPU busy (`52.1%` FP8 `_scaled_mm`,
`7.5%` bf16 `mm`, `5.2%` Triton 1-pass quant, `13.72 GiB` peak allocated); Stage 2 is 99.0% GPU busy
(`48.9%` FP8 `_scaled_mm`, `11.5%` bf16 `mm`, `4.9%` transpose-quant, `3.2%` 1-pass quant, `2.3%`
AdamW, `24.19 GiB` peak allocated). A validation (all 7 sets, free-running, voice scores,
`--park-states`) took 291 s at 26.1 GiB. Two-stage total (16k Stage 1 @ 0.428 s + 40k Stage 2 @
0.698 s + 7 validations) is ~10.3 h (~$5.25 on RTX 5090 vs ~26.7 h / $13.65 baseline).
The voice decides at 8,000 and 16,000: `mean_speaker_sim` and `mean_hnr_db` against the heads
student's and the depformer student's on the same sets.

## Runs

- `student_slices14_stage1_decay`: started 2026-11-27 15:24 UTC on box 62513802, commit e9e6e81,
  `restart.sh student-slices-decay-5090`; HF recovery prefix `student_slices14_stage1_decay`. AdamW
  continues from Stage 1's step 16,000; the rate 1.9e-4 at update 10, 1.8e-4 at 100 (1 − sqrt).
  Runs at 0.43 s an update, 15 GiB, the GPU at 100% and ~550 W; its loss continues Stage 1's
  (0.74–0.86 over updates 10–100, grad norm 0.29–0.54). Validations (cells as Stage 1's table;
  outputs in `/Volumes/data/hibiki-zero-evals/box-62513802/student_slices14_stage1_decay/`):

  | Update (date) | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|
  | 4,000 (11-27) | 38.04 | 40.98 / 9.86 / 1.38 / 1.3 / 0.34 / 4.4 | 35.09 / 15.95 / 1.00 / 1.8 / 0.16 / 5.3 | 38.13 / 10.35 / 1.47 / 3.2 / 0.27 / 5.9 | 45.81 / 17.00 / 1.24 / 2.2 / 0.33 / 4.2 | 45.90 / 18.26 / 1.17 / 2.2 / 0.41 / 4.4 | 35.43 / 16.89 / 1.15 / 1.2 / 0.26 / 4.8 | 43.59 / 22.40 / 1.08 / 1.7 / 0.30 / 5.2 |

  Health passed (empty rows: 2 on fleurs-en). At 4,000 (rate 5.7e-5), against Stage 1's 16,000:
  seven-set mean chrF 40.55 → 40.70 and selecting 38.41 → 38.04 (inside evaluation noise, ±0.7; the
  teacher-forced text loss is unchanged, 1.885), mean speaker_sim 0.280 → 0.296, HNR 4.89 → 4.89 dB;
  teacher-forced semantic / cb 1–7 / cb 8–15 1.929 / 3.938 / 4.866 → 1.914 / 3.886 / 4.819 on
  `en-vi-grounded-v2/validation`.
- `student_slices14_stage2`: started 2026-11-27 13:29 UTC on box 62513802, commit d015ab6
  (PR #205), `restart.sh student-slices-joint-5090` from Stage 1's update 16,000; HF recovery
  prefix `student_slices14_stage2`. Ran at 0.70 s an update, 26.2 GiB, the GPU at 99% and ~520 W;
  was to validate every 8,000 to 40,000; stopped by hand at 9,230 (2026-11-27 15:23 UTC; user:
  "stop stage 2 now and start the decay"), its 8,000 pair on the box. Its first updates continue Stage 1's acoustic losses
  (cb 1–7 0.94, cb 8–15 1.21); training text KL 0.095 (updates 1–200) → 0.072 (1,001–1,500), flat
  after. Validations (cells as Stage 1's table; outputs in
  `/Volumes/data/hibiki-zero-evals/box-62513802/student_slices14_stage2/`):

  | Update (date) | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|
  | 8,000 (11-27) | 33.58 | 39.31 / 8.15 / 1.42 / 1.9 / 0.36 / 4.4 | 27.85 / 11.34 / 0.98 / 2.0 / 0.22 / 5.4 | 36.15 / 8.49 / 1.58 / 4.0 / 0.31 / 5.8 | 42.83 / 14.71 / 1.19 / 2.2 / 0.35 / 4.3 | 44.24 / 16.93 / 1.14 / 2.8 / 0.45 / 4.4 | 32.29 / 15.19 / 1.11 / 0.8 / 0.29 / 5.1 | 42.00 / 20.11 / 1.06 / 2.0 / 0.32 / 5.2 |

  Health passed (empty rows: 1 each on fleurs-en, en-vi-val cv27, en-vi-val-en cv27). The joint
  stage trades text for voice: against Stage 1's 16,000, every set's chrF fell (seven-set mean
  40.55 → 37.81, selecting 38.41 → 33.58, FLEURS EN→VI −7.48, the rest −1.16 to −2.55) while the
  voice rose (mean speaker_sim 0.280 → 0.329, HNR 4.89 → 4.94 dB), though the teacher-forced text
  improved (content loss 1.885 → 1.816 on `en-vi-grounded-v2/validation`). FLEURS EN→VI answers in
  English on 77 of 392 rows (no Vietnamese letter; Stage 1: 12), garbled English rather than a
  transcript, and scores 31.96 without them. Outputs end with text EOS far more often (FLEURS-vi
  93 → 231 / 361, FLEURS-en 24 → 183 / 392) and run longer (FLEURS-vi length ratio 1.31 → 1.42).
  The trunk's text came from on-policy training (`both_heads_z`, which took FLEURS EN→VI
  wrong-language rows to 0, [heads-on-policy-runs](heads-on-policy-runs.md)); teacher-forced
  distillation at `--trunk-lr 1e-5` undoes it, as the first joint run's text did at 5e-5.
- `student_slices14_stage1`: started 2026-11-27 11:20 UTC on box 62513802, commit 65f6b51
  (PR #205), `restart.sh student-slices-5090`; HF recovery prefix `student_slices14_stage1`.
  Ran at 0.43 s an update, 15 GiB, the GPU at 100% and ~535 W; validated at 8,000 and 16,000.
  Before it, a 20-update frozen run on the same recipe saved a trunk equal to the cut's on all 119
  tensors (compared in bf16, the weight's dtype, with text_emb's row 2 set to row 3 as `build_lm`
  loads it), and all 128 slice tensors moved. Validations (cells as the table below; outputs in
  `/Volumes/data/hibiki-zero-evals/box-62513802/student_slices14_stage1/`):

  | Update (date) | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|
  | 8,000 (11-27) | 38.33 | 41.43 / 9.98 / 1.39 / 1.6 / 0.27 / 4.0 | 35.22 / 16.00 / 1.01 / 1.9 / 0.15 / 4.7 | 38.38 / 10.74 / 1.43 / 3.4 / 0.24 / 5.4 | 44.29 / 16.81 / 1.18 / 2.4 / 0.28 / 3.5 | 45.27 / 18.23 / 1.12 / 2.2 / 0.37 / 3.8 | 35.82 / 17.32 / 1.14 / 1.0 / 0.25 / 4.2 | 44.16 / 22.41 / 1.07 / 1.8 / 0.25 / 4.5 |
  | 16,000 (11-27) | 38.41 | 41.49 / 10.38 / 1.31 / 1.4 / 0.33 / 4.6 | 35.33 / 16.24 / 0.99 / 1.9 / 0.15 / 5.2 | 38.31 / 10.87 / 1.41 / 3.3 / 0.25 / 6.0 | 44.66 / 16.75 / 1.16 / 2.2 / 0.30 / 4.2 | 45.40 / 18.65 / 1.09 / 1.9 / 0.40 / 4.5 | 34.84 / 16.62 / 1.13 / 1.1 / 0.26 / 4.8 | 43.82 / 21.98 / 1.06 / 1.7 / 0.27 / 4.9 |

  Every set's health passed (empty rows at 8,000: 2 on fleurs-en, 1 on multimed; at 16,000: 3 on
  fleurs-en). Rows ending with text EOS, 8,000 → 16,000, in the table's order: 111 → 93 / 361,
  25 → 24 / 392, 511 → 458 / 1,718, 60 → 56, 58 → 51, 24 → 16, 11 → 12 / 180. At 8,000, against the
  first run's 8,000: seven-set mean 40.65 against 34.74, selecting 38.33 against 30.50, and the
  voice up on every set (FLEURS-vi 0.27 / 4.0 dB against 0.15 / 2.9). 8,000 → 16,000 the text held
  (seven-set mean 40.65 → 40.55, selecting 38.33 → 38.41: the frozen trunk writes it) while the
  voice rose (mean speaker_sim 0.259 → 0.280, HNR 4.30 → 4.89 dB). Training losses, mean of the
  1,000 updates before 8,000 → 16,000: cb 0 0.209 → 0.184, cb 1–7 1.105 → 0.940, cb 8–15 1.368 →
  1.209 (the first run's acoustic mean 1.55 at 8,000). Ended 13:26 UTC after 16,000's validation
  (2 h 06 min, ~$1.07); `best.json` names 16,000, the final model.
- `student_slices14_5090`: started 2026-11-27 06:38 UTC on box 62513802 (RTX 5090), commit
  35e35d9, `restart.sh student-slices-5090` with the trunk training; HF recovery prefix
  `student_slices14_5090`. Ran at 0.99 s an update, 26.1 GiB, the GPU at 100% and 505 W; stopped
  by TERM at 8,520 (2026-11-27 ~09:10 UTC; user: "start the slice training only"), its 8,000 pair
  on the box and the Hub. Validations
  (seed 47; chrF / BLEU / length ratio / end offset in s / speaker_sim / HNR dB; every set's
  health pass, no empty outputs; outputs in
  `/Volumes/data/hibiki-zero-evals/box-62513802/student_slices14_5090/`):

  | Update (date) | Selecting | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
  |---|---|---|---|---|---|---|---|---|
  | 8,000 (11-27) | 30.50 | 34.98 / 5.24 / 1.63 / 2.2 / 0.15 / 2.9 | 26.03 / 8.47 / 1.06 / 2.2 / 0.10 / 4.0 | 33.29 / 6.41 / 1.66 / 4.2 / 0.18 / 5.0 | 38.65 / 11.07 / 1.38 / 2.5 / 0.21 / 3.5 | 41.18 / 13.30 / 1.13 / 3.0 / 0.27 / 3.5 | 29.90 / 11.31 / 1.20 / 0.7 / 0.18 / 3.7 | 39.12 / 16.86 / 1.10 / 2.0 / 0.18 / 3.8 |

  At 8,000: text behind the heads student's final (190,126, same validation): seven-set mean
  34.74 against 37.72, selecting 30.50 against 36.92, FLEURS EN→VI −9.0, the rest −0.5 to −3.8;
  outputs run on past the translation (EOS on 242 of 361 FLEURS-vi rows). The training text KL
  fell to 0.096 (updates 2,000–3,000) and rose to 0.136 (8,000–9,000) once the rate reached its
  peak: the trunk drifting. Voice on the gate's 10 FLEURS dev rows
  ([voice-scores](voice-scores.md)): VI→EN 0.15 / 2.3 dB, EN→VI 0.14 / 4.5 dB, against the heads
  release's 0.21 / 3.2 and 0.13 / 4.8 and the depformer student's 0.27 / 4.7 and 0.20 / 7.8.

## On the Mac

The vendored moshi_mlx reads the config (`DepFormerConfig.parallel_codebooks`; the slice count
dep_q − P + 1, `generated_codebooks` still dep_q) and the weights (the last slice's `linear_out`
is `linears.8`..`15` stacked), and samples the last slice's codebooks at once, so `convert_mlx.py`
stages a checkpoint, the gate runs it through `LmGen` and `export_lm_coreml.py` exports the staging.
Checked on tiny students with larger random weights (2026-11-27): MLX's greedy `DepFormer.sample`
equals PyTorch's `depformer_step` on every codebook (4 of them; and 16, the last 8 at once, its
logits within 1.9e-6), `LmGen` steps, `convert_mlx.py` staged the 4-codebook one, and `lm_step`'s
slices on a staging match PyTorch's logits on all 16 codebooks. Not yet run: the gate and the
export on a trained checkpoint.

## Not yet

- End-to-end Core ML export and iPhone 13 Pro Max gate on a trained slices checkpoint (palettization
  in `training/student/palettize.py` and `train.py --palettize-bits` now support `depformer_parallel`
  slices models).
