# Student GRPO

Read when touching the phone student's GRPO run (`grpo_student`), its evaluations and tests, or the
model it ships (update 2,000).
Back to [CONTEXT.md](../../CONTEXT.md) · [Run](#run) · [Evaluations](#evaluations) ·
[Tests](#tests) · [Raw outputs](#raw-outputs)

The student, its cut and its distillation are in [phone-student](phone-student.md). GRPO itself
is in [grpo](grpo.md).

## Run

`grpo_student` (2026-11-15, commit 01a0d13): `grpo_launch.sh grpo_student
weights/student_dep10_distilled.safetensors --config-path weights/student_dep10/config.json
--text-weight 50 --gen-frames 1228800 --micro-batch 4 --lr 1e-6 --updates 4000`, rooms on
([noisy-source](noisy-source.md#rooms)).

- **Rate 1e-6**, 5× the 3B's. At 2e-7 the end-to-end test's student clipped ~0.01% of its tokens
  an update (largest ratio deviation 0.77), where the 3B clipped 0.5%. At 1e-6 it clips 0.03–0.08%
  up to update 2,200, with the largest deviation mostly under 5.
- **Sizes** (`grpo_profile`, the H100): generation is flat above 512 rows (a step costs 4.6 ms plus
  0.027 ms a row), and micro-batch 4 is the fastest. An update takes ~7.1 s with saves; the rooms
  are built in 15–17 s before the policy loads.
- **Stopped** by TERM after update 2,400's validation (2026-11-15), shipping 2,000 (user, "ok
  shall we stop now?"). The final sync uploaded the pair 2,400 and the logs.
  - Update 2,000 is `runs/grpo_student/best_step002000.safetensors`, linked on the box as
    `weights/student_grpo2000.safetensors`.
  - On the Mac it is `weights/student_dep10/g2000_bf16` and `g2000_q6` (6 bits, 1.24 GB, with the
    8-codebook `mimi-decoder.mlmodelc`).
  - `g2000_q6` with a short card is the Hub's `release/student_grpo2000_mlx_q6/` (1.72 GB; user,
    2026-11-15: "yep go ahead"), the model the iOS app downloads. On the iPhone 13 Pro Max it ran
    `vi2.wav` at 61 ms an 80 ms frame (p90 63–64 ms, two runs), footprint 1.8 GB, against the 3B's
    148 ms and 3.5 GB ([swift-runtime](swift-runtime.md#device-test)).

## Evaluations

Every 200 updates, the 8 validation sets, free-running. The table gives FLEURS dev VI→EN / EN→VI:
chrF, BLEU, median End Offset, mean text LAAL and mean length ratio. Selecting is the mean chrF
over FLEURS dev both ways; "Of 8" is the mean over all 8 sets. Every set passes health and has no
empty rows, except where noted.

| Update | chrF | BLEU | End Offset (s) | LAAL (s) | Length ratio | Selecting | Of 8 | Promoted |
|---|---|---|---|---|---|---|---|---|
| 0 | 32.10 / 29.16 | 4.89 / 10.08 | 2.34 / 1.90 | 2.52 / 2.78 | 1.50 / 1.08 | 30.63 | 34.69 | |
| 200 | 31.82 / 31.85 | 5.28 / 10.82 | 1.98 / 1.40 | 2.28 / 2.13 | 1.45 / 1.16 | 31.83 | 35.05 | yes |
| 400 | 32.79 / 32.19 | 5.78 / 11.91 | 1.57 / 1.39 | 2.24 / 2.25 | 1.43 / 1.10 | 32.49 | 35.92 | yes |
| 600 | 32.59 / 33.37 | 5.38 / 12.62 | 1.25 / 1.10 | 2.05 / 2.03 | 1.39 / 1.10 | 32.98 | 36.12 | yes |
| 800 | 32.82 / 33.80 | 5.65 / 13.10 | 0.91 / 1.04 | 2.14 / 2.01 | 1.35 / 1.10 | 33.31 | 36.57 | yes |
| 1,000 | 34.33 / 33.50 | 6.69 / 12.78 | 0.88 / 1.21 | 2.13 / 2.11 | 1.34 / 1.08 | 33.91 | 37.01 | yes |
| 1,200 | 34.17 / 34.16 | 6.61 / 13.36 | 1.10 / 1.31 | 2.23 / 2.14 | 1.32 / 1.07 | 34.17 | 37.52 | yes |
| 1,400 | 34.10 / 34.25 | 6.54 / 13.78 | 0.91 / 1.23 | 2.21 / 2.13 | 1.29 / 1.07 | 34.17 | 37.46 | yes |
| 1,600 | 35.52 / 34.73 | 7.31 / 14.07 | 0.72 / 0.98 | 2.09 / 2.03 | 1.30 / 1.06 | 35.12 | 37.47 | yes |
| 1,800 | 35.71 / 34.56 | 7.31 / 13.97 | 0.86 / 1.07 | 2.22 / 2.05 | 1.29 / 1.07 | 35.13 | 37.67 | yes |
| **2,000** | **36.51 / 34.49** | 7.41 / 13.90 | 0.93 / 1.20 | 2.28 / 2.15 | 1.26 / 1.06 | **35.50** | **38.12** | yes |
| 2,200 | 34.84 / 34.57 | 6.59 / 13.67 | 1.04 / 1.07 | 2.49 / 2.15 | 1.45 / 1.05 | 34.70 | 37.24 | |
| 2,400 | 23.71 / 23.00 | 3.22 / 6.98 | 1.04 / 1.22 | 3.05 / 2.53 | 3.00 / 1.82 | 23.35 | 26.54 | |

Notes on the table:
- Update 400 had 1 empty EN→VI row.
- At 2,400, 6 of the 8 sets fail the length-ratio gate.

- **It drifts after ~2,000 at 1e-6 and collapses by over-generation.** The 40-update means from
  the training log:

  | Updates to | Clip fraction | Gradient norm | Rollout BLEU |
  |---|---|---|---|
  | 2,040 | 0.06% | 82k | 36.9 |
  | 2,200 | 0.15% | 93k | 36.1 |
  | 2,320 | 0.86% | 106k | 35.0 |
  | 2,400 | 2.56% | 119k | 33.6 |

  The largest ratio deviation reached 12–28 from 2,280.
- **The EOS rate falls** over the run, from 0.80 in the first 200 updates to 0.72 in the last,
  because no reward pays for an EOS ([grpo](grpo.md)).
- **What helps:** GRPO cut FLEURS' End Offset from 2.3 / 1.9 s to 0.9 / 1.2 s, and its VI→EN
  length ratio from 1.50 to 1.26.

## Tests

Update 2,000 against the teacher (GRPO 1,400) on the test sets: seed 43, batch 64, text only.
- The student ran `/workspace/evals/test_student.sh` (2026-11-15).
- The teacher ran `test_twoway.sh` (2026-11-14), before LAAL was recorded. Its FLEURS LAAL is from
  `laal_grpo1400` (seed 43, batch 16).
- Every set passes health for both models.
- Directions: the `-vi` sets and `giga-heldout`, `multimed-vi` and `en-vi-val` (Vietnamese speech)
  are VI→EN; the `-en` sets are EN→VI.

The student's columns come first: chrF, BLEU, median End Offset, mean LAAL, length ratio and empty
rows. The teacher's chrF, BLEU, End Offset and LAAL follow.

| Set | chrF | BLEU | EO (s) | LAAL (s) | Length | Empty | Teacher chrF | BLEU | EO (s) | LAAL (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| fleurs-vi/test | 35.65 | 7.66 | 0.88 | 2.32 | 1.25 | 0/857 | 41.07 | 11.33 | 0.91 | 2.18 |
| fleurs-en/test | 34.53 | 13.77 | 1.18 | 2.12 | 1.07 | 0/641 | 38.23 | 17.56 | 1.42 | 2.52 |
| multimed-vi/select | 34.91 | 9.43 | 2.22 | 2.40 | 1.25 | 0/1718 | 40.23 | 12.90 | 2.38 | |
| multimed-vi/final | 35.18 | 9.79 | 2.24 | 2.43 | 1.25 | 1/1719 | 40.46 | 13.41 | 2.38 | |
| giga-heldout/test | 62.45 | 36.45 | 1.91 | 1.84 | 1.09 | 1/1000 | 68.49 | 43.62 | 2.19 | |
| en-vi-val/cv27 | 38.72 | 13.66 | 1.75 | 2.26 | 1.18 | 0/180 | 48.13 | 19.18 | 2.00 | |
| en-vi-val/libriheavy | 42.91 | 17.03 | 1.91 | 2.59 | 1.04 | 0/180 | 48.39 | 22.48 | 2.08 | |
| en-vi-val-noisy/cv27 | 36.21 | 10.72 | 1.98 | 2.42 | 1.20 | 0/180 | 45.64 | 16.70 | 2.16 | |
| en-vi-val-noisy/libriheavy | 39.92 | 14.51 | 1.83 | 2.56 | 1.06 | 0/180 | 44.87 | 18.10 | 2.11 | |
| en-vi-val-en/cv27 | 32.28 | 14.32 | 0.64 | 1.80 | 1.15 | 0/180 | 39.09 | 19.65 | 0.86 | |
| en-vi-val-en/libriheavy | 41.50 | 19.69 | 1.61 | 2.20 | 1.06 | 0/180 | 45.64 | 24.16 | 1.68 | |
| onevoice-vi/test | 33.99 | 11.76 | 2.16 | 2.58 | 1.28 | 0/100 | | | | |
| onevoice-en/test | 28.65 | 8.06 | 2.12 | 2.54 | 1.12 | 0/100 | | | | |

The student is 3.7–9.4 chrF below its teacher, with about the same lag.

- **Room noise** (2026-11-15, the Mac, MLX bf16, `measure_held.py`). Each run is a primer, then
  30 s of quiet with no speech, using held-out noise files (`augment.quiet_noises(held_out=True)`).
  The quiet takes 5 forms: a room at −55 and at −40 dBFS, white noise at −45, pink at −40, and
  events. Each form ran twice in each direction, 20 runs in all.
  - Update 2,000 wrote no words in any of the 20 runs.
  - The distilled student before GRPO (64,000) wrote words in 7 of 20: VI→EN 2/10 (22 tokens,
    pink), EN→VI 5/10 (71 tokens: room −55, white, pink).
  - The rooms in GRPO's streams taught it to stay quiet.

## Raw outputs

- **Box 60447659:** every run directory and every test is in
  `/Volumes/data/hibiki-zero-evals/box-60447659/`, as on the box (weights, optimizer states and
  WAVs left out).
  - `all_metrics.csv` is every validation and test value flattened into one table, one row per
    run, step, mode and set.
  - It is built by `/Volumes/data/hibiki-zero-evals/all_metrics.py FOLDER`.
- **The noise tests** (JSONL per run, and their scripts) are in
  `/Volumes/data/hibiki-zero-evals/mac/noise/`.
