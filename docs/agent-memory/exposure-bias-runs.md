# The 3B's recipe experiments: the runs

Read when judging or repeating the 3B's scheduled-sampling SFT arms or anchored GRPO, or when VI→EN
scores drop on quiet clips. Back to [CONTEXT.md](../../CONTEXT.md) · [The box](#the-box) ·
[Scheduled sampling](#scheduled-sampling) · [Their GRPO](#their-grpo) ·
[The data gate](#the-data-gate) · [The quiet clips](#the-quiet-clips) ·
[The level-stage test](#the-level-stage-test) · [Anchored GRPO](#anchored-grpo) · [Noise](#noise)

The plan, the decision rules and the method are in [exposure-bias](exposure-bias.md#experiments);
what they mean for the next round in [teacher-next](teacher-next.md#order).

## The box

On the H100 SXM 62170174 ([tranche-box](tranche-box.md#state)), 2026-11-23 14:07 to 22:33 UTC, main
at `9b3df7a`, one after another through `h100.sh` (user, 2026-11-23: "Exp 1 (cut) + exp 2 probe":
GRPO 200 updates an arm instead of 400, the rental ending). Records:
`/Volumes/data/hibiki-zero-evals/box-62170174/` (`runs/`, `logs/`, `scripts/`: the queue, the
paired bootstrap `paired.py`, the level scripts; `diag/`: [the quiet clips](#the-quiet-clips)).
No run's weights or final upload are kept on the Hub
([sft-recovery](sft-recovery.md#checkpoints-and-recovery)); `twoway_final`,
every start's origin, is. `exp1_grpo_ss01` at 200 is also in `weights/` there, the only one of
these models left.

- **SFT** (10,000 updates each, seed 61, 51,040 streams, 683 h of source): from
  `twoway_final`, `--lr 1e-6 --warmup-fraction 0.05 --schedule cosine`, the teacher-next data
  ([data](teacher-next.md#data)): the tranche's VI→EN `pho`, `vivoice`, `giga`, the noisy
  `en_speech` both ways, the tranche's noisy `en-vi`. 4,608-frame updates: 0.267 s an update for
  the control and 0.310 s for 0.1 (+16%); 57 and 62 min a run with validation, start and save.
- **GRPO** on the arms: `grpo_launch.sh`'s nine caches, 200 updates, `--gen-frames 614400
  --micro-batch 2`, seed 44, 83–85 min a run including both evaluations.
- **Anchored GRPO**: `twoway_final` as start and teacher, `--kl-coef` 0.02 and 0.1, the shipped
  run's caches and seed, 200 updates, 88 and 85 min.
- Every validation: batch 64, temperature 0.8, top-k 250, the level stage on, eight sets (ten with
  the noisy English) at seed 44 (GRPO) or 61 (SFT). "Selecting" is the mean of FLEURS dev
  VI→EN and EN→VI chrF.

## Scheduled sampling

Teacher-forced on the `vi-en` / `en-vi` validation caches, reference inputs in both arms: 0.1 reads a
hair worse (content accuracy −0.3 and −0.2 points), as its train log also does (it reads the mixed
history there).

| | Loss | Text loss | Content text loss | Content acc. | PAD acc. | Audio loss |
|---|---|---|---|---|---|---|
| control | 2.6036 / 2.3084 | 0.6695 / 0.6824 | 1.6289 / 1.5450 | 0.6724 / 0.6867 | 0.9494 / 0.9566 | 1.9341 / 1.6260 |
| 0.1 | 2.6060 / 2.3125 | 0.6712 / 0.6852 | 1.6360 / 1.5526 | 0.6697 / 0.6850 | 0.9498 / 0.9568 | 1.9349 / 1.6273 |

Train loss by deciles of updates: control 2.137 → 2.113, 0.1 2.189 → 2.155 (flat). `text_changed` 1.2%
of the content tokens throughout.

Free-running at update 10,000 (seed 61), control / 0.1. EOS counts rows ended with text EOS.
Selecting 36.19 / 35.97, mean of the ten sets 38.04 / 38.14; every set healthy.

| Set | chrF 0 / 0.1 | BLEU 0 / 0.1 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 36.40 / 36.03 | 10.57 / 10.38 | 1.10 / 1.08 | 4.26 / 4.16 | 11 / 18 | 290 / 283 / 361 | pass |
| fleurs-en/dev | 35.98 / 35.91 | 16.96 / 16.30 | 1.00 / 1.00 | 3.58 / 3.28 | 18 / 16 | 326 / 329 / 392 | pass |
| en-vi-val/cv27 | 44.28 / 43.96 | 16.91 / 18.08 | 1.09 / 1.03 | 4.23 / 3.83 | 5 / 6 | 139 / 146 / 180 | pass |
| en-vi-val/libriheavy | 44.27 / 45.16 | 19.89 / 20.51 | 1.00 / 1.01 | 5.03 / 4.87 | 2 / 1 | 143 / 145 / 180 | pass |
| en-vi-val-noisy/cv27 | 38.68 / 39.69 | 13.81 / 14.98 | 1.10 / 1.05 | 4.44 / 4.15 | 6 / 12 | 134 / 133 / 180 | pass |
| en-vi-val-noisy/libriheavy | 42.76 / 41.45 | 18.73 / 18.04 | 0.99 / 0.99 | 5.18 / 5.29 | 5 / 5 | 144 / 148 / 180 | pass |
| en-vi-val-en/cv27 | 34.29 / 33.30 | 16.43 / 16.44 | 1.09 / 1.04 | 2.21 / 2.06 | 5 / 5 | 150 / 153 / 180 | pass |
| en-vi-val-en/libriheavy | 39.56 / 39.45 | 19.08 / 19.27 | 1.03 / 1.00 | 5.11 / 4.70 | 2 / 1 | 163 / 159 / 180 | pass |
| en-vi-val-en-noisy/cv27 | 27.87 / 28.92 | 12.93 / 13.97 | 1.03 / 1.01 | 2.01 / 2.06 | 12 / 10 | 147 / 137 / 180 | pass |
| en-vi-val-en-noisy/libriheavy | 36.29 / 37.57 | 16.23 / 17.24 | 1.00 / 1.02 | 4.75 / 4.93 | 3 / 4 | 153 / 161 / 180 | pass |

## Their GRPO

200 updates from each arm's final model ([exposure-bias](exposure-bias.md#experiments)). Selecting
35.47 → 37.02 (control) and 36.01 → 37.62 (0.1); mean of ten 37.97 → 39.73 and 38.32 → 39.95; own
gain (paired, seed 43, 1,000 resamples) +1.55 [+0.58, +2.52] and +1.61 [+0.68, +2.53]. Both promoted
at 200, healthy and on time. Train log, updates 1–20 → 181–200: final BLEU 26.7 → 32.4 and
26.4 → 32.2, EOS rate 0.79 → 0.89 and 0.78 → 0.88.

Control (`exp1_grpo_ss0`):

| Set | chrF 0 → 200 | BLEU 0 → 200 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 36.21 → 36.01 | 11.02 → 9.61 | 1.09 → 1.23 | 4.08 → 2.13 | 14 → 0 | 284 → 302 / 361 | pass |
| fleurs-en/dev | 34.74 → 38.04 | 15.50 → 17.03 | 0.97 → 1.07 | 3.68 → 1.23 | 27 → 0 | 312 → 323 / 392 | pass |
| en-vi-val/cv27 | 42.58 → 45.02 | 16.36 → 17.41 | 1.12 → 1.18 | 3.67 → 2.64 | 5 → 0 | 139 → 140 / 180 | pass |
| en-vi-val/libriheavy | 44.50 → 46.55 | 19.31 → 21.34 | 1.00 → 1.03 | 5.10 → 2.91 | 3 → 0 | 147 → 138 / 180 | pass |
| en-vi-val-noisy/cv27 | 40.24 → 41.27 | 15.41 → 14.14 | 1.09 → 1.18 | 4.39 → 2.72 | 6 → 0 | 139 → 144 / 180 | pass |
| en-vi-val-noisy/libriheavy | 41.99 → 42.67 | 18.35 → 17.19 | 0.99 → 1.06 | 4.51 → 3.21 | 3 → 0 | 143 → 128 / 180 | pass |
| en-vi-val-en/cv27 | 34.41 → 37.56 | 17.43 → 17.95 | 1.10 → 1.15 | 2.15 → 1.04 | 2 → 1 | 162 → 152 / 180 | pass |
| en-vi-val-en/libriheavy | 38.66 → 41.92 | 18.02 → 20.23 | 0.99 → 1.07 | 5.30 → 1.87 | 8 → 0 | 151 → 159 / 180 | pass |
| en-vi-val-en-noisy/cv27 | 29.74 → 29.46 | 12.79 → 11.59 | 1.09 → 1.15 | 2.34 → 1.01 | 1 → 1 | 151 → 158 / 180 | pass |
| en-vi-val-en-noisy/libriheavy | 36.60 → 38.79 | 16.86 → 17.29 | 1.00 → 1.09 | 5.31 → 1.98 | 4 → 0 | 155 → 152 / 180 | pass |

0.1 (`exp1_grpo_ss01`):

| Set | chrF 0 → 200 | BLEU 0 → 200 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 36.79 → 36.98 | 10.95 → 9.52 | 1.08 → 1.23 | 4.05 → 2.14 | 13 → 0 | 267 → 314 / 361 | pass |
| fleurs-en/dev | 35.24 → 38.26 | 15.99 → 17.03 | 0.97 → 1.08 | 3.71 → 1.35 | 22 → 0 | 316 → 329 / 392 | pass |
| en-vi-val/cv27 | 44.51 → 45.51 | 16.68 → 17.66 | 1.11 → 1.18 | 4.04 → 2.54 | 2 → 0 | 137 → 140 / 180 | pass |
| en-vi-val/libriheavy | 45.31 → 46.97 | 20.73 → 21.02 | 1.00 → 1.05 | 4.71 → 2.86 | 2 → 0 | 146 → 140 / 180 | pass |
| en-vi-val-noisy/cv27 | 39.69 → 41.80 | 13.85 → 13.88 | 1.13 → 1.22 | 3.98 → 2.59 | 7 → 0 | 138 → 155 / 180 | pass |
| en-vi-val-noisy/libriheavy | 40.80 → 42.79 | 17.70 → 18.03 | 0.99 → 1.04 | 4.78 → 2.92 | 3 → 0 | 136 → 138 / 180 | pass |
| en-vi-val-en/cv27 | 34.59 → 36.68 | 17.61 → 16.20 | 1.07 → 1.18 | 1.94 → 0.83 | 5 → 0 | 154 → 152 / 180 | pass |
| en-vi-val-en/libriheavy | 39.21 → 42.00 | 19.30 → 19.38 | 0.98 → 1.09 | 4.83 → 2.43 | 8 → 0 | 156 → 158 / 180 | pass |
| en-vi-val-en-noisy/cv27 | 29.93 → 30.56 | 13.06 → 12.63 | 1.07 → 1.16 | 2.66 → 0.98 | 8 → 2 | 147 → 162 / 180 | pass |
| en-vi-val-en-noisy/libriheavy | 37.15 → 37.98 | 17.28 → 17.04 | 1.01 → 1.07 | 4.47 → 2.27 | 5 → 0 | 158 → 153 / 180 | pass |

**0.1 against the control** (paired bootstrap, 0.1 − control): selecting −0.23 [−1.06, +0.68] at the
SFT's end, +0.54 [−0.36, +1.47] at GRPO's start, +0.59 [−0.20, +1.42] at 200; the mean of the ten
sets +0.11 [−0.41, +0.64] and, at 200, +0.22 [−0.23, +0.72]. A tie: the plan's "wins all three"
is not met, so 0.1 was **not adopted**, its speech check and a 0.25 arm not run.

## The data gate

The control is also the round's first look at its data. Its SFT against `twoway_final` on today's
code, the eight sets, paired. The control is scored twice, at seed 44 (as GRPO's update 0) and at
seed 61 (its own evaluation); `twoway_final` at seed 44 both times:

| | Seed 44 | Seed 61 |
|---|---|---|
| FLEURS-vi | **−3.85 [−5.54, −2.30]** | **−3.66 [−5.21, −1.92]** |
| FLEURS-en | +0.83 [−0.67, +2.34] | +2.08 [+0.85, +3.34] |
| `en-vi-val-en` cv27 / Libriheavy | +0.77 / −1.22 | +0.65 / −0.32 |
| `en-vi-val` cv27 / Libriheavy | −2.19 / +0.09 | −0.49 / −0.14 |
| `en-vi-val-noisy` cv27 / Libriheavy | +0.36 / −0.77 | −1.20 / −0.01 |
| selecting mean | −1.51 [−2.59, −0.40] | −0.79 [−1.83, +0.29] |
| mean of eight | −0.75 [−1.43, −0.08] | −0.39 [−1.10, +0.34] |

- **The gate the plan set, VI→EN FLEURS dev within 1 chrF, fails** at both seeds. It is the one
  set whose interval excludes zero at both; the selecting mean and the mean of eight do at seed 44
  only, and of EN→VI's sets only FLEURS-en does, at seed 61.
- At 200 GRPO updates against the shipped run's 200 (older code): FLEURS-vi −4.60 [−6.30, −2.99],
  FLEURS-en +2.64 [+1.50, +3.77], `en-vi-val-en` cv27 +3.30 [+1.13, +5.47], mean of eight +0.09
  [−0.54, +0.73], selecting −0.98 [−1.93, −0.00]. So GRPO doesn't recover FLEURS-vi, while EN→VI ends
  ahead of the shipped run's.
- `twoway_final` itself, today's code against the shipped run's update 0 (older code): selecting 36.98
  (FLEURS-vi 40.06, FLEURS-en 33.91) against 37.45, −0.46 [−1.03, +0.09].

## The quiet clips

**Where the loss is.** FLEURS-vi dev clips by raw RMS level (dBFS), quartiles of 90–91 rows: change
in sentence chrF from `twoway_final`:

| Model | < −48.5 | −48.5 to −32.5 | −32.5 to −30.7 | > −30.7 |
|---|---:|---:|---:|---:|
| control SFT (11 empty rows against 0) | −8.2 | −8.1 | −0.1 | −0.9 |
| control GRPO 200 | −9.6 | −7.1 | −0.6 | −0.5 |
| 0.1 SFT (18 empty rows) | −10.0 | −7.1 | −1.9 | −1.3 |
| 0.1 GRPO 200 | −9.1 | −5.2 | +0.1 | −0.4 |
| anchored 0.02 at 200 (from `twoway_final`) | +1.0 | +2.6 | +1.0 | +2.3 |
| anchored 0.1 at 200 | +0.7 | +1.9 | +0.6 | +0.3 |

So the new SFT loses 5–10 chrF on quiet Vietnamese clips (RMS under −32.5 dBFS, half of FLEURS-vi)
and 0–2 on the loud half. The same split in four models, found after the fact.

**What the input does to it** (2026-11-23; follow-ups outside the plan, run while the box was idle).
The 180 quiet rows re-levelled four ways (and the loud clips lowered two), `twoway_final` against
the only one of these models left, the 0.1 arm after GRPO; `eval.py` seed 44, sentence chrF, 95%
intervals from 1,000 resamples of the rows. "Stage" = the level stage on, as every evaluation runs; "bypass" = off:

| Input | `twoway_final` | 0.1 arm, GRPO 200 | Difference |
|---|---:|---:|---|
| as heard (stage) | 40.7 | 33.5 | **−7.2 [−9.8, −4.6]** |
| one flat gain to a −6 dBFS peak, stage on | 41.2 | 38.9 | −2.4 [−4.5, −0.0] |
| the stage's output, then one flat gain to −6 (bypass) | 40.9 | 40.1 | **−0.7 [−2.6, +1.4]** |
| flat to −6 (bypass) | 41.8 | 36.5 | −5.4 [−7.8, −3.0] |
| flat to −20 (bypass; the stage's target) | 40.3 | 29.8 | −10.5 [−13.4, −7.8] |

- `twoway_final` reads every variant alike (40.3–41.8). **The other model depends on the input's
  level:** the same stage output, moved from about −20 to −6 dBFS, reads +6.6 [+4.6, +8.8] chrF
  higher (33.5 → 40.1), and the flat clips fall 6.7 from −6 to −20.
- **That is part of the cause, not all of it.** At a −6 dBFS peak without the stage the gap is still
  −5.4; with the stage's gain changes at that peak it reads 3.6 higher (36.5 → 40.1), so the clips'
  dynamics matter too (presumably as the stage raises their quiet parts). The loud clips
  lowered 26 dB (float, or 16-bit steps) and heard through the stage lose 2.6 [0.5, 4.6] in that
  model (1.1 [−0.7, 3.1] in `twoway_final`), far less than the quiet ones' 7.2, though both reach
  Mimi near the stage's target. And FLEURS-en's clips are quieter still (median −59.5 dBFS), yet
  its EN→VI score rose. What else the quiet Vietnamese recordings share is not known.
- Why an SFT at 1e-6 over 10,000 updates of this data loses its level robustness is **not
  established**. The training sources are loud and never see the stage
  ([sft-evaluation](sft-evaluation.md#input-level)); noisy copies use `augment.GAIN_DB`, −15 to 0 dB.
- **Two candidate fixes:** deeper level augmentation in the next SFT's data, so the model stays
  level-robust (not run); or the stage's target raised from −20 toward −6 dBFS, which costs
  `twoway_final` nothing on these rows (+0.2 [−1.7, +2.0]) but lifts a quiet room's noise with it:
  measured on every set and noise alone, and adopted ([the level-stage test](#the-level-stage-test)).
- Caveats: one model (0.1 after GRPO; the control's weights are gone, so it is untested), one set
  (FLEURS-vi dev), one seed, row-sampling intervals only.

## The level-stage test

The input level stage as it was (target −20 dBFS, noise ceiling −40, gain cap 30 dB) and with
all three 10 and 14 dB louder (`eval_stage.py` scales `level.py`'s constants), 2026-11-25 on the
RTX 5090 box 62513802 ([tranche-box](tranche-box.md#state)), main at `4aa3177`. `eval.py` seed
43, batch 64, text only, one call a model and setting: the 11 test sets, FLEURS dev both ways,
the noisy English sets and noise alone (24 rooms from the bank's held-out files by
`augment.room_tracks`' recipe, seed 44, and the 5 synthetic noises of
[sft-evaluation](sft-evaluation.md#input-level), 30 s each, fed as either direction's source with
empty references). 15 min a 3B pass, 5 min the student's. Paired bootstrap over rows, seed 43,
1,000 draws. Records: `/Volumes/data/hibiki-zero-evals/box-62513802/level/` (`records.tgz`:
predictions, metrics, logs, scripts, the noise clips; `tables.txt`: every set and pair).

| Model | Mean of 11: as it was / +10 / +14 dB | +14 − as it was | Quiet FLEURS-vi dev | Noise rows with text (of 58), words |
|---|---|---|---|---|
| update 1,400 (shipped teacher) | 45.45 / 45.54 / 45.34 | −0.11 [−0.49, +0.24] | 42.46 / 42.85 / 43.02 | 44 / 42 / 41; 599 / 709 / 606 |
| `twoway_final` | 41.34 / 41.42 / 41.65 | +0.31 [−0.12, +0.73] | 39.12 / 40.87 / 41.42 | 42 / 32 / 31; 603 / 353 / 376 |
| 0.1 arm, GRPO 200 (`exp1_grpo_ss01`) | 42.71 / 42.88 / 43.12 | +0.41 [+0.04, +0.79] | 33.02 / 37.65 / 39.43 | 31 / 28 / 27; 370 / 409 / 302 |
| phone student, 6 bits ([update 4,000 snapped](student-palettize.md#tests)) | 42.30 / 42.42 / 42.46 | +0.16 [−0.18, +0.51] | 40.63 / 40.59 / 40.91 | 7 / 7 / 9; 80 / 63 / 88 |

Quiet FLEURS-vi dev: its 180 rows under −32.5 dBFS RMS, as [the quiet clips](#the-quiet-clips).
Every scored set healthy in every pass (the noise rows have no reference).

- **Most of the data gate's loss is the stage's level** (on the 0.1 arm after GRPO, the gate's
  only model left; the control SFT is gone). It against `twoway_final`, both as it was → both at
  +14 dB: FLEURS-vi dev −2.76 [−4.42, −1.09] → −0.94 [−2.35, +0.33], its quiet half −6.11 →
  −1.99 [−4.03, −0.15], FLEURS-vi test −3.21 → −0.83 [−1.72, +0.02], the mean of 11 +1.37 → +1.47.
  So a quiet-clip gap of ~2 remains, as [the quiet clips](#the-quiet-clips) found (part of the
  cause, not all). +10 dB gets about half (quiet half +4.63, still −3.23 behind).
- **A trade between directions.** VI→EN FLEURS test rose in all four models (+0.43 to +3.65;
  past its interval in all but the student) and EN→VI FLEURS, the quietest set (median RMS −59.5
  dBFS), fell in all eight model-and-split cells (test −0.04 to −0.78, dev −0.07 to −1.56, ~0.75
  on average). Losses past their intervals at +14: update 1,400 FLEURS-en test −0.78
  [−1.53, −0.02] and dev −1.09 [−2.11, −0.12], `twoway_final` FLEURS-en dev −1.56 [−2.77, −0.43],
  the 0.1 arm `en-vi-val/libriheavy` −1.46 [−2.63, −0.40], the student `giga-heldout` −1.03
  [−1.71, −0.36]; run-to-run noise reaches ~1.3 a set ([Noise](#noise)).
- **Noise alone:** the 3B models write fluent invented sentences on most held-out rooms at every
  setting ("Thank you for joining our New Day Life") and on 1–4 of the 10 synthetic rows; at +14
  on fewer rows than as it was (44 → 41, 42 → 31, 31 → 27). The student writes on 5–8 of 48 rooms
  (7 → 9 rows, 80 → 88 words), and it runs without a voice gate on the phone.
  The Mac check that found update 1,400 silent on noise alone used other rooms and a primed stream
  ([noisy-source](noisy-source.md#rooms)); these rows start cold.
- The student as it was reproduces its tests (mean of 11 42.30 against 42.48).
- **Adopted** (2026-11-26): the +14 dB stage in `level.py` and Swift's `Audio.Leveller`
  ([sft-evaluation](sft-evaluation.md#input-level)).

## Anchored GRPO

`exp2_anchor002` (β 0.02) and `exp2_anchor01` (β 0.1), from `twoway_final` with itself as the
teacher, against the shipped run at 200 (older code, plain GRPO, selecting 38.00):

| | Start | 200 | Own gain | vs the shipped run's 200 |
|---|---|---|---|---|
| β 0.02, selecting | 36.98 | **39.12** | +2.14 [+1.33, +2.94] | +1.12 [+0.44, +1.88] |
| β 0.1, selecting | 37.00 | 38.31 | +1.31 [+0.43, +2.09] | +0.31 [−0.46, +1.13] |
| β 0.02, mean of 8 | 39.91 | 41.66 | +1.75 [+1.16, +2.35] | +0.62 [+0.04, +1.21] |
| β 0.1, mean of 8 | 39.57 | 41.67 | +2.09 [+1.49, +2.67] | +0.63 [+0.07, +1.19] |

`exp2_anchor002`:

| Set | chrF 0 → 200 | BLEU 0 → 200 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 40.06 → 41.70 | 12.04 → 11.94 | 1.12 → 1.18 | 2.74 → 1.66 | 0 → 0 | 288 → 285 / 361 | pass |
| fleurs-en/dev | 33.91 → 36.53 | 15.15 → 16.43 | 1.02 → 1.06 | 3.54 → 2.00 | 4 → 1 | 344 → 344 / 392 | pass |
| en-vi-val/cv27 | 44.77 → 44.77 | 17.90 → 16.46 | 1.09 → 1.14 | 3.52 → 3.04 | 3 → 1 | 126 → 134 / 180 | pass |
| en-vi-val/libriheavy | 44.41 → 45.89 | 18.86 → 19.72 | 1.02 → 1.05 | 4.33 → 3.47 | 0 → 0 | 150 → 131 / 180 | pass |
| en-vi-val-noisy/cv27 | 39.88 → 42.34 | 13.18 → 14.25 | 1.20 → 1.18 | 3.46 → 2.55 | 4 → 0 | 139 → 130 / 180 | pass |
| en-vi-val-noisy/libriheavy | 42.77 → 44.86 | 16.63 → 18.55 | 1.06 → 1.08 | 4.40 → 3.62 | 0 → 0 | 150 → 139 / 180 | pass |
| en-vi-val-en/cv27 | 33.64 → 35.07 | 15.92 → 16.54 | 1.12 → 1.16 | 1.33 → 0.68 | 3 → 0 | 159 → 136 / 180 | pass |
| en-vi-val-en/libriheavy | 39.88 → 42.13 | 18.94 → 20.40 | 1.04 → 1.08 | 3.85 → 2.43 | 0 → 1 | 158 → 149 / 180 | pass |

`exp2_anchor01`:

| Set | chrF 0 → 200 | BLEU 0 → 200 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 40.46 → 41.03 | 12.55 → 11.22 | 1.13 → 1.22 | 2.75 → 1.98 | 0 → 0 | 284 → 303 / 361 | pass |
| fleurs-en/dev | 33.55 → 35.60 | 15.16 → 16.22 | 1.01 → 1.04 | 3.44 → 2.41 | 5 → 0 | 338 → 335 / 392 | pass |
| en-vi-val/cv27 | 44.25 → 46.12 | 16.89 → 17.39 | 1.10 → 1.15 | 3.67 → 3.04 | 2 → 0 | 137 → 128 / 180 | pass |
| en-vi-val/libriheavy | 44.33 → 46.23 | 19.27 → 20.39 | 1.02 → 1.06 | 4.20 → 3.69 | 0 → 0 | 153 → 145 / 180 | pass |
| en-vi-val-noisy/cv27 | 38.59 → 43.34 | 12.04 → 15.59 | 1.21 → 1.20 | 3.43 → 2.67 | 3 → 0 | 141 → 136 / 180 | pass |
| en-vi-val-noisy/libriheavy | 42.53 → 43.69 | 16.85 → 18.56 | 1.06 → 1.06 | 5.08 → 3.55 | 0 → 0 | 150 → 130 / 180 | pass |
| en-vi-val-en/cv27 | 33.14 → 35.38 | 15.91 → 16.29 | 1.11 → 1.16 | 1.35 → 0.91 | 2 → 1 | 153 → 143 / 180 | pass |
| en-vi-val-en/libriheavy | 39.74 → 41.94 | 18.74 → 20.09 | 1.04 → 1.08 | 4.27 → 2.77 | 0 → 0 | 157 → 157 / 180 | pass |

- β 0.1 against 0.02 at 200: selecting −0.81 [−1.60, −0.12], mean of 8 0.00. At 200 both are ahead
  of the shipped run (0.1's selecting within noise); against its update 400 (selecting 38.89, mean
  of 8 41.80) 0.02 is +0.23 [−0.46, +0.97]
  and 0.1 −0.58 [−1.35, +0.26], mean of 8 −0.14 for both.
- The KL stays small: `kl_text`/`kl_audio` per token 0.0003/0.0010 over updates 1–20 (the profile's
  floor: 0.0002/0.0009), 0.0023/0.0027 at 91–110 and 0.0049/0.0048 at 181–200 for β 0.02
  (0.0018/0.0027 and 0.0037/0.0041 for 0.1: the stronger β holds it ~15–25% lower). The log doesn't
  split the loss or the gradient by term (total gradient norm 4.8–5.3e5 in every arm, anchored or not),
  so the KL's share of the update is **unmeasured**.
- The EOS share **fell**: rows ended with text EOS over the eight sets, 82.6% → 79.0% (0.02) and
  82.5% → 80.6% (0.1), against the shipped run's 83.1% → 81.9% and the unanchored arms' 80.6% →
  81.1% and 79.1% → 83.3% (a share's standard error is ~0.9 points). So the anchor's motive, holding
  the EOS share, shows nothing in 200 updates; the shipped run's fall (0.83 → 0.68) came by 1,400.
- End offsets fall on every set (FLEURS VI→EN 2.74 → 1.66 s for 0.02).
- **What's missing:** a plain GRPO from `twoway_final` on today's code, so the anchor's share of
  the +1.12 is not isolated from the code's own drift. It wants ~85 min of an H100.

Peak memory with the anchor: 74.9 GiB at micro-batch 2 (gen batch 512: 72.2 GiB); the profile
reports micro-batch 1 out of memory, after 2 in the same process (not checked alone). 8.8–11.1 s an
update step (`sec_per_update`), ~22 s with generation.

## Noise

- Row sampling only is covered by the bootstrap (seed 43, 1,000 resamples). The generation isn't
  reproducible: `twoway_final`, seed 44, twice (the starts of the two anchored runs): selecting +0.02
  [−0.58, +0.63], mean of 8 −0.34 [−0.74, +0.06], one set up to 1.3 either way. A seed change (61 to 44)
  moved the control arm −0.72 [−1.66, +0.24] and the 0.1 arm +0.05 on selecting.
- One training seed per arm: read a difference under ~0.7 on a mean, or ~1.3 on a set, as noise.
