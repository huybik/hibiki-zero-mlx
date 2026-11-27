# The slices student's on-policy stage and phone finish

Read when running or judging the slices student's on-policy run (`both_slices_z`), its 6-bit
palettization and its export to the phone. Back to [CONTEXT.md](../../CONTEXT.md) ·
[Profile](#profile) · [The run](#the-run) · [Refit](#refit) · [Phone finish](#phone-finish). The student and its
off-policy stages are in [student-slices](student-slices.md); the method in
[on-policy-distillation](on-policy-distillation.md), GRPO in [grpo](grpo.md).

## Why on-policy

Training the trunk off-policy on the SFT data lost text on every set (the joint stage,
[student-slices](student-slices.md#runs)): the trunk's text came from `both_heads_z`'s on-policy
run. So the trunk changes only on-policy, with `both_heads_z`'s recipe (both terms, z-scored, the
3B teacher, 1e-6), which also distils the audio codebooks on the student's own rollouts.

## Profile

`grpo_profile` on box 62513802 (RTX 5090, 2026-11-27 16:42 UTC) from Stage 1's decay at 8,000
(`weights/student_slices14_decay8000.safetensors`, hard-linked), `both_heads_z`'s data, the
teacher, `--park-states`; raw report `runs/grpo_profile_slices/profile.json`, copied to
`/Volumes/data/hibiki-zero-evals/box-62513802/grpo_profile_slices/`:

- Generation: 192 / 320 / 384 rows at 14.5 / 20.2 / 23.2 ms a step, peaks 15.1 / 23.4 / 27.6 GiB;
  a step costs 5.7 ms + 0.045 ms a row. A window's generation is 6.72 s an update at 320 rows.
- π_old's pass and the update at micro-batch 1 / 2 / 4: 1.36 / 1.26 / 1.18 s and 8.14 / 7.72 /
  7.35 s an update, peaks 20.9 / 23.9 / 28.0 GiB. First update's ratio deviation 0 at each.
- Start: `kl_text` 0.13–0.14, `kl_audio` 0.76–0.90 (the heads student's 1.3–1.4), final BLEU of the
  profile's rollouts 38.1.
- Kernels (`torch.profiler` over the profile at 320 rows and micro-batch 2, `prof_grpo.py` beside
  the records): the narrowed grouped-KV step attention (`fmha_cutlassF`, a boolean mask) is 49% of
  the CUDA time, ~0.8 ms a layer a step, near the card's bandwidth for 320 rows' KV cache; the
  update's GEMMs (bf16 under fp32 master weights) most of the rest.
- Levers left out: FP8 in the update (train.py's `--fp8`, `--fp8-teacher`) would save ~12% of an
  update at the cost of the teacher's targets (`all` flips 12–17% of acoustic argmaxes) and the
  policy's log-probabilities; `--bf16-adamw` cannot hold a 1e-6 step in bf16 weights; 384 rows
  and micro-batch 4 save ~1% and ~2% but peak within 4 GiB of 32 (the heads run held more than its
  profile). Sizes chosen: `--gen-frames 384000 --micro-batch 2 --park-states`, ~16.5 s an update.

## The run

`both_slices_z`: started 2026-11-27 17:00 UTC on box 62513802, commit 7a71224, HF recovery prefix
`both_slices_z`, through

```bash
T=/workspace/tranche/repo/cache HF_HUB_DISABLE_XET=1 ./training/tranche/box/grpo_launch.sh both_slices_z \
  weights/student_slices14_decay8000.safetensors \
  --cache-dir $T/pho $T/vivoice $T/giga $T/en-vi/pho $T/en-vi/vivoice $T/en-vi/giga \
    data/cache/en-vi/en-vi-grounded-v2/train \
  --config-path weights/student_slices14/config.json --save-every 100 --keep-checkpoints 1 \
  --teacher-weight weights/grpo_twoway_best1400.safetensors --lr 1e-6 \
  --gen-frames 384000 --micro-batch 2 --park-states --seed 45
```

(`both_heads_z`'s caches, which won over `both_heads_z2`'s noisy EN→VI copies; the launcher's
`--pair-cache-dir` times the `en_speech` rows; seed 45 for streams the trunk's run did not see;
8,979,646 rows, 1,713,710 streams; 2,000 updates, validated every 200 on the 10 default sets, 2,193
rows, stopped after two validations without a new best). ~16 s an update with its window's
generation, 176 s a validation; `ratio` ≤ 1.0006, grad norm 1.2–2.0e5 over updates 1–100.

Validations (chrF / BLEU / length ratio / end offset in s / speaker_sim / HNR dB; selecting =
FLEURS dev both ways; raw outputs in `/Volumes/data/hibiki-zero-evals/box-62513802/both_slices_z/`):

| Update | Selecting | fleurs-vi/dev | fleurs-en/dev | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-noisy cv27 | en-vi-val-noisy libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy | en-vi-val-en-noisy cv27 | en-vi-val-en-noisy libriheavy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 37.94 | 40.62 / 9.50 / 1.36 / 1.3 / 0.34 / 4.5 | 35.26 / 16.21 / 1.01 / 1.9 / 0.16 / 5.5 | 44.67 / 15.80 / 1.20 / 2.1 / 0.31 / 4.1 | 45.74 / 18.60 / 1.12 / 2.3 / 0.41 / 4.3 | 42.33 / 14.38 / 1.24 / 2.3 / 0.25 / 4.0 | 44.01 / 16.51 / 1.14 / 2.3 / 0.34 / 4.3 | 35.78 / 16.90 / 1.14 / 1.0 / 0.27 / 4.9 | 43.94 / 22.46 / 1.07 / 1.9 / 0.29 / 5.2 | 23.82 / 10.44 / 0.99 / 2.0 / 0.25 / 4.4 | 36.04 / 17.51 / 1.01 / 2.8 / 0.27 / 4.8 |
| 200 | 38.74 | 41.58 / 10.68 / 1.31 / 1.6 / 0.33 / 3.6 | 35.90 / 16.52 / 1.01 / 2.1 / 0.15 / 4.6 | 45.83 / 18.10 / 1.15 / 2.4 / 0.33 / 3.4 | 46.06 / 20.19 / 1.05 / 2.2 / 0.43 / 3.4 | 43.34 / 15.39 / 1.24 / 2.7 / 0.26 / 3.3 | 43.95 / 17.46 / 1.08 / 2.7 / 0.35 / 3.3 | 36.80 / 18.32 / 1.13 / 0.9 / 0.26 / 4.1 | 44.12 / 22.27 / 1.06 / 1.9 / 0.28 / 4.1 | 24.67 / 10.37 / 1.01 / 2.1 / 0.24 / 3.7 | 35.53 / 16.92 / 1.01 / 2.5 / 0.26 / 3.7 |

At 200 the text rose (mean chrF of 10 39.22 → 39.78, selecting +0.80) and speaker_sim held
(0.290 → 0.289), but HNR fell on every set (mean 4.59 → 3.73 dB; the teacher's 6.50 on the
seven SFT sets): the GRPO term's BLEU advantages also move the audio tokens. Stopped by TERM after
200's save (2026-11-27 18:15 UTC), its 200 pair and best on the box and the Hub, to compare
distillation alone on the same streams (`opd_slices`, below).

`opd_slices`: the same launch with `--reward-coef 0` (on-policy distillation alone, no π_old pass),
2026-11-27 18:15–19:25 UTC, stopped by TERM after its validation at 200 (raw outputs in
`/Volumes/data/hibiki-zero-evals/box-62513802/opd_slices/`). Distillation alone lost on every
count: selecting 37.94 → 37.47, mean chrF of 10 39.22 → 38.46, speaker_sim 0.290 → 0.286, HNR
4.59 → 3.42 dB, FLEURS-vi's end offset 1.3 → 2.2 s; `kl_text` 0.081, `kl_audio` 0.80 at 200,
grad norm ~2.3e4. Unlike the heads student's (`opd_heads_lr1e6`, +1.8 chrF each way by 200,
[heads-on-policy-runs](heads-on-policy-runs.md)), this student starts close to the teacher
(`kl_text` 0.09, `kl_audio` 0.84 against the heads' 0.10 / 2.12), so the teacher's pull adds little
and its leaning on the student's own audio (the caveat in
[on-policy-distillation](on-policy-distillation.md)) costs clarity. At 200:

| Set | chrF / BLEU / length ratio / end offset / speaker_sim / HNR |
|---|---|
| fleurs-vi/dev | 40.58 / 9.99 / 1.32 / 2.2 / 0.32 / 3.1 |
| fleurs-en/dev | 34.37 / 15.33 / 0.99 / 2.8 / 0.15 / 4.3 |
| en-vi-val cv27 | 45.26 / 17.66 / 1.20 / 2.8 / 0.33 / 3.3 |
| en-vi-val libriheavy | 45.19 / 19.56 / 1.06 / 2.5 / 0.42 / 3.2 |
| en-vi-val-noisy cv27 | 42.25 / 14.47 / 1.24 / 2.8 / 0.27 / 3.2 |
| en-vi-val-noisy libriheavy | 43.21 / 17.11 / 1.10 / 2.6 / 0.35 / 3.1 |
| en-vi-val-en cv27 | 33.83 / 16.60 / 1.13 / 1.9 / 0.26 / 3.8 |
| en-vi-val-en libriheavy | 43.40 / 22.14 / 1.07 / 2.5 / 0.26 / 3.8 |
| en-vi-val-en-noisy cv27 | 22.15 / 9.22 / 0.99 / 3.6 / 0.24 / 3.2 |
| en-vi-val-en-noisy libriheavy | 34.40 / 16.41 / 0.99 / 3.4 / 0.26 / 3.3 |

## Refit

The text is the trunk's and the speech the slices': so the trunk keeps its on-policy update and the
slices are distilled again on it, off-policy, as Stage 1 trained them (`restart.sh
student-slices-refit-5090`: the trunk frozen, the 3B on the SFT data, from `both_slices_z`'s best,
a fresh optimizer at 5e-5, cosine over 2,000 updates, validated at 1,000 and 2,000 on the slices
runs' seven sets, seed 47). Probe from update 200, started 2026-11-27 19:31 UTC from commit
904afa4, after its baseline: `eval.py` on the same seven sets and seed
(`runs/slices_eval7/both_slices_z200_seed47`, `--no-write-audio`, 4 min), beside the decay's 8,000
([student-slices](student-slices.md#runs)), chrF / BLEU / length ratio / end offset / speaker_sim /
HNR:

| Model | Selecting | Mean chrF | speaker_sim | HNR | fleurs-vi/dev | fleurs-en/dev | multimed-vi/select | en-vi-val cv27 | en-vi-val libriheavy | en-vi-val-en cv27 | en-vi-val-en libriheavy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decay 8,000 (on-policy start) | 38.32 | 40.54 | 0.293 | 4.94 | 41.15 / 9.74 / 1.39 / 1.5 / 0.34 / 4.5 | 35.49 / 16.47 / 1.01 / 1.9 / 0.16 / 5.4 | 37.98 / 10.29 / 1.48 / 3.4 / 0.27 / 6.0 | 45.93 / 17.93 / 1.18 / 2.1 / 0.32 / 4.1 | 44.98 / 17.89 / 1.11 / 2.1 / 0.41 / 4.4 | 34.36 / 17.19 / 1.11 / 1.1 / 0.26 / 5.0 | 43.92 / 22.29 / 1.07 / 1.6 / 0.29 / 5.2 |
| `both_slices_z` 200 | 38.49 | 40.93 | 0.293 | 4.11 | 41.18 / 10.21 / 1.33 / 1.7 / 0.33 / 3.7 | 35.81 / 16.37 / 1.00 / 2.0 / 0.16 / 4.6 | 38.41 / 10.99 / 1.41 / 3.4 / 0.28 / 5.3 | 45.49 / 17.75 / 1.15 / 2.5 / 0.33 / 3.5 | 46.09 / 20.40 / 1.06 / 2.1 / 0.42 / 3.4 | 35.45 / 16.91 / 1.15 / 1.0 / 0.27 / 4.1 | 44.12 / 22.95 / 1.07 / 2.0 / 0.27 / 4.2 |

On these sets the on-policy text gain is smaller (+0.17 selecting, +0.39 mean) and the clarity
loss the same (−0.83 dB, every set).

## Phone finish
