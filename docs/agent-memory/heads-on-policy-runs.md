# The heads student's on-policy runs

Read when running or judging the heads student's on-policy runs on a box: the probes' results,
`both_heads_z`, and `both_heads_z2` with its preparation and pre-run scores.
Back to [CONTEXT.md](../../CONTEXT.md) · [Probe results](#probe-results) ·
[The run, on the next box](#the-run-on-the-next-box)

The method, its arguments and its checks are in [on-policy-distillation](on-policy-distillation.md);
GRPO itself in [grpo](grpo.md); the box in [tranche-box](tranche-box.md#state).

## Probe results

On the RTX PRO 6000 (2026-11-21/22): `/workspace/probes.sh` (box-only) through `grpo_launch.sh`,
200 updates each, `--gen-frames 614400 --micro-batch 4 --save-every 100 --keep-checkpoints 1`,
seed 44, `--lr 1e-6` unless named, the teacher GRPO update 1,400. Validation at 0 and 200 on the
eight sets (batch 64, temperature 0.8, top-k 250). An update took ~5.5–6 s with the teacher and
~3.1 s without; a probe ~60 min plus its final upload. Raw outputs (`val_log.jsonl`, `eval/`,
`run_config.json`) are in `/Volumes/data/hibiki-zero-evals/box-61345966/runs/<run>/`, and the
pairs, bests and logs on the Hub under `<run>/`.

"Selecting" is the mean of FLEURS dev VI→EN and EN→VI chrF; "of 8" the mean over all eight
sets. End offsets are FLEURS dev's medians, VI→EN / EN→VI; the start's are in brackets. KL is the
train log's `kl_text` / `kl_audio` at update 1 → 200.

| Run | Start | Terms | Selecting 0 → 200 | Of 8 0 → 200 | End offset at 200 (start) | Health | Gate | KL |
|---|---|---|---|---|---|---|---|---|
| `opd_heads_lr1e6` | 176,000 | distillation | 36.78 → 38.61 | 37.24 → 38.82 | 2.21 / 2.25 s (1.95 / 1.45) | pass | blocked: EN→VI +0.80 s | 0.100 / 2.12 → 0.091 / 1.28 |
| `opd_heads_lr3e6` | 176,000 | distillation, 3e-6 | 36.78 → 38.52 | 37.24 → 39.01 | 2.29 / 2.46 s (1.95 / 1.45) | pass | blocked: EN→VI +1.01 s | → 0.093 / 1.29 |
| `opd_heads_lr1e5` | 176,000 | distillation, 1e-5 | stopped before its first pair: 3e-6 had gained nothing over 1e-6 | | | | | |
| `grpo_heads_lr1e6` | 176,000 | GRPO (the control) | 36.78 → 29.82 | 37.24 → 27.92 | 2.02 / 1.07 s (1.95 / 1.45) | 4 sets fail | on time | — |
| **`both_heads_z`** | 190,126 | both, z-scored | 37.33 → **38.92** | 37.88 → **39.44** | 1.71 / 1.63 s (2.00 / 1.42) | pass | on time, promoted | 0.102 / 2.18 → 0.087 / 1.34 |
| `both_heads_centered` | 190,126 | both, centered | 37.33 → 38.34 | 37.88 → 38.63 | 1.47 / 1.39 s (2.00 / 1.42) | pass | on time, promoted | → 0.103 / 1.42 |

- **Plain GRPO collapses the student**: in 200 updates at the 3B's rate its outputs ran long
  (length ratio 1.39–3.09) and every set lost 3.6–13.5 chrF; the depformer student's took 2,400
  ([student-grpo](student-grpo.md#evaluations)). The teacher's term is what holds the student.
- **Distillation alone gains but ends later**: every set but `en-vi-val/cv27` up, length ratios
  down (FLEURS VI→EN 1.40 → 1.25), but EN→VI ends 0.8 s later at 1e-6 and 1.0 s at 3e-6, and
  text EOS falls (FLEURS EN→VI 191 → 89 of 392 rows): past the 0.5 s gate. The faster rate
  bought nothing.
- **The reward keeps the timing**: `both_heads_z` gained as much (+1.59 selecting, +1.56 of 8,
  every set up), ending VI→EN 0.29 s sooner and EN→VI 0.21 s later. Centered ends sooner both
  ways but gains less (+1.01, +0.75); its GRPO term in BLEU points outweighs the KL (grad norm
  ~6e5 against z's ~1.5e5). z won: quality first (user, 2026-11-22: "quality is what we after,
  latency can be bit more relax").
- Why distillation alone ends later is unmeasured. Its KL prices no time (only the reward pays a
  word for coming early), and reverse KL is mode-seeking: where the next word is uncertain the
  teacher's mass is split over many words while waiting is one token, so the student may learn to
  wait. It is not the teacher's own timing: the 3B ends EN→VI ~1.5 s after the source
  ([grpo](grpo.md#running)), about where the student started.
- Wave 1 started at 176,000, wave 2 at 190,126 (0.55 higher selecting at update 0): compare
  probes by their change from their own update 0.

`both_heads_z` per set (FLEURS and `en-vi-val` rows; EOS counts the rows ending with text EOS):

| Set | chrF 0 → 200 | BLEU 0 → 200 | Length ratio | End offset (s) | Empty | EOS | Health |
|---|---|---|---|---|---|---|---|
| fleurs-vi/dev | 38.51 → 40.54 | 8.82 → 10.76 | 1.34 → 1.26 | 2.00 → 1.71 | 0 → 1 | 171 → 133 / 361 | pass |
| en-vi-val/cv27 | 35.93 → 36.52 | 11.23 → 12.88 | 1.34 → 1.22 | 2.78 → 3.07 | 1 → 0 | 81 → 55 / 180 | pass |
| en-vi-val/libriheavy | 39.87 → 41.29 | 14.50 → 16.78 | 1.16 → 1.08 | 3.11 → 2.79 | 0 → 0 | 88 → 52 / 180 | pass |
| en-vi-val-noisy/cv27 | 37.65 → 39.45 | 11.27 → 13.82 | 1.22 → 1.16 | 2.79 → 2.79 | 0 → 2 | 89 → 54 / 180 | pass |
| en-vi-val-noisy/libriheavy | 41.45 → 43.21 | 15.13 → 17.37 | 1.15 → 1.07 | 3.23 → 3.06 | 0 → 0 | 88 → 49 / 180 | pass |
| fleurs-en/dev | 36.15 → 37.30 | 15.71 → 16.86 | 1.05 → 1.04 | 1.42 → 1.63 | 0 → 0 | 203 → 92 / 392 | pass |
| en-vi-val-en/cv27 | 32.71 → 33.86 | 14.03 → 15.33 | 1.16 → 1.14 | 0.66 → 1.19 | 0 → 0 | 83 → 42 / 180 | pass |
| en-vi-val-en/libriheavy | 40.78 → 43.32 | 19.86 → 21.99 | 1.05 → 1.06 | 1.94 → 2.00 | 0 → 0 | 55 → 36 / 180 | pass |

### The run, on the next box

- **A new run, `both_heads_z2`, on the new data** (user, 2026-11-23: "ok go with your
  recommendation"): `both_heads_z`'s recipe from its start, the heads student's final model at
  190,126, on main and `grpo_launch.sh`'s caches ([grpo](grpo.md#running)): EN→VI from the
  tranche's noisy copies and round 4e's 500 `en_speech` shards clean and noisy. Its arguments
  after the launcher's: `--config-path weights/student_heads14_cb16/config.json --save-every 100
  --keep-checkpoints 1 --teacher-weight weights/grpo_twoway_best1400.safetensors --lr 1e-6` and
  its box's profile sizes (both terms, z-scored, seed 44 by default; `--updates 2000` from the
  launcher). It stops by the user's rule, two validations (every 200 updates) without a new best.
- Why not continue `both_heads_z`: main can't resume it (`pair_cache_dir` is a resume key,
  [sft-training](sft-training.md#target-timing)), and from its 200 pair the run would mix two
  datasets with AdamW restarted. Compare the two by each one's change from its own update 0 over
  the same eight sets (`both_heads_z`: +1.59 selecting, +1.56 of 8): the evaluation reseeds by
  batch over all its rows, so two more sets or another `--eval-batch-size` move every set's
  absolute scores. The change measures the data and main's streams together (the start floor,
  the babble and OCR drops, [sft-training](sft-training.md#target-timing)). `both_heads_z` stays on
  the Hub, its commit `9d9573e` as the branch `run/both_heads_z`.
- The teacher never heard noisy English, so the box scored the teacher and the start on
  `en-vi-val-en` and its noisy copy before the run (user, 2026-11-23: "should we train the teacher
  a bit on noisy data?"; training the 3B on it is the next two-way run's, [envi-data](envi-data.md#decision)).
  The noisy set also gates promotion. Scored 2026-11-23 on the H100 box, `eval.py --batch-size 64
  --seed 44 --text-only`, 180 rows a set (raw outputs in
  `/Volumes/data/hibiki-zero-evals/box-62170174/runs/prerun_z2/`). Every set passes its health
  gates for both, and the teacher stays ahead on the noisy sets (+3.0 and +5.3 chrF, against +6.1
  and +5.0 clean), so its KL on noisy rows still points to a better model. Teacher (GRPO 1,400) →
  start (the heads student at 190,126):

  | Set | chrF | BLEU | EO (s) | Length |
  |---|---|---|---|---|
  | `en-vi-val-en/cv27` | 38.98 → 32.89 | 19.59 → 14.72 | 0.92 → 0.93 | 1.17 → 1.17 |
  | `en-vi-val-en-noisy/cv27` | 26.60 → 23.58 | 11.85 → 8.70 | 1.42 → 1.62 | 1.06 → 1.11 |
  | `en-vi-val-en/libriheavy` | 45.73 → 40.75 | 23.99 → 19.74 | 1.92 → 1.92 | 1.08 → 1.06 |
  | `en-vi-val-en-noisy/libriheavy` | 37.94 → 32.68 | 19.52 → 15.11 | 2.31 → 2.55 | 1.02 → 1.00 |

- `h100.sh` takes the RTX 5090 and sizes its checks from the weights: 24.9 GiB of GPU states, 54
  GiB of RAM in the container and 42 GiB of disk ([sft-recovery](sft-recovery.md#launcher)). The
  RAM is the 3B's measured structure, 34.4 GiB pinned and 17 more: measure the run's own once it
  is past window 2 (`HostOffload` pins AdamW's states from then) and its first save (the writer's
  buffers), by the cgroup's `memory.stat` `anon` or the process's `VmHWM`. `grpo_profile` has no
  writer, and `memory.peak` counts the packs' page cache. On a 350 GB disk: the caches ~172 GB as
  packs ([envi-data](envi-data.md#the-runs-on-it)), the run 42, weights ~25, the env and
  evaluation sets ~15.
- Profile on its box first (`grpo_profile --teacher-weight`; the H100 box's `/workspace/prof_z2.sh`):
  `--micro-batch`, `--gen-frames`, `--eval-batch-size` (64 so far; validation runs with the update's
  states on the GPU). The RTX PRO 6000 measured ([on-policy-distillation](on-policy-distillation.md#checks)) the update near 26 GiB at
  micro-batch 1 (31.0 at 2) and generation 2.4 GiB plus ~0.066 a row: on a 32 GB 5090, micro-batch
  1 and ≤ ~400 rows (`--gen-frames` = rows × 1,200), at little cost (9.0 rollouts/s at 512 rows,
  9.3 at 1,024).
- Not on 32 GB: the 3B's full fine-tunes (a pair is 12.5 GB of weights and 25 GB of AdamW), and
  the palettization finish at the heads run's batch (50.9 GiB, [tranche-box](tranche-box.md#jobs)).
