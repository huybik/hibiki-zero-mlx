# The heads student's on-policy runs

Read when running or judging the heads student's on-policy runs on a box: the probes' results,
`both_heads_z` and its continuation past 200, and `both_heads_z2` with its pre-run scores, results
and the sizes a 5090 takes. Back to [CONTEXT.md](../../CONTEXT.md) · [Probe results](#probe-results) ·
[`both_heads_z2`](#both_heads_z2) · [`both_heads_z` continued](#both_heads_z-continued) ·
[On the 5090](#on-the-5090)

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

### `both_heads_z2`

- **Done 2026-11-23, 18:53–21:42 UTC** (commit `f0b3b8d`, on the RTX 5090 box 62188551; its records, without weights, are in
  `/Volumes/data/hibiki-zero-evals/box-62188551/`, the pairs of updates 200 and 400 on the Hub): `both_heads_z`'s
  recipe from its start, the heads student's final model at 190,126, on main and
  `grpo_launch.sh`'s caches ([grpo](grpo.md#running)), the new data: EN→VI from the tranche's
  noisy copies and round 4e's 500 `en_speech` shards clean and noisy, `--pair-cache-dir` timing
  it. Arguments after the launcher's: `--config-path weights/student_heads14_cb16/config.json
  --save-every 100 --keep-checkpoints 1 --teacher-weight weights/grpo_twoway_best1400.safetensors
  --lr 1e-6 --gen-frames 384000 --micro-batch 2 --park-states` (both terms, z-scored, seed 44;
  `--updates 2000` from the launcher).
- **It was stopped at 400 and promoted nothing.** The agent took two validations without a
  promotion for the user's rule ("stop distil when 2 rounds of validation not improving",
  2026-11-15), but the selecting mean rose at both: they were blocked by the latency gate alone,
  so by the rule's words it would have gone on (the reading since, [below](#both_heads_z-continued)).
  It was healthy but ended too late, past the 0.5 s gate on FLEURS dev by 0.19 s VI→EN
  and 0.16 s EN→VI at 200 and, at 400, by 0.04 s on VI→EN alone. Validation, 0 → 200 → 400:
  selecting mean 37.06 → 38.49 → 38.63; mean of the original 8 sets 37.83 → 38.16 → 37.98
  (`both_heads_z`: 37.33 → 38.92 and 37.88 → 39.44); end offset VI→EN 1.73 → 2.42 → 2.27 s and
  EN→VI 1.50 → 2.16 → 1.82 s. Both 200 and 400's pairs are on the Hub; no best exists.
- **`both_heads_z`'s best is the better model.** Test sets (2026-11-24, the box, `eval.py` seed 43,
  batch 64, text only, LAAL; the 11 of [student-heads](student-heads.md#tests) and the two
  `en-vi-val-en-noisy` in one call, so absolute scores differ from that table's; every set passes
  health for all four). chrF, and paired bootstrap over rows (500 draws, 11 sets):

  | Set | Start (190,126) | `both_heads_z` 200 | `z2` 200 | `z2` 400 |
  |---|---|---|---|---|
  | fleurs-vi/test | 36.70 | 39.51 | 38.69 | 38.34 |
  | fleurs-en/test | 34.63 | 36.78 | 37.13 | 37.52 |
  | multimed-vi/select | 35.64 | 36.64 | 35.22 | 33.92 |
  | multimed-vi/final | 35.74 | 36.72 | 35.17 | 34.09 |
  | giga-heldout/test | 61.64 | 64.49 | 63.63 | 63.80 |
  | en-vi-val/cv27 | 37.80 | 39.23 | 37.70 | 36.76 |
  | en-vi-val/libriheavy | 41.45 | 43.74 | 43.36 | 40.99 |
  | en-vi-val-noisy/cv27 | 38.08 | 40.66 | 39.30 | 39.82 |
  | en-vi-val-noisy/libriheavy | 41.28 | 42.68 | 43.05 | 43.34 |
  | en-vi-val-en/cv27 | 31.72 | 32.83 | 33.63 | 34.68 |
  | en-vi-val-en/libriheavy | 40.59 | 43.89 | 43.12 | 43.21 |
  | en-vi-val-en-noisy/cv27 | 22.76 | 24.93 | 26.78 | 26.42 |
  | en-vi-val-en-noisy/libriheavy | 33.22 | 37.14 | 37.76 | 37.94 |
  | Mean of 11 (`-noisy` in `-en` left out) | 39.57 | **41.56** | 40.91 | 40.59 |
  | Mean of 13 | 37.79 | **39.94** | 39.58 | 39.29 |
  | Mean LAAL of 11 (s) | 2.66 | 2.78 | 3.02 | 3.05 |
  | Mean median end offset of 11 (s) | 2.70 | 2.62 | 2.98 | 2.98 |

  Against the start, `both_heads_z` +1.99 [+1.62, +2.32], `z2` 200 +1.34 [+0.97, +1.69], 400 +1.01
  [+0.61, +1.41]; `z2` 200 against `both_heads_z` −0.65 [−0.96, −0.27], 400 −0.97 [−1.34, −0.60]
  (11 sets). Against `both_heads_z`, `z2` (200, 400) is ahead on
  EN→VI `en-vi-val-en` cv27 (+0.8, +1.9) and its noisy cv27 (+1.9, +1.5), and behind on MultiMed-ST
  (−1.4 to −2.7), FLEURS-vi (−0.8, −1.2), `giga-heldout` (−0.9, −0.7) and `en-vi-val` cv27 (−1.5, −2.5);
  the rest are within ±1 at 200 but `en-vi-val-noisy` cv27 (−1.4; 400 also `en-vi-val` libriheavy −2.8).
  **Its cause is not isolated**: the data, main's streams since
  `9d9573e` (the start floor, the babble and OCR drops,
  [sft-training](sft-training.md#target-timing)), the sizes (`--micro-batch 2 --gen-frames 384000`,
  not 4 and 614400) and ten evaluation sets all changed.
- **Wrong-language drift**: rows whose output is in the other language (a check for Vietnamese
  letters, so approximate). Test sets, VI→EN, start / `both_heads_z` / `z2` 200 / `z2` 400:
  `multimed-vi/select` 96 / 103 / 205 / 305 of 1,718, `final` 132 / 140 / 245 / 308 of 1,719,
  FLEURS-vi 32 / 22 / 46 / 53 of 857, `giga-heldout` 17 / 12 / 24 / 32 of 1,000, `en-vi-val` cv27
  31 / 26 / 42 / 50 of 180. EN→VI falls (FLEURS-en 15 / 7 / 1 / 1 of 641; `en-vi-val-en-noisy` libriheavy
  28 / 20 / 7 / 5 of 180). On validation's `en-vi-val` cv27 it went 36 → 54 → 70 of 180 in
  `z2`, against 40 → 39 in `both_heads_z` (0 → 200). Some of those outputs repeat the source's
  own Vietnamese (chrF ≥ 50 against it: 2 → 11 → 15 of the wrong rows on that set).
- A set's chrF moves more than its row-resampling interval says: at 200, `en-vi-val` cv27 was
  −4.30 in validation (seed 44, rows' 95% interval −6.3 to −2.3) and −0.10 in the test call (seed 43,
  other batch mates, interval −2.1 to +2.2). Judge on several sets, not one.
- `both_heads_z2` was a new run rather than `both_heads_z` resumed: main can't resume it
  (`pair_cache_dir` is a resume key, [sft-training](sft-training.md#target-timing)), and from its
  200 pair the run would mix two datasets with AdamW restarted.
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

### `both_heads_z` continued

- **Why**: `both_heads_z` was a 200-update probe (`probes.sh`). Its long run, queued 2026-11-22,
  was dropped for `both_heads_z2`, yet its update 200 beats `both_heads_z2` on the tests, and the 3B's
  GRPO kept rising to 1,400 ([grpo](grpo.md#running)). So it goes on to 2,000 from its pair at
  200, on its own data (user, 2026-11-24: "i rent a new box, let's go").
- **The code**: `run/both_heads_z` at `c058985`, `9d9573e` with main's `h100.sh` and #177's
  `grpo.py` and `grpo_profile.py` (`--park-states`; evaluation parks AdamW's states and the
  teacher; `ALLOC_CONF`). They move memory, not results (12 parked AdamW steps bitwise equal,
  [On the 5090](#on-the-5090)); its `grpo.py` differs from main only by #174's
  `--pair-cache-dir` and two evaluation sets.
- **The resume** ([sft-recovery](sft-recovery.md)): 200's pair, best and logs from the Hub;
  `run_config.json` and `train_args.json` edited by hand (`updates` and `total_steps` 2,000,
  `--gen-frames 384000 --micro-batch 2 --park-states`, neither size a resume key); `run_id.json`'s
  commit set to `c058985` and uploaded as `run.json`. The originals are in `/workspace/logs/z200/`
  on the box. 200's files are also copied to `both_heads_z_at200/` on the Hub, as the run prunes
  its own prefix.
- **Running** on the RTX 5090 box 62366622 ([tranche-box](tranche-box.md#state)) from 2026-11-24
  15:44 UTC, through `grpo_launch.sh both_heads_z` with `HF_HUB_DISABLE_XET=1`: the same 9,088,899
  rows and 1,721,906 streams. Update 201's `kl_text` 0.089, `kl_audio` 1.43 and grad norm 1.5e5
  continue 200's (0.087, 1.34, 1.5e5). An update takes ~9.8 s, its window's generation ~12.9 s and
  π_old's pass ~2.3 s: ~85 min from one validation to the next. MemAvailable was ≥ 47.7 GiB
  through update 220 (an 83.8 GiB container).
  The box's profile: generation at 320 rows 44.3 ms a step (peak 23.4 GiB), the update at
  micro-batch 2 ~8.0 s (22.4 GiB). Update 400 promoted: selecting mean 39.36 (200: 38.92), mean
  of 8 39.44 (39.44), end offsets VI→EN 1.74 s and EN→VI 1.82 s (200: 1.71, 1.63).
- **The gate 3 s from update 500** (user, 2026-11-25; [grpo](grpo.md#design)): the run stops at its
  pair of 500 and continues on `c4c061f`, which changes only the gate (validation, so a commit
  move, [finetune](../finetune.md#move-a-running-run-to-another-pod)). Every validation so far is under 3 s.
- **The stop**: two validations in a row with no new best on the selecting mean or the mean of
  8. A validation the latency gate blocked while either mean rose is not "not improving"
  (the agent's reading, put to the user 2026-11-24, not yet answered). Then the test sets as
  `both_heads_z2`'s.
- **Then `both_heads_z2` from its 400** on the same box (user, 2026-11-25: "can later try both head
  z2 too min 800 steps"): to 800 at least, then the same stop, on main's HEAD. Its pair of 400 and
  its five caches `both_heads_z` lacks download meanwhile (packed once `both_heads_z` stops). At
  3 s its 200 and 400 pass the gate, so its `best.json` is set to 400 (selecting 38.63) first.

### On the 5090

- `h100.sh` takes the RTX 5090 and sizes its checks from the weights: 24.9 GiB of GPU states, 54
  GiB of RAM in the container and 42 GiB of disk ([sft-recovery](sft-recovery.md#launcher)). The
  RAM is the 3B's measured structure, 34.4 GiB pinned and 17 more. On the 5090 box (58.0 GiB
  cgroup) the first 40 updates held 34.4 GiB pinned and 13.5 of `anon` past window 2 and the save at
  20, `memory.current` peaking at 52.3 GiB (page cache included) and the host's MemAvailable bottoming
  at 4.3 to 4.7 in three runs, and 3.79 in `both_heads_z2`'s 2 h 48 min: tight, so run with `HF_HUB_DISABLE_XET=1` and watch MemAvailable. Pinned memory is charged to
  the cgroup as `shmem` (add it to `anon`, from `memory.stat`), and `memory.peak` counts the packs'
  page cache. `grpo_profile` has no writer. On a 350 GB disk: the caches ~172 GB as
  packs ([envi-data](envi-data.md#the-runs-on-it)), the run 42, weights ~25, the env and
  evaluation sets ~15.
- Profile on its box first (`grpo_profile --teacher-weight`): `--micro-batch` and `--gen-frames`.
  It has no evaluation stage, so check `--eval-batch-size` (64 so far) by a short run past an
  evaluation with the optimizer stepped. On the 5090 (2026-11-24, 575 W, one run of each): generation at 320 rows (`--gen-frames 384000`) 45 ms a step and 5.8 rollouts/s,
  peak 23.4 GiB, at 384 rows 27.6 GiB, 448 out of memory. Without `--park-states` the
  micro-batches peaked at 29.0 GiB at 1 and a run of it ran out of memory at update 7 (31.0 GiB in
  use; its log was overwritten by the next run); with it, the
  micro-batches peaked at 19.5, 21.6, 23.8 and 25.9 GiB at 1 to 4 (8.6, 8.2, 8.2 and 7.9 s an
  update by the profile's estimate, π_old's pass 1.2–1.4 more, the parking copies not counted);
  the optimizer's step holds 24.9. The run holds more than the profile, so it stays at `--gen-frames 384000
  --micro-batch 2 --park-states`: ~9.3 s an update, ~13 s of generation and ~1.4 s of π_old's pass
  an update, so 2,000 take ~13.2 h and 11 evaluations of ~2 min. The evaluation at batch 64 with
  AdamW's states on the card ran out of memory, so `evaluate` parks them and the teacher.
- `--park-states` (grpo.py and `grpo_profile`): only `optimizer.step` reads AdamW's states, so they
  wait in pinned host memory while π_old's logprobs and an update's micro-batches run
  (`grpo.park_states`; 9.5 GiB on the heads student, 23.3 on the 3B). It costs a copy out and back an
  update (~0.8 s here, extrapolated from a 0.81 GB test, without `empty_cache` and the remapping) and changes no result: 12 fused AdamW steps
  were bitwise equal with and without it. Cards where the update fits without it leave it off.
- Not on 32 GB: the 3B's full fine-tunes (a pair is 12.5 GB of weights and 25 GB of AdamW), and
  the palettization finish at the heads run's batch (50.9 GiB, [tranche-box](tranche-box.md#jobs)).
