# Training against the palettization

Read when touching `training/student/palettize.py`, `train.py --palettize-bits` or `--park-states`,
or training, exporting or judging a heads student against the phone's Core ML palettization. Back
to [CONTEXT.md](../../CONTEXT.md) · [Method](#method) · [The 6-bit run](#the-6-bit-run) ·
[Tests](#tests) · [Neural Engine agreement](#neural-engine-agreement) ·
[Release and phone](#release-and-phone). The student is in [student-heads](student-heads.md), its Core ML export and the
Precision table in [lm-npu](lm-npu.md#precision).

Plan stage 6.4 (2026-11-21): on the Neural Engine the 6-bit palettized LM agrees with bf16's
argmax on 92.1% of tokens against fp16's 94.6% ([lm-npu](lm-npu.md#precision)), and 4 bits
(~490 MB) would free bandwidth.

## Method

- **What the export does** (measured 2026-11-21 on a layer used as `lm_step` uses it, `x @ W.T`):
  every trunk and head weight over 2^16 elements (2^20 until 2026-11-27; every heads-student
  weight is over both) gets one k-means table of 2^N entries per 16 output
  channels, rows of the torch weight (tables [rows / 16, 1, 2^N, 1]); `convert_mlx.py` keeps the
  rows in order (the heads are concatenated, 2,048 rows each). A weight whose every 16 rows already
  hold at most 2^N values comes back exact: the export's k-means (scikit-learn here, kmeans1d where
  installed) finds them.
- **`fit`**: entries spread as the density's cube root over 1,024 equal-count bins, then 10 Lloyd
  iterations on each group's sorted values (an entry's values one contiguous range, means from
  float64 prefix sums: deterministic on a device), an empty entry moved to the worst-fit value of
  another entry's range. On 8 random groups each of the update-16,000 student's `in_proj`,
  `linear_out`, `text_linear` and heads, its error is 0.94–1.07 times the export's own k-means at
  6 bits and 0.93–0.97 at 4: snapping alone exports within ~7% of the plain export's error.
- **Training** (`--palettize-bits N`, `--palettize-refit` 100): the forward and backward run on
  the weights snapped to their tables; the gradients there update the full-precision weights
  (straight-through), restored before AdamW steps. Validation scores the palettized model. The
  tables are refit on the full-precision weights at the start, every `--palettize-refit` steps
  (which must divide `--save-every`) and after the last step, so a resume and the snap tool (on
  CUDA, as training fits) get the tables the run had. Saved checkpoints hold the full-precision
  weights. The swap copies each palettized weight twice a step (~50 GB of memory traffic on the
  student, an estimated 6–9% of a step) and keeps a 4.2 GB full-precision copy.
- **Parking** (`--park-states`): while validation translates free-running, AdamW's states and
  the teacher wait in pinned host memory (`common.HostOffload`, as GRPO's evaluation parks them):
  9.5 + 2.4 GiB on the heads student distilling itself, more as pinned blocks, since train.py
  leaves torch's pinned allocator rounding each up to a power of two (GRPO's `ALLOC_CONF` stops
  that). Without it, that run's validation on a 32 GB card ran out of memory at Mimi's encode of
  the first batch, at batch 64 and 32 (2026-11-25; 28 GiB held, the palettizer's full-precision
  copy among it). Off by default: the 3B's SFT fits on 80 GB without it, and parking would pin
  23 GiB more of host memory, which `h100.sh`'s fixed 80 GiB RAM check for SFT doesn't count.
  GRPO after `HostOffload` and `adam_states` moved to `common.py` (2026-11-25, main e8dfbe8,
  the 5090): 4 updates of the heads student from `both_heads_z` update 1,200 (4 streams × 4
  rollouts, `--park-states`, an evaluation and a save every 2) ran to the final save; first
  ratio 1.0000, then ≤ 1.0003, no clipping, gradient norms 2.7–4.4e5 (the 3B's 2026-11-13 check:
  4.4–4.9e5). A smoke test, not a comparison with the code before the move. Records:
  `/Volumes/data/hibiki-zero-evals/box-62513802/grpocheck/`.
- **Export**: `python -m training.student.palettize --weight CKPT --bits N --out SNAPPED` snaps a
  checkpoint onto its tables; then `convert_mlx.py` (bf16) and `export_lm_coreml.py --bits N` as
  before.
- **The export round trip** (2026-11-21, the Mac, `student_heads14_cb16`'s best at 112,000): the
  snap tool on the CPU (66 s, one thread), `convert_mlx.py` (bf16; the heads need a worktree's
  `PYTHONPATH`, [mlx-inference](mlx-inference.md)) and `export_lm_coreml.py --bits 6` (compiled
  both models). Every palettized weight, read back from the palettized programs before compiling
  (`constexpr_lut_to_dense`: `lut[row // 16, 0, index, 0]`), equals the fp16 weight it replaced bit
  for bit, 64 values in every 16 rows: the trunk's 57 (each layer's four projections and
  `text_linear`) and the 16 heads. So a snapped student exports without loss.
- **Validated on the box** (2026-11-21, with the rest of the new code, after the heads run and
  before any GRPO, [on-policy-distillation](on-policy-distillation.md#experiments) item 0): from
  the best at 176,000, distilled from GRPO update 1,400, 1e-5 constant, `--palettize-refit 50`:
  - 73 weights palettized at 6 bits, the export's count; 0.70 s a step against the unpalettized
    run's 0.69.
  - SIGKILLed after the update-100 pair (the run had reached step 130 while the pair was written);
    the resumed run's step 130 logged the same loss to four decimals (0.4054), so the model, AdamW
    and the tables came back as they were. FLEURS dev 38.18 / 34.92 at 150, promoted.
  - The snap tool on CUDA: all 73 weights at most 64 values in every 16 rows, every other tensor
    unchanged, the `palettized` metadata written. At seed 47 (`eval.py`) the snapped model scored
    38.06 / 35.23 and the full-precision weights it came from 39.00 / 34.45: means 36.65 and
    36.73, within noise.
- Checked on the CPU (tiny modules, scratch script): at most 2^N values a group, snapping
  idempotent, the full-precision weights restored exactly after a step, the gradient equal to that
  of a model holding the snapped weights.

## The 6-bit run

`pal6_heads_z1200c` (2026-11-25 13:47–14:13 UTC, box 62513802's RTX 5090, 32 GB,
[tranche-box](tranche-box.md#state); commit aef7f68 of PR #188). Records, scripts and logs, no
weights: `/Volumes/data/hibiki-zero-evals/box-62513802/pal6/` (`records.tgz`; `table.txt` and
`paired.txt` hold the tests below).

- **Recipe**: from `both_heads_z`'s best at 1,200, the best heads student
  ([heads-on-policy-runs](heads-on-policy-runs.md#both_heads_z-continued); Hub
  `both_heads_z/best/best_step001200.safetensors`, sha256
  `0b94ea3e43b7a3f3835ac5c5755e64b912fe6510cda5e1b00696ab65ac68396c`), distilled from its own
  bf16 weights (`--teacher-weight` that checkpoint, `--teacher-config` its config):
  `--palettize-bits 6 --palettize-refit 100 --park-states --batch-frames 3072 --lr 5e-7
  --max-steps 4000` (cosine, 5% warmup), seed 50, validation every 1,000 at eval batch 64. Caches:
  the tranche's `pho`, `vivoice` and `giga` both ways, `en_speech_sub`'s noisy train and
  `en-vi/en-vi-grounded-v2/train`. 0.20–0.21 s an update, 155 s a validation: 26 min in all.
- **Memory**: the heads run's batch needs 50.9 GiB ([tranche-box](tranche-box.md#jobs)); 4,096
  frames ran out of memory in the backward near update 125 (`pal6_heads_z1200`); 3,072 fit, with
  parking for validation.
- **The rate is low on purpose**: the KL from its own bf16 weights is ~0.003, and Adam's step does
  not shrink with so small a gradient, so a usual rate walks the weights away from the teacher.
  At a 4e-6 peak (`pal6_heads_z1200b`, stopped at 1,290) the KL fell to 0.0025 by update 170 and
  then rose above the start's, 0.0045 at 810. Constant-rate probes (1,200 updates each from the
  start, same data and seed, `probe_lr.sh`), mean KL over updates 1,001–1,200: 2e-6 0.00257
  (rising since 400), 1e-6 0.00209 (flat since 600), 5e-7 0.00193 (still falling). 5e-7 was
  chosen.
- **Training KL** (mean of each 200 updates): 0.00304 at 200 (text 0.00087, audio 0.00217),
  0.00197 at 1,000, 0.00185 at 2,000, 0.00179 at 3,000, 0.00168 at 4,000 (text 0.00048, audio
  0.00121); still falling as the rate decayed to zero.
- **Validations** (the palettized model, seed 50): selecting chrF (FLEURS dev both ways) 39.87 at
  1,000, 39.65 at 2,000, 39.61 at 3,000, 39.65 at 4,000, all healthy; 1,000 was kept as best, a
  difference inside noise. Every set's values are in the records' `val_log.jsonl`.

## Tests

The same 13 sets and call as `both_heads_z`'s
([heads-on-policy-runs](heads-on-policy-runs.md#both_heads_z-continued)): `eval.py` seed 43, batch
64, text only, LAAL, the box (4.3 min a model). Three models: the start as it is; the start
snapped onto its 6-bit tables without training; the run's final update 4,000 snapped (the tables
the run trained against). Snapping ran on CUDA (`training.student.palettize`).

Each cell is chrF / BLEU / median End Offset (s) / mean LAAL (s) / length ratio / empty rows;
every set healthy:

| Set | Start | Start snapped | Update 4,000 snapped |
|---|---|---|---|
| fleurs-vi/test | 40.21 / 10.96 / 1.6 / 3.03 / 1.22 / 0 | 39.77 / 10.64 / 1.5 / 2.93 / 1.22 / 1 | 39.80 / 10.82 / 1.7 / 2.99 / 1.23 / 1 |
| fleurs-en/test | 37.71 / 17.23 / 1.9 / 2.77 / 1.04 / 1 | 37.61 / 17.25 / 1.9 / 2.76 / 1.04 / 1 | 38.04 / 17.51 / 2.1 / 2.79 / 1.03 / 0 |
| multimed-vi/select | 37.54 / 11.44 / 3.9 / 3.25 / 1.31 / 2 | 37.31 / 11.41 / 3.8 / 3.20 / 1.31 / 5 | 37.39 / 11.29 / 4.0 / 3.28 / 1.31 / 4 |
| multimed-vi/final | 37.42 / 11.60 / 4.0 / 3.29 / 1.29 / 6 | 37.27 / 11.17 / 4.0 / 3.27 / 1.31 / 7 | 37.40 / 11.50 / 4.1 / 3.31 / 1.30 / 7 |
| giga-heldout/test | 65.86 / 38.50 / 2.8 / 2.43 / 1.30 / 2 | 65.57 / 38.17 / 2.9 / 2.43 / 1.38 / 2 | 65.88 / 39.30 / 2.8 / 2.45 / 1.29 / 2 |
| en-vi-val/cv27 | 39.60 / 14.83 / 2.8 / 2.88 / 1.18 / 0 | 40.32 / 14.48 / 2.7 / 2.81 / 1.20 / 0 | 41.15 / 14.54 / 2.7 / 2.88 / 1.15 / 1 |
| en-vi-val/libriheavy | 43.94 / 18.80 / 3.0 / 3.21 / 1.07 / 0 | 43.26 / 18.15 / 2.7 / 3.12 / 1.06 / 0 | 44.30 / 19.25 / 2.7 / 3.17 / 1.06 / 0 |
| en-vi-val-noisy/cv27 | 41.94 / 14.62 / 2.8 / 2.90 / 1.15 / 0 | 41.30 / 14.26 / 2.8 / 2.88 / 1.21 / 0 | 42.16 / 15.41 / 2.9 / 2.91 / 1.17 / 3 |
| en-vi-val-noisy/libriheavy | 43.90 / 18.70 / 2.8 / 3.11 / 1.04 / 0 | 44.17 / 18.23 / 2.7 / 3.10 / 1.05 / 0 | 43.21 / 17.54 / 2.7 / 3.13 / 1.05 / 0 |
| en-vi-val-en/cv27 | 35.09 / 16.19 / 1.3 / 1.97 / 1.11 / 0 | 35.25 / 17.94 / 1.3 / 1.97 / 1.12 / 0 | 34.03 / 16.06 / 1.1 / 2.00 / 1.12 / 0 |
| en-vi-val-en/libriheavy | 44.68 / 23.52 / 2.2 / 2.62 / 1.05 / 0 | 44.59 / 23.68 / 2.1 / 2.61 / 1.05 / 0 | 43.97 / 22.71 / 2.2 / 2.67 / 1.05 / 0 |
| en-vi-val-en-noisy/cv27 | 26.01 / 11.34 / 2.2 / 2.95 / 0.99 / 5 | 25.95 / 11.21 / 2.4 / 2.86 / 1.01 / 5 | 26.25 / 11.83 / 2.1 / 2.93 / 1.02 / 5 |
| en-vi-val-en-noisy/libriheavy | 37.98 / 18.94 / 2.8 / 3.30 / 0.99 / 0 | 38.15 / 19.28 / 3.1 / 3.37 / 1.00 / 1 | 38.10 / 18.93 / 3.1 / 3.29 / 1.02 / 0 |
| **Mean of 11** (no en-noisy) | chrF 42.53, BLEU 17.85, LAAL 2.86 | 42.40, 17.76, 2.83 | 42.48, 17.81, 2.87 |

- **Paired bootstrap of chrF** (`paired.py`: 500 draws of each set's rows, the same rows for both
  models), mean of 11 sets: start snapped − start −0.13 [−0.48, +0.19]; update 4,000 snapped −
  start −0.05 [−0.36, +0.25]; update 4,000 snapped − start snapped +0.08 [−0.24, +0.41]. Means
  of 13: −0.10, −0.01, +0.09, all intervals across zero. Per set, two of the last comparison's 13
  intervals miss zero, about what chance gives: `en-vi-val/libriheavy` +1.05 [+0.02, +2.39] and
  `en-vi-val-noisy/libriheavy` −0.96 [−2.00, −0.06].
- **Reading**: at 6 bits the palettization costs no measurable translation quality on the GPU,
  snapped without training or not, so free-running chrF cannot rank them; the Neural Engine's
  agreement below does.

## Neural Engine agreement

The [Precision](lm-npu.md#precision) measurement (2026-11-25, the Mac, M4 Pro): the start in MLX
bf16 (`z1200_bf16`) translated the first 8 FLEURS dev rows each way (seed 42 + row, the gate's),
recording every frame's inputs and sampled text; each Core ML model replayed those 4,533 frames on
the Neural Engine. Share of argmaxes equal to the reference's:

| Model | Text | Codebook 0 | 1–7 | 8–15 | All 16 |
|---|---|---|---|---|---|
| Start, Core ML fp16 (`z1200_ane16`) | 99.6% | 98.9% | 97.0% | 96.9% | 97.1% |
| Start, 6 bits, the export's own k-means (`z1200_ane6`) | 99.3% | 97.8% | 93.9% | 93.3% | 93.8% |
| Start snapped, 6 bits (`z1200snap_ane6`) | 99.4% | 97.7% | 93.6% | 92.9% | 93.5% |
| Update 4,000 snapped, 6 bits (`pal6c4000_ane6`) | 99.5% | 98.2% | 94.8% | 94.3% | 94.7% |

- **Training wins on the phone's arithmetic.** Paired over the 16 clips (10,000 draws), All 16:
  update 4,000 − the plain 6-bit export +0.89 [+0.71, +1.07], − the snapped start +1.23 [+0.98,
  +1.55]; snapping alone −0.34 [−0.57, −0.16] (its k-means fits slightly worse than the
  export's). Text +0.15 [+0.00, +0.32], at the ceiling. It closes about a quarter of the 6-bit
  export's gap to fp16 (3.3 points).
- 2,543 of the frames follow the source (eight clips wrote no EOS and ran the 20 s silent tail),
  where agreement comes easier. While the source plays (1,990 frames), All 16: fp16 95.9, plain
  6 bits 91.3, snapped 90.8, update 4,000 92.4.
- Update 4,000 is a trained model scored against the start's argmaxes, so its gain is less
  palettization error net of any drift from the teacher it distilled.
- The export round trip: every palettized weight read back from the compiled models equals the
  snapped one, bit for bit in 72 of update 4,000's 73 weights (one group of
  `layers.4.gating.linear_in` stores −0.0 where the export writes +0.0) and all 73 of the snapped
  start's.
- Tools and records: `/Volumes/data/hibiki-zero-evals/mac/npu-agree/` (`npu_agree.py record` and
  `replay`; `phase2.sh TAG CHECKPOINT` stages, exports at 6 bits and replays a checkpoint against
  the saved trajectory in ~12 min; `roundtrip.py`, `compare.py`; run through `py.sh`, which puts
  the worktree's code first at nice 19). On update 16,000's models and 4 clips it reproduces the
  Precision table within a point once the silent tails are cut. Models are under
  `/Volumes/data/models/hibiki/weights/student_heads14_cb16/` (`z1200_*`, `z1200snap_*`,
  `pal6c4000_*`).

## Release and phone

- **Release** (2026-11-25): update 4,000 snapped, exported at 6 bits, with update 16,000's
  16-codebook Mimi decoder, is the Hub's `release/student_heads_coreml6/` (1.74 GB, 16 files and a
  card). On the Mac: `/Volumes/data/models/hibiki/weights/student_heads14_cb16/pal6c4000_ane6/`;
  on the phone, HibikiBench's `Documents/pal6c4000/`.
- **Phone** (HibikiBench `--model pal6c4000 --clip clip-vi.wav --minutes 10`, the iPhone 13 Pro
  Max wired and from cool, Mimi on the GPU): 53.0–53.5 ms a frame (p50) every minute, p90 ≤ 54.3
  after minute 1, the Core ML LM 51.1–51.8, lag ≤ 0.12 s, thermal "serious" from minute 7 without
  slowing, footprint 541–542 MB (717 in minute 1): update 16,000's speed
  ([lm-npu](lm-npu.md#speed)). Over the 10-minute loop it wrote two sentences and a few words,
  then nothing; update 16,000 rambled into garbage. A loop past the 3,000-frame context tests no
  quality. Files: `/Volumes/data/hibiki-zero-evals/mac/npu-agree/phone/`.
- **Speech Live**: [onevoice-ai-challenge PR #8](https://github.com/anthoai97/onevoice-ai-challenge/pull/8)
  copies the Core ML LM into the app's runtime and makes this release its download (merged
  2026-11-26).
  On the phone (2026-11-25, the phone hot from the bench), a `--file` run of two FLEURS
  Vietnamese sentences 4 s apart translated both turns at 53 ms a frame (p50; p90 65, max 110),
  ≤ 0.2 s behind, 752 MB peak footprint; the MLX depformer student on the same clip: 75 ms,
  1.7 GB. A phone with other models downloads it from Settings › Model.
- **Speech Live's sampling** (2026-11-26): the app samples as `lm_gen_config` says (0.8 / top 250;
  the Mac app forces 0.4 / 25). Phone A/B, 10 FLEURS dev sentences each way
  played with `--file`, 2 reps in ABBA order: chrF tied (0.8 − 0.4, per sentence over 20: +0.7 ±
  8.4, 95% −2.8 to +4.4; VI→EN 39.0 vs 38.1, EN→VI 38.6 vs 40.3). 0.8 ended 13 of 40 turns on its
  end of text against 6 (p = 0.11), cut 1 turn short against 2 and never left a whole EN→VI turn
  in English (0.4 did twice), but wrote more junk words. Heat changed the timing, not the text.
  Files: `/Volumes/data/hibiki-zero-evals/mac/npu-agree/phone/speechlive-sampling-2026-11-25/`.
- **Speech Live's level stage** (2026-11-26): the app keeps the −20 dBFS stage. At +14 dB
  ([the level-stage test](exposure-bias-runs.md#the-level-stage-test)), same clips and sampling:
  EN→VI 32.8 chrF over three runs (29.8, 36.2, 32.3) against 38.6 (37.9, 39.2), all three below
  all four old-level runs of either sampling (p ≈ 0.03; per sentence −3.4, 95% −11.1 to +3.9),
  mostly mistranslation, not English left in; VI→EN 37.4 against 39.0 (three runs, no clear
  change). The gate opened on the same frames (±1 a turn); on the test's 870 s of noise alone it
  opened once (23 frames) and wrote nothing. Files: `level-d27d931/` in the folder above.
- 4 bits is not run yet.
