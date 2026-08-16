# Vietnamese checkpoint validation

Teacher-forced validation is diagnostic. Only paired free-running evaluation
can qualify or promote a checkpoint.

## Calibrate the evaluator before the pilot

Use the same 128-row manifests, batch size, seed, top-k settings, and text
temperature for every comparison. First run the Vietnamese base checkpoint, the
old phase-1 checkpoint, and a healthy French-to-English base control. Build the
frozen French manifest with:

```bash
python remote_dataset/download_covost2.py --limit 128
```

Run the healthy French control once with `--text-temp 0` and once with the
production setting `--text-temp 0.4`. Greedy decoding previously over-collapsed;
use fixed-seed temperature 0.4 for calibration and checkpoint selection unless
the control evidence disproves it. The French invocation adds:

```text
--pairs remote_dataset/covost2_fr_en_control/manifest.csv
--source-column fr_audio --duration-column fr_duration_s
--reference-column text_en --id-column id
```

The paired controls lock fixed-seed temperature 0.4 and minimum source gaps of
`HIBIKI_MIN_SOURCE_BLEU_GAP=1.0` and `HIBIKI_MIN_SOURCE_CHRF_GAP=5.0`.
Grounded-v2 preflight, smoke, training, and resume reject unspecified values.
On 128 rows, healthy French passed health with gaps 23.08 BLEU / 38.80 chrF;
Vietnamese base failed health with gaps -0.07 / 1.23; phase-1 passed health but
had gaps only 0.01 / 1.03. Thus phase-1's 19.57 correct-source chrF was mostly
target-side modeling, not Vietnamese routing. In a 10,000-resample paired
bootstrap, the largest Vietnamese-null upper bounds were 0.04 BLEU and 2.32
chrF, while healthy French lower bounds were 19.51 and 35.56. Greedy French
failed health; temperature 0.4 passed.

## Paired selection run

```bash
python finetune/eval.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --checkpoint <model_stepNNNNNN.safetensors> \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0.4 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir <run>/eval_val128_stepNNNNNN
```

The evaluator freezes a SHA-256-verified duration-matched derangement, snapshots
and restores Python/NumPy/Torch CPU/CUDA/MPS RNG, and reseeds every correct and
shuffled batch identically. It writes condition artifacts under `correct/` and
`shuffled/`, plus a consolidated one-row-per-target `predictions.csv` and a
paired `metrics.json`. Metrics include BLEU, chrF, nonempty BLEU/chrF, all four
correct-minus-shuffled gaps, and health eligibility.

A checkpoint is promotable only when correct-source generation passes all
val128 health gates (at least 122 nonempty, at least 116 EOS, at most 12 repeated
4-gram failures, mean length ratio at most 2.0) and both calibrated source-gap
minimums. Rank qualified checkpoints lexicographically by correct-source
`(BLEU, chrF)`. `best.json` contains the complete paired metrics and thresholds;
the recovery sync preserves it unchanged.

## Corrected ordinary-pilot result

The exact-cohort ordinary pilot produced no promotable checkpoint:

| Step | Nonempty | EOS | Repeat-4 failures | BLEU | chrF | BLEU gap | chrF gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 120 | 13 | 0.218 | 14.62 | 0.111 | 1.144 |
| 250 | 52 | 122 | 14 | 0.064 | 4.95 | 0.011 | 0.208 |
| 500 | 128 | 114 | 26 | 0.072 | 9.12 | -0.023 | -0.587 |
| 750 | 105 | 119 | 19 | 0.102 | 8.26 | -0.029 | 0.446 |
| 1,000 | 126 | 116 | 24 | 0.069 | 9.86 | -0.071 | 0.225 |

The final shuffled BLEU was 0.140, above correct-source BLEU 0.069. All paired
artifacts and the final model/trainer pair are preserved under the model repo's
isolated `grounded_v2_pilot/` prefix. This result triggers the exact-membership
75--100% delay repeat; it does not justify full training.

## High-delay pilot result

The exact same ordered 50,000-entry cohort was rebuilt with deterministic
75--100% source-duration delay. Training used batch 8 / accumulation 2 under a
480-frame cap; the observed maximum was 479. Complete teacher-forced validation
used batch 1 under a separate 704-frame cap and observed a 701-frame maximum.
The worst-case smoke peaked at 89.9/93.6 GiB and passed save/eval/resume.

| Step | Nonempty | EOS | Repeat-4 failures | Length ratio | BLEU | chrF | BLEU gap | chrF gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 120 | 13 | 1.06 | 0.218 | 14.62 | 0.111 | 1.144 |
| 250 | 128 | 64 | 85 | 4.61 | 0.024 | 11.74 | -0.012 | 1.249 |
| 500 | 128 | 34 | 109 | 7.38 | 0.024 | 9.48 | 0.008 | 0.393 |
| 750 | 128 | 41 | 99 | 6.16 | 0.032 | 9.68 | 0.014 | 0.523 |
| 1,000 | 128 | 31 | 111 | 7.03 | 0.033 | 9.34 | 0.008 | 0.725 |

No checkpoint was healthy or source-dependent. Increasing delay made output
termination and repetition materially worse without moving the calibrated
source gaps. This rejects delay alone as the cause and triggers an isolated
duration-matched shuffled-source margin-loss pilot; it still does not justify
full training.

For that pilot, teacher-forced logs record contrastive margin loss,
shuffled-minus-correct English-content NLL, and active-margin fraction. These
verify that the intervention is operating but do not qualify a checkpoint.
Promotion still requires free-running correct-source health plus the calibrated
1.0 BLEU and 5.0 chrF paired gaps, ranked by correct-source `(BLEU, chrF)`.

## Diagnostics and final test

For the masked text/source-only pilot, teacher-forced validation must also pass
`--audio-loss-weight 0 --mask-target-audio-input`. After selecting without
inspecting test output, run the paired evaluator once on full FLEURS validation
and once on test. Preserve hashes, commit, environment, manifests, run config,
logs, paired artifacts, selected model/trainer pair, and derangement mapping.
