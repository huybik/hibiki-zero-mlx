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

## Diagnostics and final test

For the masked text/source-only pilot, teacher-forced validation must also pass
`--audio-loss-weight 0 --mask-target-audio-input`. After selecting without
inspecting test output, run the paired evaluator once on full FLEURS validation
and once on test. Preserve hashes, commit, environment, manifests, run config,
logs, paired artifacts, selected model/trainer pair, and derangement mapping.
