# Vietnamese checkpoint validation

Teacher-forced validation can improve while free-running generation collapses,
so only free-running outputs qualify a checkpoint.

## Deterministic selection run

Evaluate every candidate on the fixed 128-row FLEURS validation subset:

```bash
python finetune/eval.py \
  --device cuda --dtype float32 \
  --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
  --checkpoint <model_stepNNNNNN.safetensors> \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir <run>/eval_val128_stepNNNNNN
```

The evaluator writes `predictions.csv` and `metrics.json`. A checkpoint is
eligible only when all of these gates pass:

| Gate | Requirement |
|---|---|
| Nonempty | At least 122/128 predictions |
| EOS | At least 116/128 predictions |
| Loops | At most 12 repeated-4gram failures |
| Length | Mean prediction/reference word ratio at most 2.0 |

Rank eligible checkpoints by corpus chrF. Nonempty chrF must not fall by more
than one absolute point while corpus chrF rises. `model_best.safetensors` tracks
raw chrF only, so perform this eligibility check before final selection.

## Diagnostics and final test

Teacher-forced diagnostics are available with:

```bash
python finetune/validate.py \
  --device cuda --dtype float32 \
  --cache-dir finetune/cache/validation \
  --checkpoint <checkpoint.safetensors> \
  --batch-size 8 --out-json <run>/teacher_forced.json
```

After selecting without inspecting test outputs, run the same free-running
command once on the full FLEURS validation set and once on FLEURS test. Preserve
the checkpoint/base hashes, repository commit, environment, cache/manifests,
run configuration, logs, predictions, metrics, and selected model/trainer pair.
