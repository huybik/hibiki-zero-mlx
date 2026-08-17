# Direct-run validation

The production loop uses deterministic teacher-forced validation on the raw
grounded-v2 validation cache. Free-running evaluation is optional and uses only
the correct Vietnamese source. There is no shuffled validation, source
derangement, contrastive metric, or source-gap promotion gate.

## Teacher-forced validation

Training validates every 9,000 steps and at step 224,725. The frozen receipt is:

- 138 eligible rows at raw `max_frames=280`;
- observed maximum 277 frames;
- batch 4;
- deterministic length order with `shuffle=False`;
- target-audio teacher forcing and audio CE weight `1.0`.

Each `val_log.jsonl` row records total, audio, text, content-text, and PAD-text
losses, token counts, accuracies, and maximum frame sizes. During the run verify:

- all values are finite;
- `samples` is 138 for a full validation pass;
- `audio_tokens` and `text_tokens` are positive;
- `max_frames` is 277;
- training remains at physical B16, fixed LR `1e-6`, and audio weight `1.0`.

Teacher-forced total loss is the best-checkpoint metric. Whenever a 9,000-step
validation improves it, the trainer replaces the local best model and the sync
worker publishes it under `best/`. The final five-epoch checkpoint is also
validated and eligible, while the newest two recovery pairs remain separate.

## Optional correct-source free-running evaluation

Materialize FLEURS validation audio once:

```bash
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split validation
```

Then evaluate a checkpoint directly against its Vietnamese inputs:

```bash
step=NNNNNN
run_dir=finetune/runs/vi_grounded_v2_full_direct_voice_5epoch
./.venv/bin/python finetune/eval.py \
  --device cuda --dtype bfloat16 \
  --checkpoint "$run_dir/model_step${step}.safetensors" \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 \
  --audio-temp 0.8 --text-temp 0.4 \
  --top-k 250 --top-k-text 250 \
  --stop-on-eos --seed 42 \
  --out-dir "$run_dir/eval_step${step}"
```

This writes generated audio, text sidecars, `predictions.csv`, and
`metrics.json`. Add `--text-only` for a faster text diagnostic that skips WAV
decode/write. Track nonempty output, EOS, repeated 4-grams, length, BLEU, chrF,
and WER, and listen to generated speech when evaluating the audio path.

Do not run a shuffled-source condition or use a correct-minus-shuffled score.
This optional evaluation does not change training order, checkpoints, or the
five-epoch stop.

## Final integrity check

At completion require:

- `model_step224725.safetensors` and `trainer_step224725.pt` form a complete
  pair;
- `run_config.json` records upstream initialization, target-audio teacher
  forcing, audio CE, no transform, B16/accum1, both 280-frame caps, fixed LR and
  AdamW settings, 719,120 rows per epoch, 224,725 steps, and validation shuffle false;
- `sample_manifest.jsonl` and `full_data_receipt.json` match the frozen receipt;
- the remote `grounded_v2_full_direct_voice_5epoch/` prefix contains the final pair or
  newest two recovery pairs, the promoted best model, run metadata, and final logs.

Any integrity mismatch is a stop condition, not permission to modify the
receipt. Recover the same run as documented in [finetune.md](finetune.md).
