# Vietnamese full-model SFT

The only supported training path is base-start, full-model Vietnamese-to-English
SFT on CUDA. LoRA, warm-start, replay, variable-batch, and campaign experiment
paths have been removed.

Core files:

- `train.py`: training, exact full-model checkpoints, resume, and periodic validation.
- `eval.py`: free-running text/audio evaluation and BLEU/chrF/WER metrics.
- `validate.py`: teacher-forced diagnostics.
- `common.py`: shared dataset, loss, checkpoint, generation, and metric logic.
- `cache_codes.py`: FLEURS pair audio to cached Mimi/text codes.
- `cache_phomt_stream.py`: published PhoMT parquet to cached codes without staging WAVs.
- `build_pairs.py`: FLEURS manifests to deterministic pair files and val subsets.
- `hf_sync.py`: rolling recovery pairs and best-model backup under `full_run/`
  in the public checkpoint model repo.

See [the mechanics](../docs/finetune.md), [training recipe](../docs/training_plan.md),
and [validation contract](../docs/validation_plan.md).
