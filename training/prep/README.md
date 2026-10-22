# Training data preparation

Run from the repository root with the environment from `./training/h100.sh setup`.
Everything generated lands under the gitignored `data/`.

## FLEURS Vietnamese–English

```bash
for split in train validation test; do
  ./.venv/bin/python -m training.prep.download_fleurs_vi_en --split "$split"
done
./.venv/bin/python -m training.prep.build_pairs --splits train validation test
```

This creates `data/fleurs_vi_en/<split>/`, then deterministic pair manifests under
`data/pairs/`, including the `val128.jsonl` selection set.

## Grounded-v2 caches

`./training/h100.sh cache-grounded` streams PhoMT into `data/cache/phomt_grounded_v2/`
(`cache_phomt_stream`) and caches the FLEURS train and validation pairs into
`data/cache/{train,validation}_grounded_v2/` (`cache_codes`). Both CTC-time the English
text with `text_timing`. `publish_grounded_cache` validates and uploads finished caches;
restoring them on a pod is in [docs/finetune.md](../../docs/finetune.md).
