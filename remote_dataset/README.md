# FLEURS Vietnamese–English data

Generate the real-speech training and validation data used by the current SFT
path with the environment from `./finetune/h100.sh setup`. Generated WAVs and
manifests are intentionally not tracked.

```bash
for split in train validation test; do
  ./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split "$split"
done
./.venv/bin/python finetune/build_pairs.py --splits train validation test
```

This creates `remote_dataset/fleurs_vi_en/`, then deterministic pair manifests
under `finetune/pairs/`, including the `val128.jsonl` selection set.
