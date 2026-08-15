# FLEURS Vietnamese–English data

Generate the real-speech training and validation data used by the current SFT
path. Generated WAVs and manifests are intentionally not tracked.

```bash
pip install -r remote_dataset/requirements.txt
for split in train validation test; do
  python remote_dataset/download_fleurs_vi_en.py --split "$split"
done
python finetune/build_pairs.py --splits train validation test
```

This creates `remote_dataset/fleurs_vi_en/`, then deterministic pair manifests
under `finetune/pairs/`, including the `val128.jsonl` selection set.
