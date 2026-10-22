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

# Pooled English read speech (~10k hours)

Common Voice Scripted Speech 27.0 English plus Libriheavy, written as parquet
shards (original mono MP3/Opus bytes, cased text, speaker, duration) for Mimi
tokenization. The Common Voice step needs `MDC_API_KEY` in `.env` and the dataset
terms accepted on Mozilla Data Collective.

```bash
python remote_dataset/build_en_speech.py cv          # download, verify, shard, delete archive
python remote_dataset/build_en_speech.py libriheavy  # dev, small, medium, then large files to 10k train hours
python remote_dataset/build_en_speech.py card        # write README.md with statistics
```

Output defaults to `/Volumes/data/datasets/en_speech`; its `README.md` documents
the schema, selection and loading at 24 kHz.
