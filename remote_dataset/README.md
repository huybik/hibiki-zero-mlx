# Remote Dataset Samples

Download French speech test data, run Hibiki-Zero translations, and score the
generated English transcripts.

Run all commands from the repo root.

## Install

```bash
pip install -r remote_dataset/requirements.txt
```

## CoVoST 2 (recommended benchmark)

`fixie-ai/covost2` is a CoVoST 2 mirror with the Common Voice audio bundled in
(no separate `data_dir`), `fr_en` config, `validation`/`test` splits, human
English references. This is the dataset to compare against published numbers
(Whisper / SeamlessM4T report CoVoST 2 fr→en BLEU in the mid-30s, offline).

If your shell points `HF_HOME` at an unwritable path, override it:
`export HF_HOME="$(pwd)/.hf_cache"`.

```bash
# 1. download N test samples (decoded to wav + manifest.csv)
python remote_dataset/download_covost2.py --limit 30

# 2. translate them with the q4 fast MLX path (LM loaded once)
python remote_dataset/run_batch.py \
  --manifest remote_dataset/covost2_fr_en_test/manifest.csv \
  --out-dir translations/covost2_fr_en_test_q4 --suffix q4

# 3. score BLEU / chrF / WER
python remote_dataset/evaluate_translation_text.py \
  --manifest remote_dataset/covost2_fr_en_test/manifest.csv \
  --pred-dir translations/covost2_fr_en_test_q4 \
  --pattern "{stem}_q4.txt" \
  --out-json translations/covost2_fr_en_test_q4/metrics_all.json
```

Notes:
- Hibiki is *simultaneous* (~6 s lag), so it scores below offline systems; q4
  costs a little more. `run_batch.py` flushes the lag tail and early-stops once
  the model goes quiet (`--tail-s`, default 8 s) — without the flush, short clips
  get truncated and BLEU collapses.
- WER is reported but is a poor MT metric (synonyms/word-order count as errors);
  read BLEU/chrF. Use a larger `--limit` (100+) for a low-variance number.
