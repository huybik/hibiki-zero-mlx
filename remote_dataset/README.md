# Remote Dataset Samples

Download French speech test data, run Hibiki-Zero translations, and score the
generated English transcripts.

Run all commands from the repo root.

## Install

```bash
pip install -r remote_dataset/requirements.txt
```

## odunola/french-english-unprocessed

This dataset has `audio`, French `sentence`, and English `english_transcript`
fields. It does not need a separate Common Voice `data_dir`.

## Download Test Data

Fetch 10 short samples:

```bash
python remote_dataset/download_french_english_unprocessed.py --limit 10
```

Fetch 20 short samples:

```bash
python remote_dataset/download_french_english_unprocessed.py --limit 20
```

Fetch 10 samples between 8 and 15 seconds:

```bash
python remote_dataset/download_french_english_unprocessed.py \
  --limit 10 \
  --min-duration 8 \
  --max-duration 15 \
  --out-dir remote_dataset/french_english_unprocessed_8_15s
```

This writes:

```text
remote_dataset/french_english_unprocessed_8_15s/
  fr_0000.wav
  fr_0001.wav
  ...
  manifest.csv
```

`manifest.csv` contains:

```csv
audio_file,duration_s,transcript_fr,translation_en
```

## Run Q4 Translation

```bash
mkdir -p translations/french_english_unprocessed_8_15s_q4

for wav in remote_dataset/french_english_unprocessed_8_15s/fr_*.wav; do
  stem=$(basename "$wav" .wav)
  python main.py "$wav" \
    -o "translations/french_english_unprocessed_8_15s_q4/${stem}_q4.wav" \
    --text-out "translations/french_english_unprocessed_8_15s_q4/${stem}_q4.txt"
done
```

## Run BF16 Translation

Requires `weights/hibiki.bf16.safetensors`. Build it first if needed:

```bash
python scripts/convert_mlx_bf16.py
```

Then translate the dataset:

```bash
mkdir -p translations/french_english_unprocessed_8_15s_bf16

for wav in remote_dataset/french_english_unprocessed_8_15s/fr_*.wav; do
  stem=$(basename "$wav" .wav)
  python scripts/infer_mlx_bf16.py "$wav" \
    -o "translations/french_english_unprocessed_8_15s_bf16/${stem}_bf16.wav" \
    --text-out "translations/french_english_unprocessed_8_15s_bf16/${stem}_bf16.txt"
done
```

## Evaluate Q4

```bash
uv run --with sacrebleu python remote_dataset/evaluate_translation_text.py \
  --manifest remote_dataset/french_english_unprocessed_8_15s/manifest.csv \
  --pred-dir translations/french_english_unprocessed_8_15s_q4 \
  --pattern "{stem}_q4.txt" \
  --out-json translations/french_english_unprocessed_8_15s_q4/metrics_all.json
```

## Evaluate BF16

```bash
uv run --with sacrebleu python remote_dataset/evaluate_translation_text.py \
  --manifest remote_dataset/french_english_unprocessed_8_15s/manifest.csv \
  --pred-dir translations/french_english_unprocessed_8_15s_bf16 \
  --pattern "{stem}_bf16.txt" \
  --out-json translations/french_english_unprocessed_8_15s_bf16/metrics_all.json
```

This reports BLEU, chrF, and word error rate for the generated English text.
