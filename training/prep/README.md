# Training data preparation

Run from the repository root. Generated caches land under the gitignored `data/`.

## English read-speech corpus (~10.5k hours)

Common Voice Scripted Speech 27.0 English plus Libriheavy, written as parquet
shards (original mono MP3/Opus bytes, cased text, speaker, duration). Only clips of
5–22.4 s are kept, as longer clips can never fit a 280-frame training row. Every recording
must reach 10.5 kHz of audio bandwidth, and train keeps at most 30 hours per speaker
([en-speech-corpus](../../docs/agent-memory/en-speech-corpus.md#quality-filters)). The Common
Voice step needs `MDC_API_KEY` in `.env` and the dataset terms accepted on Mozilla Data
Collective; the Libriheavy step needs `HF_TOKEN`.

```bash
python -m training.prep.build_en_speech cv          # download, verify, shard, delete archive
python -m training.prep.build_en_speech libriheavy  # dev, medium, small, then large files to --target-hours
python -m training.prep.build_en_speech card        # write README.md with statistics
```

The 9,200-hour default leaves room for the ~0.8k-hour PhoMT cache, restored as in the
[pod guide](../../docs/finetune.md), so that training gets ~10k hours.

Output defaults to `/Volumes/data/datasets/en_speech`; its `README.md` documents the
schema, selection and loading at 24 kHz.

## Grounded caches

`training.prep.cache_codes` turns a pair file into SFT cache shards. Each row needs
`id`, `split`, `vi_audio`, `en_audio`, `vi_duration_s`, `en_duration_s`, `text_vi`
and `text_en`, with local audio paths. English text is word-timed by CTC alignment
(`text_timing`), and rows that fail alignment are rejected.

```bash
./.venv/bin/python -m training.prep.cache_codes --pairs PAIRS.jsonl --out-dir data/cache/NAME
```

## Vietnamese translations of `en_speech`

Runs on the Mac with the conda Python. Claude Sonnet 5.5 translates through the local Claude API;
the Libriheavy texts claimed for a paid key go to Gemini 3.8 Flash through the Gemini Batch API
or to Muse Spark 1.3 Contributor through OpenRouter. Costs and rules are in
[en-vi-translation](../../docs/agent-memory/en-vi-translation.md),
[en-vi-gemini](../../docs/agent-memory/en-vi-gemini.md) and
[en-vi-muse](../../docs/agent-memory/en-vi-muse.md).

```bash
python -m training.prep.translate_en_speech texts       # select every en_speech row; not during a prune
set -a && . ./.env && set +a                            # the Gemini and OpenRouter keys
python -m training.prep.translate_en_speech claim --key OPEN_ROUTER  # all untranslated Libriheavy; while translate is stopped
nohup caffeinate -i python -m training.prep.translate_en_speech muse --key OPEN_ROUTER --concurrency 96 \
  >> /Volumes/data/datasets/en_speech_vi_text/muse.log 2>&1 &        # `kill` stops it cleanly; rerun to resume
nohup caffeinate -i python -m training.prep.translate_en_speech translate \
  --profile claude-p --profile claude-l --profile claude-5x --profile default \
  >> /Volumes/data/datasets/en_speech_vi_text/translate.log 2>&1 &   # `kill` stops it cleanly; rerun to resume
python -m training.prep.translate_en_speech export      # {train,validation}.parquet for fully translated splits
```

`claim --hours H` gives each key H more hours instead, from the end of the translation order,
Libriheavy only; `gemini --key GEMINI_API_KEY_1 --key GEMINI_API_KEY_2` sends those keys' claims
as Batch API jobs. `translate` skips every claimed text. Its `--concurrency` (default 8) sets how
fast the accounts' quota drains. Rerunning `texts` after the corpus changes keeps every
translation already made; a running `translate` reads the selection and the claims once, so start
it again afterwards. Every run resumes when started again.

Output defaults to `/Volumes/data/datasets/en_speech_vi_text`.

## EN-VI speech dataset on Hugging Face

`training.prep.publish_en_vi_speech` publishes `huybik/en-vi-speech`: every `en_speech` row with
its Vietnamese text (Vietnamese speech to come from TTS), plus the PhoMT cache rows with 5–22.4 s
English clips, with both voices as FLAC
([en-vi-speech-dataset](../../docs/agent-memory/en-vi-speech-dataset.md)).

```bash
set -a && . ./.env && set +a                            # HF_TOKEN
mkdir -p /Volumes/data/datasets/en_vi_speech_upload     # staging and logs
nohup caffeinate -i python -m training.prep.publish_en_vi_speech en_speech \
  >> /Volumes/data/datasets/en_vi_speech_upload/en_speech.log 2>&1 &
nohup caffeinate -i python -m training.prep.publish_en_vi_speech phomt \
  >> /Volumes/data/datasets/en_vi_speech_upload/phomt.log 2>&1 &
python -m training.prep.publish_en_vi_speech card       # once both have finished
```

Each step commits shards in batches and skips those already in the repo, so rerun to resume.
