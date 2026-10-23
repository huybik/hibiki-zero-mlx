# Training data preparation

Run from the repository root. Generated caches land under the gitignored `data/`.

## EN-VI speech dataset (~10.5k hours)

`/Volumes/data/datasets/en_vi_speech`, laid out as the Hugging Face dataset `huybik/en-vi-speech`,
holds the English read-speech corpus `en_speech`: Common Voice 27.0 English plus Libriheavy, ~10.5k
hours of 5–22.4 s clips with cased text, speaker and the original mono MP3/Opus bytes, and their
Vietnamese translations ([en-speech-corpus](../../docs/agent-memory/en-speech-corpus.md)).

Their Vietnamese speech comes from the TTS step below. The `en_speech` step
moved the finished corpus in, shard by shard. The builder that downloaded Common Voice and
Libriheavy is retired; restore `build_en_speech.py` from git (commit `7fde7f5`) to rebuild.

```bash
set -a && . ./.env && set +a                            # HF_TOKEN
R=/Volumes/data/datasets/en_vi_speech && mkdir -p $R
nohup caffeinate -i python -m training.prep.publish_en_vi_speech en_speech >> $R/en_speech.log 2>&1 &
python -m training.prep.publish_en_vi_speech card       # README.md with statistics
nohup caffeinate -i python -m training.prep.publish_en_vi_speech upload >> $R/upload.log 2>&1 &  # rerun after adding shards
```

Every step may be stopped (Ctrl-C, `kill`) and resumes when run again; `upload` loses at most
its unfinished commit (~4 GB).

## Vietnamese speech of `en_speech` (TTS)

Runs on the Mac with the conda Python: VieNeu-TTS v3 Turbo in MLX speaks each row's `text_vi` in
the voice of its own English clip, into one Opus shard per dataset shard
([vi-tts](../../docs/agent-memory/vi-tts.md)).

```bash
O=/Volumes/data/datasets/en_speech_vi_tts && mkdir -p $O
nohup caffeinate -i python -m training.prep.tts_en_speech >> $O/tts.log 2>&1 &   # `kill` stops it; rerun to resume
```

`--shards N` stops after N shards. A stop loses only the shards not yet written.

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
python -m training.prep.translate_en_speech texts       # select the en_speech rows of the EN-VI dataset
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
