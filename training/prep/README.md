# Training data preparation

Run from the repository root. Generated caches land under the gitignored `data/`.

## EN-VI speech dataset (~10.8k hours)

`/Volumes/data/datasets/en_vi_speech`, laid out as the Hugging Face dataset `huybik/en-vi-speech`,
holds the English read-speech corpus `en_speech`: Common Voice 27.0 English plus Libriheavy, ~10.5k
hours of 5–22.4 s clips with cased text, speaker and the original mono MP3/Opus bytes, and their
Vietnamese translations ([en-speech-corpus](../../docs/agent-memory/en-speech-corpus.md)). Its
`cv-anquach` shards hold, already voiced, the Common Voice rows of `anquachdev/common-vi-en`'s
earlier version that the corpus lacks ([en-vi-speech-dataset](../../docs/agent-memory/en-vi-speech-dataset.md)).
Once every shard is voiced, the dataset moves to `anquachdev/common-vi-en`
([en-vi-speech-dataset](../../docs/agent-memory/en-vi-speech-dataset.md#purpose)).

Their Vietnamese speech comes from the TTS step below. The builder that downloaded Common Voice
and Libriheavy is retired; restore `build_en_speech.py` from git (commit `7fde7f5`) to rebuild,
and the publisher's `en_speech` step that moved the corpus in (commit `a236ad5`).

```bash
set -a && . ./.env && set +a                            # HF_TOKEN
R=/Volumes/data/datasets/en_vi_speech
python -m training.prep.publish_en_vi_speech anquach    # data/train/cv-anquach-* from the earlier common-vi-en
python -m training.prep.publish_en_vi_speech card       # README.md with statistics (each upload pass writes it too)
nohup caffeinate -i python -m training.prep.publish_en_vi_speech upload --follow >> $R/upload.log 2>&1 &
```

Every step may be stopped (Ctrl-C, `kill`) and resumes when run again; `upload` loses at most
its unfinished commit (~4 GB). Without `--follow` it ends after one pass, once every shard is on
the Hub and every voiced local shard is current there; with it, it looks again every 10 min until every local shard is voiced, here or on
the other machine.

## Vietnamese speech of `en_speech` (TTS)

Runs on the Mac with the conda Python: VieNeu-TTS v3 Turbo in MLX speaks each row's `text_vi` in
the voice of its own English clip, and rewrites each dataset shard with its `vi_audio` filled
([vi-tts](../../docs/agent-memory/vi-tts.md)). It goes top down through the sorted shard paths
while another machine goes bottom up. Run `upload --follow` above beside it: it publishes the
shards as they are voiced and downloads the ones the other machine voiced, which the TTS then
skips.

```bash
R=/Volumes/data/datasets/en_vi_speech
nohup caffeinate -i python -m training.prep.tts_en_speech >> $R/tts.log 2>&1 &   # `kill` stops it; rerun to resume
```

`--shards N` stops after the next N unvoiced shards, counting those the upload pulls in meanwhile.
A stop loses only the shards not yet rewritten. Once both machines are done (the other's upload
stopped), voice the retranslated rows again:

```bash
python -m training.prep.tts_en_speech --texts /Volumes/data/datasets/en_speech_vi_text/retexts.parquet >> $R/tts.log 2>&1
```

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
python -m training.prep.translate_en_speech retranslate --profile claude-p --profile claude-l \
  --profile claude-5x --profile default                  # CJK-script lines again on Claude; retexts.parquet
python -m training.prep.translate_en_speech export      # {train,validation}.parquet for fully translated splits
```

`claim --hours H` gives each key H more hours instead, from the end of the translation order,
Libriheavy only; `gemini --key GEMINI_API_KEY_1 --key GEMINI_API_KEY_2` sends those keys' claims
as Batch API jobs. `translate` skips every claimed text. Its `--concurrency` (default 8) sets how
fast the accounts' quota drains. Rerunning `texts` after the corpus changes keeps every
translation already made; a running `translate` reads the selection and the claims once, so start
it again afterwards. Every run resumes when started again.

Output defaults to `/Volumes/data/datasets/en_speech_vi_text`.
