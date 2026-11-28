# EN-VI speech dataset

Run from the repository root. Its training caches are built in [training/cache](../cache/README.md).

## The dataset (~10.5k hours)

`/Volumes/data/datasets/en_vi_speech`, laid out as the Hugging Face dataset `anquachdev/common-vi-en`,
holds the English read-speech corpus `en_speech`: Common Voice 27.0 English plus Libriheavy, ~10.5k
hours of 5–22.4 s clips with cased text, speaker and the original mono MP3/Opus bytes, and their
Vietnamese translations ([en-speech-corpus](../../docs/agent-memory/en-speech-corpus.md)).
It was voiced on `huybik/en-vi-speech` and moved, all voiced, to `anquachdev/common-vi-en`, the one
copy used ([en-vi-speech-dataset](../../docs/agent-memory/en-vi-speech-dataset.md#purpose)).

Their Vietnamese speech comes from the TTS step below. The builder that downloaded Common Voice
and Libriheavy is retired; restore `build_en_speech.py` from git (commit `7fde7f5`) to rebuild,
and the publisher's `en_speech` step that moved the corpus in (commit `a236ad5`).

```bash
set -a && . ./.env && set +a && export HF_TOKEN=$HF_TOKEN_AN   # anquachdev's write token
R=/Volumes/data/datasets/en_vi_speech
python -m training.en_vi_dataset.publish card       # README.md with statistics (each upload pass writes it too)
nohup caffeinate -i python -m training.en_vi_dataset.publish upload --follow >> $R/upload.log 2>&1 &
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
nohup caffeinate -i python -m training.en_vi_dataset.tts >> $R/tts.log 2>&1 &   # `kill` stops it; rerun to resume
```

`--shards N` stops after the next N shards with rows to voice, counting those the upload pulls in
meanwhile.
A stop loses only the shards not yet rewritten. Once both machines are done (the other's upload
stopped), voice the retranslated rows again:

```bash
python -m training.en_vi_dataset.tts --texts /Volumes/data/datasets/en_speech_vi_text/retexts.parquet >> $R/tts.log 2>&1
```

## Vietnamese translations of `en_speech`

Every `en_speech` transcript is translated: the Common Voice text by Claude Sonnet 5.5, Libriheavy
by Gemini 3.8 Flash and Muse Spark 1.3 Contributor. The lines are the dataset's `text_vi`; the
working files stay in `/Volumes/data/datasets/en_speech_vi_text`
([en-vi-translation](../../docs/agent-memory/en-vi-translation.md),
[en-vi-gemini](../../docs/agent-memory/en-vi-gemini.md),
[en-vi-muse](../../docs/agent-memory/en-vi-muse.md)). The code, `translate_en_speech.py`, is in git
at commit `364091a`.
