# Training data preparation

Run from the repository root. Generated caches land under the gitignored `data/`.

## English read-speech corpus (9,200 hours of 5–15 s clips)

Common Voice Scripted Speech 27.0 English plus Libriheavy, written as parquet
shards (original mono MP3/Opus bytes, cased text, speaker, duration). Only clips of
5–22.4 s are kept, as longer clips can never fit a 280-frame training row. Every recording
must reach 10.5 kHz of audio bandwidth, and train keeps at most 30 hours per speaker
([en-speech-corpus](../../docs/agent-memory/en-speech-corpus.md#quality-filters)). The Common
Voice step needs `MDC_API_KEY` in `.env` and the dataset terms accepted on Mozilla Data
Collective; the Libriheavy step needs `HF_TOKEN`.

```bash
python -m training.prep.build_en_speech cv          # download, verify, shard, delete archive
python -m training.prep.build_en_speech libriheavy  # dev, medium, small, then large files' 5–15 s clips
python -m training.prep.build_en_speech card        # write README.md with statistics
```

The 9,200-hour default leaves room for the ~0.8k-hour PhoMT cache, restored as in the
[pod guide](../../docs/finetune.md), so that training gets ~10k hours of 5–15 s clips.

Output defaults to `/Volumes/data/datasets/en_speech`; its `README.md` documents the
schema, selection and loading at 24 kHz. Never upload it: the Common Voice terms
forbid re-sharing.

## Grounded caches

`training.prep.cache_codes` turns a pair file into SFT cache shards. Each row needs
`id`, `split`, `vi_audio`, `en_audio`, `vi_duration_s`, `en_duration_s`, `text_vi`
and `text_en`, with local audio paths. English text is word-timed by CTC alignment
(`text_timing`), and rows that fail alignment are rejected.

```bash
./.venv/bin/python -m training.prep.cache_codes --pairs PAIRS.jsonl --out-dir data/cache/NAME
```

## Vietnamese translations of `en_speech`

Runs on the Mac with the conda Python and translates with Claude Sonnet 5.5 through the local
Claude API; costs and rules are in
[en-vi-translation](../../docs/agent-memory/en-vi-translation.md).

```bash
python -m training.prep.translate_en_speech texts --min-seconds 5 --max-seconds 15 --hours 1000  # select a tranche
nohup caffeinate -i python -m training.prep.translate_en_speech translate \
  --profile claude-p --profile claude-l --profile default \
  >> /Volumes/data/datasets/en_speech_vi_text/translate.log 2>&1 &   # `kill` stops it cleanly; rerun to resume
python -m training.prep.translate_en_speech export      # {train,validation}.parquet for fully translated splits
```

With the same duration range, a larger `--hours` extends the selection; translations already
made are always kept.

Output defaults to `/Volumes/data/datasets/en_speech_vi_text`. Never upload it: it holds Common
Voice text.
