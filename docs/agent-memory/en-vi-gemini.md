# EN→VI translation on Gemini

Read when touching the `gemini` step of `training/prep/translate_en_speech.py` or
`/Volumes/data/datasets/en_speech_vi_text/gemini`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Claims and files](#claims-and-files) ·
[Measurements](#measurements) · [Running](#running)

## Purpose

The Claude subscriptions' weekly caps stretch the whole corpus over weeks
([en-vi-translation](en-vi-translation.md)), so the user chose to translate 1,000 hours on each
of two paid Gemini API keys with Gemini 3.8 Flash through the Batch API, at half the interactive
price. Prompt, `n|` line format and acceptance check are the Claude path's.

## Claims and files

- `gemini --hours H` gives each `--key` that has no claim yet H hours of untranslated audio (the
  durations of every row sharing a text), walking back from the end of the translation order:
  Claude works from the front and Gemini from the back. The end of the order is Libriheavy train,
  so no Common Voice text reaches Google while the claims stay within the Libriheavy hours.
- Claims go to `gemini/claims.parquet` (`text`, `key`) under `translate.lock`, so `translate` must
  be stopped. It reads them once at start and skips every claimed text, finished or not.
- Each key runs rounds: one batch job with all its pending claimed lines in 8,000-character
  batches, then its rejected lines in quarter-size batches, then alone; a line rejected alone
  stops the run. `gemini/<KEY>.job.json` holds the round's budget and, while a job runs, its name
  and batches, so a rerun resumes polling. A job expired after 48 h is resubmitted; a failed or
  cancelled one stops the run.
- Accepted batches go to `gemini/translations.jsonl` (`en`, `vi`, `key`, `usage`) and rejected
  output to `gemini/rejects.jsonl`; `export` merges both engines' translation files.
- The SDK retries 429s, 5xx replies and dropped connections 5 times, 1 to 16 minutes apart.

## Measurements

On the 71 production batches that hold 100 random Libriheavy validation lines (2,843 lines,
560k characters, 10.6 h), one job per key:

- Each job finished in 3-4 minutes, and all 71 batches passed the format check.
- `thinking_level` LOW (3.8 Flash has no minimal) used no thinking tokens.
- Gemini counts 0.29 input and 0.33 output tokens per English character; the instructions are
  311 tokens. Its Vietnamese is 1.10× the English characters, Sonnet's 1.00×, which matters for
  the 280-frame cap once voiced.
- At batch prices of $0.375 input and $1.875 output per million tokens (both double on
  2027-01-01), 1,000 hours of Libriheavy cost ~$39.
- Blind judges scored adequacy/fluency (1-5) on the 100 lines: Opus gave Sonnet 4.31/4.12 and
  Gemini 4.18/4.07; Sonnet gave 3.95/3.81 and 4.00/3.77. Gemini's errors are nuance: word
  choice, a plural "you", a stronger or weaker shade than the English.
- A trimmed 120-token prompt saved ~1% of cost but lost 0.06-0.14 adequacy and ~0.16 fluency on
  both judges, so the full prompt stays.
- Batch jobs are limited per project in enqueued tokens: 3M on Tier 1, 400M on Tier 2. A
  1,000-hour job is ~15M input tokens, so it needs Tier 2 or a split into smaller jobs.

## Running

- Load `.env` first (`set -a && . ./.env && set +a`). `--key` names the variables, and every
  record stores the name, never the value.
- When the corpus changes: stop `translate`, run `texts`, start `gemini` (it claims, then
  submits), then restart `translate`.
- One process drives every key and checks its jobs each minute; `gemini.lock` stops a second
  one. `kill` it and run it again to resume.
