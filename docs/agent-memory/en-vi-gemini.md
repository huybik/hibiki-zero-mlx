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
That share finished in 30 minutes for ~$75. The user then chose not to pay ~$300 more for the
rest; the remaining Libriheavy went to Muse instead ([en-vi-muse](en-vi-muse.md)).

## Claims and files

- `claim --hours H` gives each `--key` H more hours of untranslated audio (the durations of every
  row sharing a text), or all of it without `--hours`, walking back from the end of the
  translation order: Claude works from the front, the paid engines from the back. It skips every
  text with a Common Voice row, so no Common Voice text reaches Google or Meta.
- Claims go to `claims.parquet` (`text`, `key`) under `translate.lock` and `claims.lock`, so
  `translate` must be stopped. It reads them once at start and skips every claimed text.
- Each key runs rounds: one batch job with all its pending claimed lines in 8,000-character
  batches, then its rejected lines in quarter-size batches, then alone. Lines that fail alone are
  removed from the claims, so `translate` takes them when started again. A retry round holding
  over 10% of a key's lines stops the run instead, since that is a systematic failure; delete the
  key's `job.json` to send them again at full size.
- `gemini/<KEY>.job.json` holds the round's budget, then its batches and a unique display name
  just before the job is created, then the job's name. A rerun resumes polling, or finds a job
  that a killed run created but did not record, so it is never created and paid for twice; job
  creation is not idempotent, so it is never retried. A job expired after 48 h is resubmitted; a
  failed or cancelled one stops the run, and a rerun resubmits it.
- Accepted batches go to `gemini/translations.jsonl` (`en`, `vi`, `key`, `usage`) and rejected
  output to `gemini/rejects.jsonl`; `export` merges every engine's translation file.
- Other API calls retry 429s, 5xx replies and connection errors 5 times, 1 to 16 minutes apart,
  silently; any other error stops the run, and a rerun resumes.

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
  1,000-hour job (~6,650 batches, ~52M characters, ~15M input tokens) was accepted on both keys,
  so they are above Tier 1.

The full 2 × 1,000-hour run (513,727 texts, 105M characters):

- Both first-round jobs (~6,650 batches each) finished within 6 and 20 minutes; with three retry
  rounds per key the whole run took 30 minutes. First rounds accepted 99.9% of lines.
- Tokens: 30.1M input and 34.0M output, so ~$75, or ~$0.72 per million English characters.
- Rejections: 10 empty replies (`finishReason` absent, blocked), 1 `PROHIBITED_CONTENT` and 1
  `RECITATION`, on Freud-style passages and verse; one such line failed alone and went back to
  `translate`.
- Gates: 30 lines identical to the English and 8 over twice its length; the 35 long lines
  without Vietnamese diacritics are Latin, name lists and addresses kept as written.

## Running

- Load `.env` first (`set -a && . ./.env && set +a`). `--key` names the variables, and every
  record stores the name, never the value.
- When the corpus changes: stop `translate`, run `texts`, then `claim`, then start `gemini` and
  `translate`.
- One process drives every key and checks its jobs each minute; `claims.lock` stops a second
  one, and a `claim` or `muse` run. `kill` it and run it again to resume.
