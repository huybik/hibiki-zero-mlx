# EN→VI transcript translation

Read when touching `training/prep/translate_en_speech.py` or
`/Volumes/data/datasets/en_speech_vi_text`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Output](#output) ·
[Token budget](#token-budget) · [Model and prompt](#model-and-prompt) · [Running](#running) ·
[How to work](#how-to-work)

## Purpose

Step 2 of phase 1 in [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md): each `en_speech`
clip gets a Vietnamese line for voice-transfer TTS to read. Vietnamese is the model's source
stream, so each line must stay aligned with its English audio. The TTS step is [vi-tts](vi-tts.md).

## Output

`/Volumes/data/datasets/en_speech_vi_text/` stays local; the exported translations are published
with the audio in `anquachdev/common-vi-en` ([en-vi-speech-dataset](en-vi-speech-dataset.md)).

- `texts` reads the shards of the EN-VI dataset (`--src`) and writes `texts.parquet` (`text`, the unique transcripts in translation order)
  and `rows.parquet` (`split`, `id`, `text`, `duration`) for every row with a letter in its
  transcript, so the clip range is the corpus's own ([en-speech-corpus](en-speech-corpus.md)).
  Validation comes first, then each Libriheavy recording in segment order, so a batch carries
  the neighbouring book text. `texts` stops if a clip is over the 22.4 s (280-frame) cap.
  Dataset shards are written by rename, so `texts` may read a snapshot while shards are only
  added. A running `translate` reads the selection and the claims once, so start it again after
  `texts` or a claim.
- The selection on disk is the final corpus (PR #25): 3,388,263 texts (526M characters) in
  3,490,088 rows. Every text is translated: Sonnet 616,834 (all Common Voice), Gemini 375,868 and
  Muse 2,395,561, with no line left over for `translate`. `export` wrote `train.parquet`
  (3,480,566 rows) and `validation.parquet` (9,522) on 2026-10-23; its gates flagged 267 lines
  identical to the English and 167 over twice its length.
- `translations.jsonl` (`en`, `vi`, `profile`, `usage` per accepted batch) and `rejects.jsonl`
  (`en`, raw output) are append-only, and `translate` resumes from them by English text, so
  rebuilding `texts` keeps every finished translation. `translate.lock` stops a second run on
  the directory; `export` takes it and `claims.lock`.
- `translate` skips texts claimed for Gemini or Muse, which write under `gemini/` and `muse/`
  ([en-vi-gemini](en-vi-gemini.md), [en-vi-muse](en-vi-muse.md)); `export` merges all three.
  Sonnet translates the Common Voice text.
- `export` writes `{split}.parquet` (`id`, `text_vi`) once every row of the split is translated,
  and removes it while not. It reports the plan's two gates (identical to the English, over
  twice its length) but drops nothing.

## Token budget

Measured with Sonnet 5.5:

- Output is 0.60-0.62 tokens per English character (Vietnamese costs 0.585 per character) and
  ~88% of the cost at API price weights.
- One 5-hour window held ~4-5M output tokens on `claude-p` and `claude-l`, ~3.2M on `claude-5x`
  (Max 5x) and ~11M on `default`. A full window used ~5% of `claude-5x`'s week and ~11% of
  `default`'s, so their weeks hold ~64M and ~100M. At 8 calls (~600M a week) the weekly caps,
  not the windows, bound the run: every named account, `default` included, reaches its weekly
  cap each week until the corpus is done.
- Measured levers, as changes to total cost: one call per line +80%; no prompt cache for the
  687-token instructions +2.2% (they are ~0.3% of cost cached, so trimming them saves nothing);
  no output line numbers −2.1%, but merged or split lines would pass unnoticed; 16k-character
  batches −0.7%; Telex or VIQR output +26% or +61%.
- Thinking can't be disabled through the local API; at low effort it shows in ~25% of calls but
  adds under 1% of output.
- With the English delayed by U[0, 0.5 × Vietnamese duration] and the Vietnamese as long as the
  English, ~22% of characters are in rows likely to overflow 280 frames; clips over 15 s hold
  44% of them.

## Model and prompt

- In a blind benchmark on 100 validation transcripts, mean adequacy judged by Opus/Sonnet was
  4.59/4.31 for Opus, 4.43/4.24 for Sonnet (tied on short Common Voice lines) and 3.82/3.74 for
  Haiku, which made meaning errors. Sonnet at low effort was chosen; Opus 5.5 costs twice
  Sonnet 5.5 per token.
- The prompt keeps dialogue quotes and the English order of clauses. A wording that banned
  quotation marks made Sonnet move speech tags to the front of the line.
- A batch is accepted only with exactly one `n|` line per input, numbered 1..N. Rejections are
  rare (about 1 batch in 350): most break a translation across lines, and a few
  answer with a stray `echo skip` tool call. Rejected lines are retried in quarter-size batches
  down to ~500 characters, then alone; a line that still fails alone ends the run.
- One Common Voice sentence did that: Sonnet returns `refusal` and an empty body for "Aqueous
  cyclodextrin solutions can generate aerosols in particle size ranges suitable for pulmonary
  deposition." Its clip was deleted from the corpus ([en-speech-corpus](en-speech-corpus.md)).
- The gates mostly flag good output: on the validation texts, the 22 lines identical to the
  English are names, and the 17 over twice the source length translate very short lines.

## Running

- Commands are in [training/prep/README.md](../../training/prep/README.md). The script runs on the
  Mac with the conda Python against the local Claude API (`~/.local/share/claude-local-api/`).
- `--concurrency` (default 8, at most the server's 48) sets the calls in flight, and quota
  drains in proportion: 8 calls spend ~3.6M output tokens an hour, so a 5-hour window lasts
  from ~1 hour on `claude-5x` to ~3 on `default`.
- `--profile` names each subscription to spend, in order of use; the run uses `claude-p`,
  `claude-l`, `claude-5x`, then `default`. When all are out of quota it waits for the earliest
  reset, checking the clock every minute so it also resumes after the Mac sleeps. `default` is
  the `~/.claude` login; the user works interactively on another account, so the run may spend
  it like the others, weekly cap included. Every named profile must
  have extra usage off (`/v1/health` shows `overageStatus`), or the server keeps serving it and
  bills.
- Run long jobs as `nohup caffeinate -i python …`, logging to `translate.log` in the output
  directory. Calls turn off the server's response cache, so a retried batch is generated again.
- 5xx replies, dropped connections and a 429 without `retry-after` are retried after 1, 2, 4, 8
  and 16 minutes, then stop the run; any other API error stops it at once.
- Ctrl-C, or `kill <pid>` for a background run, lets in-flight batches finish and be logged. A
  run refuses to start when a hard kill has cut off the last line of `translations.jsonl`.
- Restarting the local API (after editing its `profiles.json`) cancels running calls, so stop the
  run first. The restart forgets which profiles are spent, but each refuses its next call with
  `retry-after`, so a new run skips it again.

## How to work

- Before committing pipeline changes or starting a quota-heavy run, test a small sample and
  report measured alternatives, as the user asked.
- The user chose to translate the whole corpus, with no limit beyond the corpus's own clip
  range, with few calls in flight so the accounts drain slowly, and with the run resuming by
  itself after quota resets.
