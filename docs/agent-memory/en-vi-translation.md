# EN→VI transcript translation

Read when touching `training/prep/translate_en_speech.py` or
`/Volumes/data/datasets/en_speech_vi_text`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Output](#output) ·
[Token budget](#token-budget) · [Model and prompt](#model-and-prompt) · [Running](#running) ·
[How to work](#how-to-work)

## Purpose

Step 2 of phase 1 in [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md): each `en_speech`
clip gets a Vietnamese line for voice-transfer TTS to read. Vietnamese is the model's source
stream, so each line must stay aligned with its English audio. The TTS step is not built.

## Output

`/Volumes/data/datasets/en_speech_vi_text/` holds Common Voice text, so never upload it
([en-speech-corpus](en-speech-corpus.md)).

- `texts` writes `texts.parquet` (`text`, the unique transcripts in translation order) and
  `rows.parquet` (`split`, `id`, `text`) for every corpus row with a letter in its transcript;
  the corpus keeps only 5–22.4 s clips. Validation comes first, then each Libriheavy recording in
  segment order, so a batch carries the neighbouring book text. `texts` stops if a clip is over
  the 22.4 s (280-frame) cap, and must not run while a build is writing `en_speech`.
- The selection on disk is still the first tranche (5-15 s clips, 1,000 train hours plus
  validation, 410,618 texts) while `en_speech` is pruned and topped up; rerun `texts` after.
- `translations.jsonl` (`en`, `vi`, `profile`, `usage` per accepted batch) and `rejects.jsonl`
  (`en`, raw output) are append-only, and `translate` resumes from them by English text, so
  rebuilding `texts` keeps every finished translation. `translate.lock` stops a second run on
  the directory; `export` takes it too.
- `export` writes `{split}.parquet` (`id`, `text_vi`) once every row of the split is translated,
  and removes it while not. It reports the plan's two gates (identical to the English, over
  twice its length) but drops nothing.

## Token budget

Measured with Sonnet 5.5:

- Output is 0.60-0.62 tokens per English character (Vietnamese costs 0.585 per character) and
  ~88% of the cost at API price weights.
- One 5-hour window held ~4-5M output tokens on `claude-p` and `claude-l`, ~3.2M on `claude-5x`
  (Max 5x) and ~11M on `default`. A full window used ~5% of `claude-5x`'s week, and `default`'s
  week rose about 1 point per 9 of its window, so weekly limits bound a long run. `claude-p`
  was at 87% of its week on 2026-10-23.
- Mid-rebuild on 2026-10-23, the 5–22.4 s corpus held 384M characters in 2.48M unique texts,
  171M of them (~104M output tokens) in 15–22.4 s clips. The Libriheavy top-up to ~9k hours of
  5-15 s clips adds ~220M, so the whole corpus needs ~370M output tokens.
- Measured levers, as changes to total cost: one call per line instead of batches +80%; no
  prompt cache for the 687-token instruction prefix +2.2%; no output line numbers −2.1%, but
  merged or split lines would pass unnoticed; 16k-character batches −0.7%, since Claude Code
  adds ~316 uncached input tokens per call; Telex or VIQR output +26% or +61%.
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
  rare (11 of ~3,900 batches by 2026-10-23): most break a translation across lines, and a few
  answer with a stray `echo skip` tool call. Rejected lines are retried in quarter-size batches
  down to ~500 characters, then alone; a line that still fails alone ends the run.
- The gates mostly flag good output: on the validation texts, the 22 lines identical to the
  English are names, and the 17 over twice the source length translate very short lines.

## Running

- Commands are in [training/prep/README.md](../../training/prep/README.md). The script runs on the
  Mac with the conda Python against the local Claude API (`~/.local/share/claude-local-api/`).
- `--concurrency` (default 8, at most the server's 48) sets the calls in flight, and quota
  drains in proportion: 8 calls spend ~3.6M output tokens an hour, so a 5-hour window lasts
  from ~1 hour on `claude-5x` to ~3 on `default`; 64 calls spent ~20M an hour.
- `--profile` names each subscription to spend, in order of use; the run uses `claude-p`,
  `claude-l`, `claude-5x`, then `default`. When all are out of quota it waits for the earliest
  reset, checking the clock every minute so it also resumes after the Mac sleeps. Name
  `default`, the interactive Claude Code login, only if the user asks. Every named profile must
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
- The user chose on 2026-10-23 to translate the whole corpus to the end, with no limit beyond
  the corpus's own 5–22.4 s, after a first 1k-hour tranche of 5-15 s clips. They asked for few
  calls in flight so the accounts drain slowly, and for the run to resume by itself after quota
  resets.
