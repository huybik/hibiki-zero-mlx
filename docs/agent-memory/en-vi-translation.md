# EN→VI transcript translation

Read when touching `training/prep/translate_en_speech.py` or
`/Volumes/data/datasets/en_speech_vi_text`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Output](#output) ·
[Token budget](#token-budget) · [Model and prompt](#model-and-prompt) · [Running](#running) ·
[How to work](#how-to-work)

## Purpose

Step 2 of the phase-1 pipeline in [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md): every
`en_speech` clip gets a Vietnamese line for voice-transfer TTS to read. Vietnamese is the
model's source stream, so each line must stay aligned with its English audio. The TTS step is
not built.

## Output

`/Volumes/data/datasets/en_speech_vi_text/` holds Common Voice text, so never upload it
([en-speech-corpus](en-speech-corpus.md)).

- `texts.parquet` (`key`, `text`): 2,812,262 unique whitespace-normalised transcripts of rows
  whose clip fits the trainer's 280-frame cap (22.4 s). Validation comes first, and
  Libriheavy is ordered by recording, then segment, so a batch carries the neighbouring book text.
- `rows.parquet` (`split`, `id`, `key`): 3,387,073 rows. The 190,951 rows with longer clips and 3
  rows without letters (`.`) are skipped.
- `translations.jsonl` (`en`, `vi`, `profile`, `usage` per accepted batch) and
  `rejects.jsonl` (`en` and the raw output of rejected batches) are append-only, and `translate`
  resumes from them. Records hold the English text rather than a key, so rebuilding `texts`, for
  example with a different scope, keeps every finished translation.
- `translate.lock` stops a second `translate` run on the same directory.
- `{split}.parquet` (`id`, `text_vi`) is written by `export` once every row of the split is
  translated.
- Validation (20,882 rows) is translated and exported. Train has had only an 18-line test batch.

## Token budget

Measured with Sonnet 5.5 on the validation split:

- Output is about 0.62 tokens per English character, because Vietnamese costs 0.585 tokens per
  character. It is ~88% of the cost at API price weights.
- The full job projects to ~240M output and ~153M input tokens over ~48k calls. Since 1.1M output
  tokens used 23% of one Max plan's 5-hour window, the job needs ~50 such windows.
- Batching is the main saving: one call per line would cost 80% more. The 687-token
  instruction prefix is read from the prompt cache; without it the job costs 2.2% more.
- Line numbers cost 2 tokens per line. Dropping them from the output would save 2.1%, but merged
  or split lines would then pass unnoticed.
- Each call also carries ~316 input tokens that Claude Code adds after the cache breakpoint, so
  16k-character batches would save only 0.7%.
- Telex (+26%) and VIQR (+61%) cost more than Unicode Vietnamese, so ASCII encodings are not
  worth retrying.
- The local API cannot disable thinking. At low effort, thinking appears in ~25% of calls and
  adds under 1% of output.
- What is translated is the remaining big lever. About 22% of characters are in clips likely to
  overflow 280 frames once the English is delayed by U[0, 0.5 × Vietnamese duration], assuming
  the Vietnamese lasts as long as the English. Clips over 15 s hold 44% of the characters.

## Model and prompt

- A blind benchmark compared the models on 100 seeded validation transcripts, one line per
  call, with Opus and Sonnet as judges. It scored adequacy for Opus at 4.59/4.31, Sonnet at
  4.43/4.24 (tied with Opus on short Common Voice lines) and Haiku at 3.82/3.74. Haiku made
  meaning errors and once answered as a chatbot. Sonnet at low effort was chosen.
- The prompt keeps dialogue quotes and the English order of clauses. An earlier wording that
  banned quotation marks made Sonnet move speech tags to the front of the line.
- A batch is accepted only with exactly one `n|` line per input, numbered 1..N. Rejections are
  rare: one of ~460 validation batches over two prompt versions, for a translation broken across
  two lines. Rejected lines are retried in quarter-size batches twice, then alone. A line that
  still fails alone ends the run with an error, and `export` never writes a split with an
  untranslated row.
- The plan's two translation checks mostly flag good output. On validation, the 22 lines
  identical to the English are names, and the 17 lines over twice the source length are correct
  translations of very short lines.

## Running

- The commands are in [training/prep/README.md](../../training/prep/README.md). The script runs on
  the Mac with the conda Python and calls the local Claude API (`~/.local/share/claude-local-api/`).
- 64 client threads keep the server's 48 slots busy and stay under macOS's 128-connection listen
  backlog.
- `translate` needs `--profile` for each subscription it may spend, in order of use. When every
  named profile is out of quota, it sleeps until the earliest reset. Don't name `default`, the
  interactive Claude Code login, unless the user asks.
- Every call turns off the server's response cache, so a retried batch is generated again.
- A 429 without `retry-after` refuses only that request, so the run retries only that batch after
  a minute.
- 5xx replies and dropped connections are retried with backoff (1, 2, 4, 8 and 16 minutes), then
  stop the run. Any other API error stops it at once.
- Ctrl-C ends quota waits and lets in-flight batches finish and be logged. If a hard kill cuts off
  the last line of `translations.jsonl`, the next run stops with that line's number instead of
  guessing.

## How to work

- Before committing pipeline changes or starting a quota-heavy run, test on a small sample and
  report measured alternatives; the user asked for this before the first full run.
- Translate in tranches, as the roadmap validates the pipeline on 1k hours before scaling, and
  check TTS on a tranche before spending more quota.
