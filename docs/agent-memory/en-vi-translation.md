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

- `texts.parquet` (`text`, the unique transcripts in translation order) and `rows.parquet`
  (`split`, `id`, `text`) hold the current selection, rebuilt by `texts`. It takes every
  validation row whose clip lasts `--min-seconds` to
  `--max-seconds` (at most the 280-frame cap, 22.4 s), plus train rows up to `--hours`. Transcripts
  without letters (`.`) are skipped, and identical ones are translated once. Train is drawn in a
  fixed random order of units: a Common Voice clip, or a whole Libriheavy recording, so a batch
  carries the neighbouring book text. With the same duration range, a larger `--hours` extends a
  smaller selection; a new range draws a different order.
- The current selection is the first tranche: 5-15 s clips, 1,000 train hours (Libriheavy
  627 h, Common Voice 373 h) and 13,664 validation rows (28.9 h), so 410,618 texts (46.5M
  characters) in 435,019 rows. With no limits it would be 2,812,262 texts in 3,387,073 rows.
- `translations.jsonl` (`en`, `vi`, `profile`, `usage` per accepted batch) and `rejects.jsonl`
  (`en`, raw output) are append-only, and `translate` resumes from them by English text, so
  rebuilding `texts` keeps every finished translation.
- `translate.lock` stops a second run on the same directory; `export` takes it too.
- `{split}.parquet` (`id`, `text_vi`) is written by `export` once every row of the split is
  translated, and removed while it is not. `export` reports the plan's two gates (identical to the
  English, over twice its length) but drops nothing.
- All 20,882 validation texts of ≤ 22.4 s are translated, and the tranche's validation rows are
  exported. The first train tranche is running on `claude-p`, then `claude-l`, then `default`.

## Token budget

Measured with Sonnet 5.5 on the validation split:

- Output is about 0.62 tokens per English character, because Vietnamese costs 0.585 tokens per
  character. It is ~88% of the cost at API price weights.
- The full corpus projects to ~240M output and ~153M input tokens over ~48k calls. Since 1.1M output
  tokens used 23% of one Max plan's 5-hour window, that is ~50 such windows. The first tranche
  needs ~28M output tokens, about 6 windows.
- Measured levers, as changes to total cost:
  - one call per line instead of batches: +80%;
  - no prompt cache for the 687-token instruction prefix: +2.2%;
  - no line numbers in the output (2 tokens per line): −2.1%, but merged or split lines would
    then pass unnoticed;
  - 16k-character batches: −0.7%, since Claude Code adds ~316 uncached input tokens per call;
  - Telex or VIQR output: +26% or +61%, so ASCII encodings are not worth retrying.
- Thinking can't be disabled through the local API; at low effort it shows in ~25% of calls but
  adds under 1% of output.
- Scope is the big lever. With the English delayed by U[0, 0.5 × Vietnamese duration] and the
  Vietnamese as long as the English, ~22% of characters are in rows likely to overflow 280
  frames; clips over 15 s hold 44% of them, so the first tranche stops at 15 s. Its 5 s minimum
  drops short Common Voice clips (median 5.1 s), which carry little simultaneous lag.

## Model and prompt

- In a blind benchmark on 100 validation transcripts, judged by Opus and Sonnet, mean adequacy
  was 4.59/4.31 for Opus, 4.43/4.24 for Sonnet (tied on short Common Voice lines) and 3.82/3.74
  for Haiku, which made meaning errors and once answered as a chatbot. Sonnet at low effort was
  chosen.
- The prompt keeps dialogue quotes and the English order of clauses. An earlier wording that
  banned quotation marks made Sonnet move speech tags to the front of the line.
- A batch is accepted only with exactly one `n|` line per input, numbered 1..N. Rejections are
  rare: one of ~460 validation batches over two prompt versions, for a translation broken across
  two lines. Rejected lines are retried in quarter-size batches until they are down to ~500
  characters, then alone. A line that still fails alone ends the run with an error.
- The plan's two gates mostly flag good output. On the ≤ 22.4 s validation texts, the 22 lines
  identical to the English are names, and the 17 lines over twice the source length are correct
  translations of very short lines.

## Running

- Commands are in [training/prep/README.md](../../training/prep/README.md). The script runs on the
  Mac with the conda Python against the local Claude API (`~/.local/share/claude-local-api/`).
- 64 client threads fill the server's 48 slots and stay under macOS's 128-connection backlog.
- `translate` needs `--profile` for each subscription it may spend, in order of use. When every
  named profile is out of quota, it sleeps until the earliest reset. Don't name `default`, the
  interactive Claude Code login, unless the user asks. Every named profile must have extra usage
  turned off (`/v1/health` shows `overageStatus`), or the server keeps serving it and bills.
- Run long jobs as `nohup caffeinate -i python …`, logging to `translate.log` in the output
  directory; quota waits pause while the Mac sleeps.
- Every call turns off the server's response cache, so a retried batch is generated again.
- 5xx replies, dropped connections and a 429 without `retry-after` (a refusal of that request only)
  are retried with backoff (1, 2, 4, 8 and 16 minutes), then stop the run. Any other API error
  stops it at once.
- Ctrl-C, or `kill <pid>` for a background run, ends quota waits and lets in-flight batches finish
  and be logged. A run refuses to start when a hard kill has cut off the last line of
  `translations.jsonl`.

## How to work

- Before committing pipeline changes or starting a quota-heavy run, test a small sample and
  report measured alternatives, as the user asked.
- Translate in tranches, as the roadmap validates the pipeline on 1k hours before scaling, and
  check TTS on a tranche before spending more quota. The user chose the first tranche's 5-15 s
  window and its profiles, including `default` as the last one.
