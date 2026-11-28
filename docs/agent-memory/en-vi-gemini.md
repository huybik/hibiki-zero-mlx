# EN→VI translation on Gemini

Read when touching `/Volumes/data/datasets/en_speech_vi_text/gemini` or weighing Gemini's Batch
API for a translation.
Back to [CONTEXT.md](../../CONTEXT.md) · [What it did](#what-it-did) ·
[Measurements](#measurements)

## What it did

The Claude subscriptions' weekly caps would have stretched the whole corpus over weeks
([en-vi-translation](en-vi-translation.md)), so the user chose to translate 1,000 hours of
Libriheavy on each of two paid Gemini API keys with Gemini 3.8 Flash through the Batch API, at
half the interactive price, with the Claude path's prompt, `n|` line format and acceptance check.
That share, 513,727 texts (105M characters), finished in 30 minutes for ~$75. The user then chose
not to pay ~$300 more for the rest; the remaining Libriheavy went to Muse
([en-vi-muse](en-vi-muse.md)). The code (`translate_en_speech`'s `claim` and `gemini` steps) is in
git at commit 364091a; `gemini/` holds the accepted batches (`translations.jsonl`: `en`, `vi`,
`key`, `usage`), the rejected output and each key's last job record (`<KEY>.job.json`).

## Measurements

On the 71 production batches that hold 100 random Libriheavy validation lines (2,843 lines,
560k characters, 10.6 h), one job per key:

- Each job finished in 3-4 minutes, and all 71 batches passed the format check.
- `thinking_level` LOW (3.8 Flash has no minimal) used no thinking tokens.
- Gemini counts 0.29 input and 0.33 output tokens per English character; the instructions are
  311 tokens. Its Vietnamese is 1.10× the English characters, Sonnet's 1.00×, which matters for
  the 280-frame cap once voiced.
- Blind judges scored adequacy/fluency (1-5) on the 100 lines: Opus gave Sonnet 4.31/4.12 and
  Gemini 4.18/4.07; Sonnet gave 3.95/3.81 and 4.00/3.77. Gemini's errors are nuance: word
  choice, a plural "you", a stronger or weaker shade than the English.
- A trimmed 120-token prompt saved ~1% of cost but lost 0.06-0.14 adequacy and ~0.16 fluency on
  both judges, so the full prompt stayed.
- Batch jobs are limited per project in enqueued tokens: 3M on Tier 1, 400M on Tier 2. A
  1,000-hour job (~6,650 batches, ~52M characters, ~15M input tokens) was accepted on both keys,
  so they are above Tier 1.

The full 2 × 1,000-hour run:

- Both first-round jobs (~6,650 batches each) finished within 6 and 20 minutes; with three retry
  rounds per key the whole run took 30 minutes. First rounds accepted 99.9% of lines.
- Tokens: 30.1M input and 34.0M output, ~$0.72 per million English characters at the batch prices
  of $0.375 input and $1.875 output per million tokens (both double on 2027-01-01).
- Rejections: 10 empty replies (`finishReason` absent, blocked), 1 `PROHIBITED_CONTENT` and 1
  `RECITATION`, on Freud-style passages and verse; one such line failed alone and went to Claude.
- Gates: 30 lines identical to the English and 8 over twice its length; the 35 long lines
  without Vietnamese diacritics are Latin, name lists and addresses kept as written.
