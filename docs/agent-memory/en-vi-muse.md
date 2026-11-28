# EN→VI translation on Muse

Read when touching `/Volumes/data/datasets/en_speech_vi_text/muse` or weighing Muse through
OpenRouter for a translation.
Back to [CONTEXT.md](../../CONTEXT.md) · [What it did](#what-it-did) ·
[Measurements](#measurements)

## What it did

After the corpus's bandwidth prune, the user chose to let Sonnet finish the Common Voice text and
to translate all remaining Libriheavy text with Muse Spark 1.3 Contributor
(`meta/muse-spark-1.3-contributor`) through the paid OpenRouter API: hours instead of weeks of
Claude quota, for about an eighth of Gemini's price, with slightly stiffer Vietnamese. It
translated all 2,395,561 claimed Libriheavy texts for $33.93, with the Claude path's prompt, `n|`
format, batches and retry rounds, at `reasoning.effort` `minimal` (reasoning cannot be turned
off). The code (`translate_en_speech`'s `muse` step) is in git at commit 364091a; `muse/` holds
the accepted batches (`translations.jsonl`: `en`, `vi`, `key`, `usage` with input, output and
reasoning tokens) and the rejected output.

Contributor mode's prices ($0.10 input, $0.20 output per million tokens) buy Meta the right to
train on prompts and replies, so Muse got only Libriheavy text, never a text with a Common Voice
row.

## Measurements

On the Gemini test's 71 batches (2,843 Libriheavy lines, 560k characters):

- `minimal` accepted every line first time in 134 s at 16 calls (median 26 s per batch) for $0.050:
  $0.09 per million English characters, against Gemini's $0.73. Reasoning was ~3% of output.
- `low` spent 33% of output on reasoning, cost $0.066, scored lower, and one batch wrote `39}`
  for `39|`.
- Its Vietnamese is 1.01× the English characters (Sonnet 1.00×, Gemini 1.10×).
- Blind judges scored all four engines together on the 100 sample lines (adequacy/fluency, 1-5).
  Opus gave Sonnet 4.26/4.15, Gemini 4.20/4.04, Muse `minimal` 4.09/3.78 and Muse `low`
  3.91/3.81. The Sonnet judge gave 3.97/3.84, 3.85/3.71, 3.87/3.62 and 3.70/3.54.
- Muse `minimal` trails Sonnet in fluency by 0.37 ± 0.10 (Opus) and 0.22 ± 0.10 (Sonnet judge), with
  literal calques and stiff word choice; in adequacy it is within ~0.1 of Gemini.
- Throughput is bound by latency (~25 s a call), not by rate limits: 48 calls run ~110 batches a
  minute and 96 calls ~210, with no 429s, at ~$0.69 per 1,000 batches either way.
- OpenRouter's Batch API has no `:batch` variant of any Muse model (checked 2026-10-23), and the
  Contributor price is below most batch prices anyway, so Muse ran on the synchronous API.
- A moderation refusal comes back as a 403 whose error carries `metadata.reasons`; any other 403
  is a spent key or budget limit, and a 402 is out of account credit.
