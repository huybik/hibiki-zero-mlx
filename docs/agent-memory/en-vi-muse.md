# EN→VI translation on Muse

Read when touching the `muse` step of `training/prep/translate_en_speech.py` or
`/Volumes/data/datasets/en_speech_vi_text/muse`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [The step](#the-step) ·
[Measurements](#measurements) · [Running](#running)

## Purpose

After the corpus's bandwidth prune, the user chose to let Sonnet finish the Common Voice text and
to translate all remaining Libriheavy text with Muse Spark 1.3 Contributor through the paid
OpenRouter API. That takes hours instead of weeks of Claude quota, for about an eighth of Gemini's
price, with slightly stiffer Vietnamese.

Contributor mode's prices ($0.10 input, $0.20 output per million tokens) buy Meta the right to train
on prompts and replies. So Muse gets only Libriheavy text: `claim` never claims a text with a Common
Voice row.

## The step

- `claim --key OPEN_ROUTER` without `--hours` claims every untranslated Libriheavy text. Run it
  again after `texts` picks up new shards; it adds the new texts, but not lines a paid engine
  released, and it releases claimed texts that have gained a Common Voice row. `muse` and
  `gemini` refuse to start while any claimed text has one.
- `muse --key OPEN_ROUTER` sends the claimed lines to `meta/muse-spark-1.3-contributor` with
  `translate`'s prompt, `n|` format, batches and retry rounds, at `--concurrency` calls.
  Reasoning cannot be turned off, so `reasoning.effort` is `minimal`.
- Accepted batches go to `muse/translations.jsonl` (`en`, `vi`, `key`, `usage` with input, output
  and reasoning tokens) and rejected output to `muse/rejects.jsonl`. Lines that fail alone are
  released from the claims, so `translate` takes them when started again; if they are over 10%
  of the key's lines, the run stops instead, since that is a systematic failure.
- A moderation 403 (its error carries `metadata.reasons`) counts as a rejected batch. Any other
  403 is a spent key or budget limit, and it stops the run at once with OpenRouter's message, as
  every other client error does (402 is out of account credit). A 408, 429, 5xx, a 200 holding
  only an error, or a dropped connection is retried after 1, 2, 4, 8 and 16 minutes, then stops
  the run.
- It holds `claims.lock`, as `claim` and `gemini` do, so only one of them runs at a time.

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
- In production at 48 calls it runs ~110 batches a minute for ~$0.69 per 1,000 batches. On the
  final corpus, Muse had 1.66M lines (34,276 batches) left: ~5 h and ~$24, with $10.55 of
  the key's $50 already spent.
- OpenRouter's Batch API has no `:batch` variant of any Muse model (checked 2026-10-23), and the
  Contributor price is below most batch prices anyway, so Muse uses the synchronous API.

## Running

- Load `.env` first (`set -a && . ./.env && set +a`). The `OPEN_ROUTER` key has a $50 credit
  limit, set on OpenRouter.
- Run as `nohup caffeinate -i python -m training.prep.translate_en_speech muse --key OPEN_ROUTER
  --concurrency 48 >> /Volumes/data/datasets/en_speech_vi_text/muse.log 2>&1 &`. `kill` lets
  in-flight batches finish; rerun to resume.
- From an agent's shell, start it in a new session instead (Python `subprocess.Popen(...,
  start_new_session=True)`). A `nohup … &` start from a Claude Code Bash call was hard-killed
  12 minutes in (2026-10-23): no traceback, `caffeinate` gone too, most likely with the call's
  process group. A run that stops with no last log line needs checking with `ps`.
- When the corpus changes: stop `translate` and `muse`, run `texts`, then
  `claim --key OPEN_ROUTER`, then start both again.
