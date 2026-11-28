# EN→VI transcript translation

Read when touching the EN-VI dataset's `text_vi`, `/Volumes/data/datasets/en_speech_vi_text`, or
translating English transcripts to Vietnamese again.
Back to [CONTEXT.md](../../CONTEXT.md) · [What exists](#what-exists) ·
[Model and prompt](#model-and-prompt) · [Retranslation](#retranslation)

## What exists

Every `en_speech` clip has a Vietnamese line for voice-transfer TTS to read ([vi-tts](vi-tts.md));
Vietnamese is the model's source stream, so each line stays aligned with its English audio. The
translation is finished. The code that made it, `training/prep/translate_en_speech.py`, is in git
at commit 364091a.

- 3,388,263 unique texts (526M characters) in 3,490,088 rows: Claude Sonnet 5.5 translated 616,834
  (all Common Voice, through the local Claude API on subscription quota), Gemini 3.8 Flash 375,868
  ([en-vi-gemini](en-vi-gemini.md)) and Muse Spark 1.3 Contributor 2,395,561
  ([en-vi-muse](en-vi-muse.md)), those two only Libriheavy: no Common Voice text went to Google or
  Meta.
- The lines are the EN-VI dataset's `text_vi` ([en-vi-speech-dataset](en-vi-speech-dataset.md)).
  `/Volumes/data/datasets/en_speech_vi_text/` keeps the working files: the selection
  (`texts.parquet`, `rows.parquet`), each engine's accepted batches (`translations.jsonl`,
  `gemini/`, `muse/`, `retranslations.jsonl`: `en`, `vi`, a later file's translation of a text
  winning) and rejected output, `claims.parquet`, and the export, `train.parquet` (3,480,566 rows)
  and `validation.parquet` (9,522) of `id`, `text_vi` (2026-10-23). The export's gates flagged
  267 lines identical to the English and 167 over twice its length, and dropped none.

## Model and prompt

- In a blind benchmark on 100 validation transcripts, mean adequacy judged by Opus/Sonnet was
  4.59/4.31 for Opus, 4.43/4.24 for Sonnet (tied on short Common Voice lines) and 3.82/3.74 for
  Haiku, which made meaning errors. Sonnet at low effort was chosen; Opus 5.5 costs twice
  Sonnet 5.5 per token.
- Lines went in and out as plain numbered `n|text` in batches of up to 200, the instructions in a
  cached prefix; a batch was accepted only with exactly one `n|` line per input (about 1 batch in
  350 failed, mostly a translation broken across lines). Measured against that: one call per line
  +80% cost, no prompt cache +2.2%, no output numbers −2.1% (but merged or split lines would pass
  unnoticed), Telex or VIQR output +26% or +61%. Output was ~88% of the cost, 0.60–0.62 tokens per
  English character.
- The prompt keeps dialogue quotes and the English order of clauses. A wording that banned
  quotation marks made Sonnet move speech tags to the front of the line.
- Sonnet refuses one Common Voice sentence ("Aqueous cyclodextrin solutions can generate aerosols
  in particle size ranges suitable for pulmonary deposition."); its clip was deleted from the
  corpus ([en-speech-corpus](en-speech-corpus.md)).
- The gates mostly flag good output: on the validation texts, the 22 lines identical to the
  English are names, and the 17 over twice the source length translate very short lines.

## Retranslation

- Muse and Gemini wrote letters of other scripts into some lines, mostly mid-word: 7,095 texts
  whose English has none (checked 2026-10-24), 3,860 with Chinese, Japanese or Korean (`kính畏`)
  and the rest Cyrillic (`thế là спас`, ~2,700 rows), Arabic, Thai, Devanagari, Greek and a few
  others. Where the English has them too (`京都`), they are right and stay. Sonnet does it as
  well, on some Sino-Vietnamese words (`征 dụng`, `ngoại延`), at any effort; instructions asking
  for Latin letters only fixed some.
- Those texts were translated again, rejecting any batch that still held such letters: on Sonnet
  at low effort, then the lines failing alone on Opus at high effort (7, all fixed), into
  `retranslations.jsonl`. `retexts.parquet` (`id`, `text_vi`) lists the rows of every
  retranslated text: 7,102 texts, 7,152 rows in 1,460 shards, voiced again with
  `tts_en_speech --texts` ([vi-tts](vi-tts.md#running)).
