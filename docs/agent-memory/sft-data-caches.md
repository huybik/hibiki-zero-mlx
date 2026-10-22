# SFT pair rows and grounded caches

Read when touching `training/prep/cache_codes.py`, `training/prep/text_timing.py` or
`training/paths.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [Row layout](#row-layout)

## Flow

```text
pair file (.jsonl/.csv, local audio) ─► cache_codes ─► data/cache/<name>/shard_*.pt + index.csv
  └─ text_timing: wav2vec2 CTC word timing of the English text
data/cache/<name> ─► training.finetune.train --cache-dir / --val-cache-dir
```

- No pipeline produces pair rows yet. PhoMT and FLEURS, the earlier sources, are dropped. The
  next source is the English `en_speech` corpus ([en-speech-corpus](en-speech-corpus.md)) once
  roadmap phase 1 turns it into Vietnamese-source rows.
- Pair rows need `id`, `split`, `vi_audio`, `en_audio`, `vi_duration_s`, `en_duration_s`,
  `text_vi` and `text_en`; audio paths are absolute or repo-relative.
- `training/paths.py` holds the repo defaults (`weights/` artifacts) and file guards.
- Grounded-v2 (`hibiki_vn_grounded_cache_v2`) is the only cache format; `cache_codes` writes
  it and `CachedCodeDataset` rejects anything else:
  - Mimi at 24 kHz / 12.5 Hz, 32 rows per shard by default;
  - English audio delayed by a draw from U[0, 0.5 × Vietnamese duration], seeded by
    `1234:{split}:{id}` and inserted as left silence;
  - English text word-timed by CTC alignment with `facebook/wav2vec2-base-960h`; rows that
    fail alignment or score below 0.5 are printed as rejected and skipped.
- Each shard records each row's `vi_audio`/`en_audio` strings, so the same audio cached from
  different paths gives different shard bytes.

## Row layout

- Codes are int32 `[33, T]`:
  - row 0 is text;
  - rows 1–16 are English target Mimi;
  - rows 17–32 are Vietnamese source Mimi.
- Model config: `n_q` 32, `dep_q` 16, `card` 2048, `text_card` 48000, text pad id 3.
- Text stream: PAD fill, tokens at their CTC frames, then tokenizer EOS at the frame equal to
  the English code length, then -1.
- Target audio is -1 after its length, with no explicit EOS.
- Source audio has codec EOS 2048 at `source_len`, then -1.
- T = max(text end, target length, source length + 1).
- Batches are right-padded with -1 to a multiple of the frame bucket. There is no dataset
  transform at training time.
