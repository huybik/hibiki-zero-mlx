# SFT pair rows and grounded caches

Read when touching `training/prep/cache_codes.py`, `training/prep/text_timing.py` or
`training/paths.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [PhoMT cache](#phomt-cache) ·
[Row layout](#row-layout)

## Flow

```text
pair file (.jsonl/.csv, local audio) ─► cache_codes ─► data/cache/<name>/shard_*.pt + index.csv
  └─ text_timing: wav2vec2 CTC word timing of the English text
HF phomt-grounded-v2-5s/ ─► data/cache/phomt-grounded-v2-5s (restored, see PhoMT cache)
data/cache/<name> ... ─► training.finetune.train --cache-dir (several), --val-cache-dir
```

- No pipeline produces pair rows yet. The next source is the English `en_speech` corpus
  ([en-speech-corpus](en-speech-corpus.md)) once roadmap phase 1 turns it into
  Vietnamese-source rows; its caches train together with the [PhoMT cache](#phomt-cache).
  FLEURS is dropped.
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

## PhoMT cache

- Pooled with the `en_speech` caches by passing it as one more `--cache-dir`. Its 804 hours of
  English speech (~770 h of them 5–15 s clips) count toward the ~10k training hours of 5–15 s
  clips, which is why `en_speech` stops at 9,200 hours of them (user's decision, 2026-10-22).
- It is the HF dataset `huybik/hibiki-zero-vi-full-sft` prefix `phomt-grounded-v2-5s/`: 1,377
  `shard_*.pt` (6.5 GB) holding 361,968 `train` rows, restored as in
  [docs/finetune.md](../finetune.md). The Mac copy is
  `/Volumes/data/datasets/phomt-grounded-v2-5s/`.
- It follows `en_speech`'s clip rule (user's choice, 2026-10-23): the rows of the full
  `grounded-v2/` cache in the same repo (690,067 rows, `cache_chunk_{0..7}.tar.zst` with
  `pairs_w*.jsonl` and `manifest.json`) whose English clip lasts at least 5 s
  (`en_duration_s`) and whose codes fit 280 frames. Kept rows, shard names and metadata are
  unchanged.
- `grounded-v2/` was built from `anquachdev/PhoMT-en-vi-speech` at revision `33400f7` (565 GB of
  TTS audio on both sides) by the since-removed `cache_phomt_stream.py`, so it is never
  rebuilt. Filter from it if the rule changes. A row's cached English length counts the
  inserted delay; `en_frames - target_delay_frames` is within a frame of the clip.
- 190,911 of the kept rows carry `cross_lingual_timbre_matched`.

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
