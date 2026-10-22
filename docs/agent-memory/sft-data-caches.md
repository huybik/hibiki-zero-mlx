# SFT data and caches

Read when touching `finetune/cache_codes.py`, `finetune/cache_phomt_stream.py`,
`finetune/text_timing.py`, `finetune/build_pairs.py`, `finetune/publish_grounded_cache.py`,
`finetune/freeze_full_data_receipt.py` or `remote_dataset/download_fleurs_vi_en.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [Row layout](#row-layout) ·
[Receipt](#receipt)

## Flow

```text
google/fleurs vi_vn+en_us ─► download_fleurs_vi_en.py ─► remote_dataset/fleurs_vi_en/<split>/
  ─► build_pairs.py ─► finetune/pairs/{train,validation,test,val16,val128}.jsonl
  ─► cache_codes.py ─► finetune/cache/{train,validation}_grounded_v2/
anquachdev/PhoMT-en-vi-speech ─► cache_phomt_stream.py ─► finetune/cache/phomt_grounded_v2/
caches ─► publish_grounded_cache.py (manual) ─► HF dataset huybik/hibiki-zero-vi-full-sft/grounded-v2/
caches ─► freeze_full_data_receipt.py (in h100.sh preflight) ─► frozen receipt ─► train.py
```

- `h100.sh cache-grounded` needs `pairs/train.jsonl` and `pairs/validation.jsonl`.
  - It runs `HIBIKI_CACHE_WORKERS` (default 4) workers of
    `cache_phomt_stream.py --profile h100`. These stream one PhoMT parquet shard at a time, drop
    Vietnamese audio over 25 s and skip rows whose texts and durations repeat within that
    worker's current run.
  - It then runs `cache_codes.py` for FLEURS, at 32 rows per shard.
- Grounded-v2 (`hibiki_vn_grounded_cache_v2`) is the only cache format; the builders write it
  and `CachedCodeDataset` rejects anything else:
  - Mimi at 24 kHz / 12.5 Hz;
  - English audio delayed by a draw from U[0, 0.5 × Vietnamese duration], seeded by
    `1234:{split}:{id}` and inserted as left silence;
  - English text word-timed by CTC alignment with `facebook/wav2vec2-base-960h`; rows scoring
    below 0.5 are rejected into `alignment_rejects_w*.jsonl`.
- Publishing uploads `cache_chunk_{0..7}.tar.zst`, `fleurs_cache.tar.zst`, pairs and reject
  files, `manifest.json` and `README.md`. It refuses a non-empty prefix and verifies remote
  SHA-256s and the exact file set. Restoring is in [docs/finetune.md](../finetune.md).

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

## Receipt

- `freeze_full_data_receipt.py` writes
  `finetune/runs/grounded_v2_full_direct_5epoch_receipt/{sample_manifest.jsonl,full_data_receipt.json}`.
  Existing files are never overwritten.
- It checks four artifact hashes and the cache receipts: 1377 PhoMT, 46 FLEURS-train and
  5 FLEURS-validation shards.
- Rows over 280 frames are dropped, not truncated. It then draws
  `select_weighted([0.95, 0.05], 719,120, seed 42)`, giving 683,164 PhoMT and 35,956 FLEURS
  rows, sorts by length and shuffles batch order with batch 16 and seed 42.
- Only 1,392 FLEURS rows are eligible, so they are drawn with replacement, about 26× each.
