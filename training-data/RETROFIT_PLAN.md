# EN timbre-match retrofit plan (pre-tranche-3 Hub rows)

**Status: not started. Decision gate: only execute if training evals show the model fails to
carry speaker identity across languages (voice-consistency weakness).**

## Problem

`anquachdev/PhoMT-en-vi-speech` is split by EN voice scheme:

| Rows | Share | EN voice |
|---|---|---|
| ~357k (tranche 3, index ≥ 345600) | 51% | timbre-matched to the row's VI voice (`voice_bank/vi_to_en_voices.json`) |
| ~339k (pre-campaign + tranches 1–2) | 49% | old scheme: EN voice picked independently of VI voice |

Sampled mismatch vs the current assignment: ~95% of pre-t3 rows. VI voices mostly agree;
it is the EN pick that changed.

## Constraints

- The uploader is append-only; retrofitting means **rewriting existing shards in place**
  (same `path_in_repo`, `CommitOperationAdd` overwrites).
- Hub rows do not store voice names, and manifests only survive for tranches 1–2 (192k pairs).
  Tranche-1 VI wavs are deleted locally; older-than-t1 rows have neither manifests nor wavs.
  → Source the VI audio from the Hub shards themselves and pick the EN voice by embedding
  the actual VI audio (VieNeu 192-d speaker encoder, as in `match_voices.py`) → nearest
  same-gender EN voice on the 34-candidate grid. This is more robust than replaying
  `pick_row_voice` against historical pool states.
- Hub rate limit: 128 repo commits/hour → batch rewritten shards 5 per commit (upload.py pattern).

## Phases

1. **Inventory** — enumerate affected shards: all 78 pre-campaign parquets plus every
   `upload-state.json` shard entry with `source_index_max < 345600`. Record in a
   `retrofit-state.json` (shard path → pending/done) committed alongside each rewrite
   for resumability.
2. **Per-shard rewrite loop** (streaming, ~500 MB scratch per shard; pipeline N+1 build
   during N upload as in upload.py):
   a. Download shard, decode `audio_vi` per row.
   b. Embed VI audio → nearest same-gender EN voice (cache embedding→voice per shard).
   c. Synthesize EN (Kokoro, matched voice/speed), silence-gate each clip
      (finite + rms ≥ 1e-4, CPU rescue), ratio-check EN/VI 0.4–1.8 — regenerate once,
      keep the old EN clip if still out of band (do not drop rows: row count must not change).
   d. Replace `audio_en`, `duration_en_s`, `duration_ratio_en_vi`; write parquet;
      upload 5 shards/commit with `retrofit-state.json`.
3. **Card refresh** — `num_examples` unchanged; recompute `download_size`/`dataset_size`
   deltas once at the end.
4. **Validation** — footer-scan all rewritten shards (row counts unchanged), byte
   spot-check EN audio vs freshly synthesized copies, silence scan already inline in 2c.

## Estimates

- EN synthesis: ~339k clips ≈ ~600 audio-h at ~40× RT concurrent ≈ **15 h generation**.
- Transfer: ~170 GB down + ~170 GB up (~25 MB/s Xet up) ≈ **4–5 h**.
- Wall clock ≈ 1 day with synthesis and transfer overlapped; disk needs only scratch (~5 GB).
