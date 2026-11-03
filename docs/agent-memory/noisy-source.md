# Noisy Vietnamese source

Read when touching `training/prep/augment.py`, `training/prep/noisy_cache.py`, the
`en-vi-val-noisy` evaluation set or a `*-noisy` cache.
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Conditions](#conditions) ·
[Noisy cache](#noisy-cache) · [Checks](#checks) · [Evaluation](#evaluation)

## Why

Every Vietnamese source clip in the training caches is VieNeu TTS: full 24 kHz band, no room, no
noise. The phone hears real rooms, and FLEURS dev (real 16 kHz recordings) scores far below the
held-out TTS rows (step 18k: chrF 20.7 against 31.4 on Common Voice and 36.3 on Libriheavy).
Stage 1.2 of the [plan](../vi_en_parity_plan.md#1-data) re-encodes the source under simulated
recording conditions; `en-vi-val-noisy` measures what the channel alone costs.

## Conditions

`augment.Bank.draw(key)` draws a clip's conditions from a `random.Random(key)`, so a row gets the
same ones on every machine; `Bank.apply` applies them on any torch device without changing the
length:

```text
clip ─► reverb p 0.5 ─► + noise p 0.8, SNR U[5, 30] dB ─► band ─► gain U[-15, +5] dB, clip to ±1
        2,325 RIRs        background 0.7 (129 files)       full 0.40 · 16 kHz rate (8 kHz low-pass) 0.35
                          foreground 0.3 (806 files)       8 kHz rate (4 kHz) 0.15 · phone 300–3,400 Hz 0.10
```

- The bank is OpenSLR 28 (`RIRS_NOISES`, Apache 2.0; its point-source noises are MUSAN's
  Freesound part, public domain), 1.3 GB, pinned by SHA-256 and unpacked by `augment.fetch_bank`
  into `--bank-dir` (default `data/rirs_noises`; the Mac's copy is
  `/Volumes/data/datasets/rirs_noises`). RIRs: the 325 real ones plus 1,000 seeded simulated
  ones each from the small and medium rooms; each cut from 2.5 ms before its peak to at most
  1 s, unit energy, and the reverberant clip scaled back to the dry clip's RMS. Noises: 6.7 h,
  looped from a drawn offset. Everything is 16 kHz, read with the standard library's `wave`
  (sphn refuses the multichannel files' WAVE_FORMAT_EXTENSIBLE headers; channel 0 is used) and
  resampled to 24 kHz with soxr.
- The band filter is zero-phase on a zero-padded FFT with raised-cosine edges (10% of the
  cutoff).
- Speed changes are left out: the cached delays and text timing assume the clip's own timing,
  and a slowed source would move Vietnamese words past the English that translates them.
- Known gap: stream gaps hold the encoded digital-silence track, so a noisy row sits between
  silent gaps where a real recording stays noisy.

## Noisy cache

`training.prep.noisy_cache` (CUDA, the pod) writes, for each `shard_<name>.pt` of a clean cache
split, a shard with the same rows in the same order:

- a row is re-encoded when `random.Random(f"{seed}:select:{id}") < --fraction` (0.6); its
  conditions are `Bank.draw(f"{seed}:{id}")` and are stored as `source_conditions` (None for a
  clean row); the shard gains `source_augment` (clean shard, fraction, seed, bank SHA-256,
  dataset revision); the format string stays `hibiki_vn_grounded_cache_v2`, so the trainer
  reads it as any cache;
- the clip comes from the parquet shard the cache shard names (`source`, checked by file and
  byte size), downloaded from `anquachdev/common-vi-en` at `eval_sets.EN_VI_REVISION` two shards
  ahead and deleted once used; workers decode it as `eval_sets.decode_opus` does (matching
  PyAV, the builder's decoder, to the sample) and resample it with soxr, as the builder did;
- Mimi (moshi's, float32, TF32 off) encodes the clips shortest first in right-padded batches of
  up to `--batch-seconds` (640) of padded audio; the codes replace rows 17–32 up to `vi_frames`.
  A clip whose frame count differs from the cache's, or a row without its codec EOS, stops the
  run. Text, English codes, frames, delays and both EOS positions are untouched;
- a rerun skips written shards; a lock keeps a second run off the same out-dir.

## Checks

On the Mac CPU (2026-11-04), 48 rows of `shard_cv27-00000`:

- moshi's PyTorch Mimi on the soxr-resampled clip reproduces the cached source codes: codebook 0
  on 100% of frames, all 16 codebooks on 99.91% (2,215 frames), alone or in padded batches. An
  FFT resampler instead of soxr kept codebook 0 on 98.8% but all 16 on only 64%: the residual
  codebooks follow tiny spectral differences, so the pod resamples with soxr too.
- With drawn conditions only the source codes before `vi_frames` change; codebook 0 matches the
  clean codes on 37% of frames.

## Evaluation

`training.prep.eval_sets --sets en-vi-val-noisy` writes `en-vi-val-noisy/{cv27,libriheavy}.jsonl`
from the built `en-vi-val` rows: every clip under `Bank.draw(f"0:{id}")` (always applied, unlike
training's 60%), as 24 kHz WAV, with the drawn `conditions` in the row. Scored against the clean
set on the same checkpoint it shows what the channel costs; it is not in the trainer's default
`--eval-pairs`.
