# Noisy Vietnamese source

Read when touching `training/prep/augment.py`, `training/prep/noisy_cache.py`, the
`en-vi-val-noisy` evaluation set or a `*-noisy` cache.
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Conditions](#conditions) ·
[Noisy cache](#noisy-cache) · [Checks](#checks) · [Evaluation](#evaluation)

## Why

Every Vietnamese source clip in the training caches is VieNeu TTS: full 24 kHz band, no room, no
noise. The phone hears real rooms, and FLEURS dev (real 16 kHz recordings) scores far below the
held-out TTS rows: step 63k of the first run had chrF 23.5 on FLEURS against 39.4 on Common
Voice and 42.2 on Libriheavy `en-vi-val`. Stage 1.2 of the
[plan](../vi_en_parity_plan.md#1-data) re-encodes the source under simulated recording
conditions; `en-vi-val-noisy` measures what the channel alone costs.

## Conditions

`augment.Bank.draw(key)` draws a clip's conditions from a `random.Random(key)`, so a row gets the
same ones on every machine; `Bank.apply` applies them on any torch device without changing the
length:

```text
clip ─► reverb p 0.5 ─► + noise p 0.8, SNR U[5, 30] dB ─► band ─► gain U[-15, 0] dB, clip to ±1
        2,325 RIRs        background 0.7 (129 files), looped    full 0.40 · 16 kHz rate 0.35
                          foreground 0.3 (806 files), once      8 kHz rate 0.15 · phone 0.10
```

- The bank is OpenSLR 28 (`RIRS_NOISES`, Apache 2.0; its point-source noises are MUSAN's
  Freesound part, public domain), 1.3 GB zipped, 3.6 GB unpacked. It is pinned by SHA-256, and
  `augment.fetch_bank` unpacks it aside and renames it into `--bank-dir` (default
  `data/rirs_noises`; the Mac's copy is `/Volumes/data/datasets/rirs_noises`), so a cut-short
  unpack never passes for the bank.
- RIRs: the 325 real ones plus 1,000 seeded simulated ones each from the small and medium rooms.
  Each is cut from 2.5 ms before its peak to at most 1 s at unit energy, and the reverberant clip
  is scaled back to the dry clip's RMS.
- Noises (6.7 h):
  - background (diffuse room noise, the point-source files marked background) loops from a
    drawn offset;
  - a foreground event (a door, a cough; 26% are under 5 s) plays once at a drawn place, its SNR
    measured over the event itself.

  Noise is not reverberated.
- Everything in the bank is 16 kHz, read with the standard library's `wave` (sphn refuses the
  multichannel files' WAVE_FORMAT_EXTENSIBLE headers; channel 0 is used) and resampled to 24 kHz
  with soxr.
- The band filter is zero-phase on a zero-padded FFT with raised-cosine edges (10% of the
  cutoff). The gain only lowers: the TTS peaks at a median 0.73, and +5 dB clipped 16% of the
  `en-vi-val-noisy` clips (now 2 of 360).
- Left out:
  - speed changes: the cached delays and text timing assume the clip's own timing, and a slowed
    source would move Vietnamese words past the English that translates them;
  - low-bitrate codecs.
- Known gap: stream gaps hold the encoded digital-silence track, so a noisy row sits between
  silent gaps where a real recording stays noisy.

## Noisy cache

`training.prep.noisy_cache` writes, for each `shard_<name>.pt` of a clean cache split, a shard
with the same rows in the same order, in two steps that share `--out-dir`:

```text
fetch  (network, CPU; runs beside training)  HF parquet ─► .sources/<name>.parquet
encode (CUDA, after training)                .sources + clean shard ─► shard_<name>.pt
```

- `fetch` downloads each parquet from `anquachdev/common-vi-en` at `eval_sets.EN_VI_REVISION`,
  16 at a time, each tried 4 times. It keeps the id, the duration and the Opus bytes of the selected rows
  (`random.Random(f"{seed}:select:{id}").random() < --fraction`, 0.6), null for the others, in
  `.sources/`, about a fifth of the parquet's bytes, and deletes the download. It pauses while the disk has
  less than `--min-free-gb` (50) free: a trainer on the same disk needs a checkpoint pair
  (37.5 GB) beyond its steady state while the sync still uploads the pair it pruned. Its first run
  writes `manifest.json` (seed, fraction, dataset revision, bank SHA-256); a later fetch with
  other values stops.
- `encode` re-encodes the rows whose source holds audio:
  - the clip is decoded as `eval_sets.decode_opus` decodes it and resampled with soxr, as the
    builder did. It matches PyAV, the builder's decoder, to the sample on the Mac-voiced shards;
    on the 264 shards voiced on anquachdev's machine (hybrid SILK+CELT Opus) the decode sits 1–2
    samples apart with amplitude differences up to 0.3, though every clip keeps its frame count
    (118,986 rows of 60 shards from both machines checked);
  - the clip goes under `Bank.draw(f"{seed}:{id}")`, stored as `source_conditions` (None for a
    clean row);
  - Mimi (moshi's, float32, TF32 off) encodes the clips shortest first, in right-padded batches
    of up to `--batch-seconds` (640) of padded audio;
  - the codes replace rows 17–32 up to `vi_frames`.

  A clip whose frame count differs from the cache's, or a row without its codec EOS, stops the
  run. Text, English codes, frames, delays and both EOS positions are untouched.
- The shard gains `source_augment` (clean shard plus the manifest). The format string stays
  `hibiki_vn_grounded_cache_v2`, so the trainer reads the split as any cache.
- `encode` checks each source against its shard's `source` (file and byte size) and its rows' ids.
  It waits for a source a running fetch has not written, and skips a shard whose source is
  missing when no fetch runs, then exits non-zero naming them (run `fetch`, then `encode`
  again). It deletes each source once its shard is written and prints `every shard written` at
  the end: train only after that line, since the trainer takes whatever shards the folder holds.
- Both skip written shards; locks keep a second fetch or encode off the out-dir.
- Sizes for `en-vi-grounded-v2/train`: 1,832 parquets, 442.7 GB; 9,057 h of Vietnamese, ~5,400 h
  re-encoded. The pod downloads at ~39 MB/s (16 at a time; one alone ~10 MB/s; reading only the
  needed columns over HTTP was no faster), so `fetch` takes ~3.2 h and keeps ~80 GB of sources.
  On the Vast H100 pod's CPU, `encode` ran at 24x real time (cv27-00000: 2.3 h of audio in 347 s,
  8 threads); the GPU rate is still to be measured.

## Checks

- On the Mac CPU (2026-11-04), 48 rows of `shard_cv27-00000`: moshi's PyTorch Mimi on the
  soxr-resampled clip reproduces the cached source codes. Codebook 0 matches on 100% of frames and
  all 16 codebooks on 99.91% (2,215 frames), alone or in padded batches. With an FFT resampler
  instead of soxr, codebook 0 kept 98.8% but all 16 only 64%: the residual codebooks follow tiny
  spectral differences, so the pod resamples with soxr too.
- Right padding leaves earlier frames alone: every Mimi convolution pads on the left, and its
  transformer has a causal mask. A clip batched beside a longer, noisy one encodes to the same
  16 codebooks as alone (CPU, PR #52's review).
- On the pod (CPU, 2026-11-04), the full `shard_cv27-00000`:
  - 1,750 of 2,961 rows were re-encoded; the clean rows came out byte-identical;
  - in every row, text, English codes, frames and both EOS positions are unchanged;
  - codebook 0 of the re-encoded rows matches the clean codes on 40% of frames.

  Fetch then encode gave the same shard as a single pass.

## Evaluation

`training.prep.eval_sets --sets en-vi-val-noisy` writes `en-vi-val-noisy/{cv27,libriheavy}.jsonl`
from the built `en-vi-val` rows: every clip under `Bank.draw(f"0:{id}")` (always applied, unlike
training's 60%), as 24 kHz WAV, with the drawn `conditions` in the row. It resamples the clean
48 kHz clip with `sphn.resample`, as `eval.py` resamples the clean set, so the two sets differ by
the conditions alone. (Every evaluation set reaches Mimi through `sphn.resample`, training through
soxr: a residual-codebook difference that predates this set.) Scored against the clean set on the
same checkpoint it shows what the channel costs. It is not in the trainer's default
`--eval-pairs`; the second epoch passes it.

Measured on the Mac (`eval.py`, MPS, batch 16), step 63,000 of the first run, 2026-11-04:

| Set | Common Voice | Libriheavy | Both (360) |
|---|---:|---:|---:|
| `en-vi-val` (clean TTS) | 39.4 | 42.2 | 41.5 |
| `en-vi-val-noisy` | 34.4 | 37.5 | 36.7 |
| FLEURS dev (real speech) | | | 23.5 |

- By condition, against the same rows clean: reverb −6.1 chrF; SNR under 15 dB −8.8 (15 dB
  and up −2.9); phone band −8.9, 8 kHz −7.8, 16 kHz −4.7.
- The channel so costs ~5 of the ~18 points between clean TTS and FLEURS. The rest is real
  speech and FLEURS's content (encyclopaedic sentences, names, numbers), which noise cannot
  teach: the real-Vietnamese tranche (plan stage 1.3) is the larger lever.
