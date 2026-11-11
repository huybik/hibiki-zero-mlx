# Noisy Vietnamese source

Read when touching `training/prep/augment.py`, `training/prep/mimi_cache_cuda.py`, the
`en-vi-val-noisy` evaluation set or a `*-noisy` cache.
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Conditions](#conditions) ·
[Noisy cache](#noisy-cache) · [Speed](#speed) · [Checks](#checks) ·
[Evaluation](#evaluation)

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

`training.prep.mimi_cache_cuda` writes, for each `shard_<name>.pt` of a clean cache split, a shard
with the same rows in the same order, in two steps that share `--out-dir`:

```text
fetch  (network, CPU; runs beside training)  HF parquet ─► .sources/<name>.parquet
encode (CUDA, after training)                .sources + clean shard ─► shard_<name>.pt
```

- `fetch` downloads each parquet from `anquachdev/common-vi-en` at `eval_sets.EN_VI_REVISION`,
  16 at a time, each tried 4 times. It keeps the id and the Opus bytes of the selected rows
  (`random.Random(f"{seed}:select:{id}").random() < --fraction`, 0.6), null for the others, in
  `.sources/`, about a fifth of the parquet's bytes, and deletes the download. It pauses while the disk has
  less than `--min-free-gb` (50) free: a trainer on the same disk needs a checkpoint pair
  (37.5 GB) beyond its steady state while the sync still uploads the pair it pruned. Its first run
  writes `manifest.json` (seed, fraction, dataset revision, bank SHA-256); a later fetch with
  other values stops.
- `encode` re-encodes the rows whose source holds audio:
  - the clip is decoded by the builder's own code, `audio.decode` (PyAV), and resampled with
    soxr, so it has the samples and frame count the cache was built from. Any other decoder is
    wrong here ([Checks](#checks));
  - the clip goes under `Bank.draw(f"{seed}:{id}")`, stored as `source_conditions` (None for a
    clean row);
  - Mimi (moshi's, float32, TF32 off: the cache build's `cuda_mimi` and `mimi_batch`,
    [cache-build](cache-build.md#cuda)) encodes the clips shortest first, in right-padded batches
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
  `encode` takes ~2.3 h for the split on the pod ([Speed](#speed)).
- PyAV must be in the pod's environment (`h100.sh setup` pins it).

## Speed

Measured on the pod (H100, 2026-11-04) on six `libriheavy-medium` shards of 5–6 h re-encoded
each, every variant run from the same sources. At the default batch size every variant's shards
came out identical to the first encode's.

- The container may use 7.68 CPUs (`/sys/fs/cgroup/cpu.max`), though `nproc` shows 32. Decode
  workers past that budget exhaust the quota, and the throttle stalls the process that feeds the
  GPU with them; lowering the workers' priority (nice 19) changed nothing.
- Per shard: decode (PyAV and soxr in the workers, and the PCM back to the main process)
  ~48 CPU-s, ~450x real time per CPU; augment and Mimi on the GPU ~7 s (Mimi ~6 s); loading the
  shard and its source ~0.5 s; the save ~0.4 s.
- `encode` overlaps shards: it loads the next shard and queues its clips, shortest first, before
  the GPU encodes the current one, so the workers decode them meanwhile. Each clip's PCM is
  freed once encoded; on an error the queued clips are cancelled. The next shard's load sits on
  the GPU's path (~0.5 s, ~2%): a loader thread hid it, but needed more care to fail cleanly. An
  encode that overtakes a running fetch still waits for the next source before encoding the
  current shard, so a failure there shows only once the fetch has written it.
- Seconds per shard after the first (which adds ~10 s of start-up):

  | Loop | Workers | s/shard | Real time |
  |---|---:|---:|---:|
  | serial (the first encode) | 24 | ~14 | ~1,450x |
  | pipelined | 24 | ~10 | ~1,900x |
  | pipelined | 7 | ~8 | ~2,450x |
  | pipelined | 6 | ~8 | ~2,450x |
  | pipelined | 5 | ~8.5 | ~2,350x |
  | pipelined, no loader thread (current) | 6 (default) | ~8 | ~2,400x |

  `--workers` defaults to the CPU budget less one (6 here): the CPUs the process may run on,
  capped by the cgroup v2 or v1 quota. It keeps the throttle off (2 s summed over all threads in
  a run of six shards, against 21 s with 7 workers). Lower it when another CPU-bound job shares
  the container (two encodes split the budget: 3 workers each here); a fetch is network-bound
  and needs no share. It reads the quota at the cgroup namespace's root, which a container has
  as its own; under `--cgroupns=host` or a parent cgroup's quota it misses the limit, so check
  the printed worker count.
- `--batch-seconds` stays 640: 1280 was no faster and changed some codes (the kernels depend on
  the batch's shape), and 2560 ran out of memory beside another encode's 32 GB.
- Decode and the GPU now take about the same time per shard, so the next gain needs more CPU (a
  pod with a larger quota) as well as a faster GPU step.

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
- The decoder (pod, CUDA, 2026-11-04). The first encode decoded with `eval_sets.decode_opus`
  (sphn, cut at `vi_duration_s`) and stopped on `cv27-00036`: 8 of its 3,006 selected rows, all
  short with a frame-aligned stored duration, decode in PyAV to 312 samples past it, one frame
  more than the cut; PyAV's frame counts match the cache on every row of 9 shards checked. Mimi
  codes of 64 rows against the cached clean codes, all 16 codebooks:

  | Shard | PyAV | sphn |
  |---|---:|---:|
  | `cv27-00036` | 99.95% | 99.97% |
  | `libriheavy-medium-00100` (Mac voiced) | 99.95% | 7.4% |
  | `libriheavy-medium-00260` (anquachdev voiced) | 99.97% | 9.1% |

  On Libriheavy, libopus (sphn) and FFmpeg's decoder (PyAV) differ: 2 samples apart, then up to
  0.45 in amplitude. The 36 `cv27` shards written before the switch (`cv27-00000`–`00035`) stay:
  over every selected row of them (Mac, av 15.1.0 and soxr 1.1.0, PR #52's review), the old
  decode has PyAV's sample count and, after soxr, lies under 1e-6 from it.

## Evaluation

`training.prep.eval_sets --sets en-vi-val-noisy` writes `en-vi-val-noisy/{cv27,libriheavy}.jsonl`
from the built `en-vi-val` rows: every clip under `Bank.draw(f"0:{id}")` (always applied, unlike
training's 60%), as 24 kHz WAV, with the drawn `conditions` in the row. It resamples the clean
48 kHz clip with `sphn.resample`, as `eval.py` resamples the clean set, so the two sets differ by
the conditions alone. (Every evaluation set reaches Mimi through `sphn.resample`, training through
soxr: a residual-codebook difference that predates this set.) Scored against the clean set on the
same checkpoint it shows what the channel costs. It is not in the trainer's default
`--eval-pairs`; the second epoch passes it.

Measured on the Mac (`eval.py`, MPS, batch 16), the first run at step 63,000 and at its final
step 131,553, 2026-11-04 (chrF; "Both" is corpus chrF over the 360 rows):

| Set | Common Voice | Libriheavy | Both (360) | Final: CV · LH · both |
|---|---:|---:|---:|---:|
| `en-vi-val` (clean TTS) | 39.4 | 42.2 | 41.5 | 42.0 · 43.9 · 43.4 |
| `en-vi-val-noisy` | 34.4 | 37.5 | 36.7 | 34.9 · 39.6 · 38.4 |
| FLEURS dev (real speech) | | | 23.5 | 24.9 |

The second run's own validation translates its five sets the way `eval.py` now batches them:
one pass in its order, batch 64, `--seed 43` ([sft-evaluation](sft-evaluation.md#side-tools)). Run
that way on the Mac, the first run's final model scores FLEURS dev 25.1, `en-vi-val` 42.2 · 44.0 ·
43.5 and `en-vi-val-noisy` 34.6 · 39.2 · 38.0 (CV · LH · both), within 0.4 of the table above.
These are the baseline for the second run's validation, up to Mac/CUDA numerics. On the H100 with
the same protocol, the first run's final model scores FLEURS dev 24.6, `en-vi-val` 42.6 · 43.5 and
`en-vi-val-noisy` 34.0 · 40.2 (CV · LH).

The second run's own validation at step 18,000 (2026-11-04) scores:
- FLEURS dev 27.5, `en-vi-val` 43.0 · 43.7 and `en-vi-val-noisy` 38.4 · 41.5 (CV · LH);
- 268 of 361 FLEURS rows ended, where the first run ended at most 64 at any validation.

Its commit still gates EOS at 91% (removed from the gates on 2026-11-04), so it promotes nothing.

- By condition, against the same rows clean: reverb −6.1 chrF; SNR under 15 dB −8.8 (15 dB
  and up −2.9); phone band −8.9, 8 kHz −7.8, 16 kHz −4.7.
- The channel so costs ~5 of the ~18 points between clean TTS and FLEURS. The rest is real
  speech and FLEURS's content (encyclopaedic sentences, names, numbers), which noise cannot
  teach: the real-Vietnamese tranche (plan stage 1.3) is the larger lever.
