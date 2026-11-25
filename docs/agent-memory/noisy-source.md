# Noisy source

Read when touching `training/prep/augment.py`, `training/prep/noisy_cache.py`, the
`en-vi-val-noisy` or `en-vi-val-en-noisy` evaluation set, a `*-noisy` cache or the rooms of GRPO
and SFT (`streams.run_rooms`, `--room-share`).
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [Conditions](#conditions) ·
[Rooms](#rooms) · [Noisy cache](#noisy-cache) · [Speed](#speed) · [Checks](#checks) ·
[Evaluation](#evaluation)

## Why

Every Vietnamese source clip in the training caches is VieNeu TTS: full 24 kHz band, no room, no
noise. The phone hears real rooms, and FLEURS dev (real 16 kHz recordings) scores far below the
held-out TTS rows: step 63k of the first run had chrF 23.5 on FLEURS against 39.4 on Common
Voice and 42.2 on Libriheavy `en-vi-val`. Stage 1.2 of the
[plan](../vi_en_parity_plan.md#1-data) re-encodes the source under simulated recording
conditions; `en-vi-val-noisy` measures what the channel alone costs.

English to Vietnamese had no noisy source (2026-11-21): the tranche's English is voice-cloned TTS
(~6,000 h, studio-clean), `en_speech`'s is Common Voice's home recordings and Libriheavy's
audiobooks, and the shipped model wrote English for 18 of OneVoice's 100 outside English clips,
the quieter and conversational ones ([two-way](two-way.md#next)). The same re-encode and conditions
now take an `en-vi` cache's English source, and `en-vi-val-en-noisy` measures it (user, 2026-11-21:
"also how about noisy input?"). The `en-vi` copies take `--seed 1`: the selection and the draw key
on the seed and the row id, which both directions share, so the same seed would put the same
conditions on the same rows both ways.

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
- SFT stream gaps hold digital silence by default, so a noisy row sits between silent gaps where
  a real recording stays noisy; `train.py --room-share` fills a share with rooms ([Rooms](#rooms)).

## Rooms

GRPO's streams hold a room's noise in their source wherever nobody speaks, in a `--room-share`
(0.5) of streams; the rest keep digital silence, what a voice gate passes
([grpo](grpo.md#design)); so do SFT's with `train.py --room-share` (default 0, unused so far;
validation keeps silence). The user asked for it (2026-11-14: "can we add artifical silence into
the data so that grpo also penalise translating when user silent?"). Text after a rollout's
first EOS is neither rewarded nor trained, and 92–96% of rollouts write one after the source
EOS, so the room is taught mainly in the lead (0–2 s) and the gaps, a tenth of them 3–10 s long:
the live failure below, a room after speech with no source EOS. Watch `en-vi-val-noisy` in the
run's evaluations: a room ends at every clip, whose own edges are digital silence, a cue a real
room does not give.

- **Why:** a live-like check on the Mac (2026-11-14, MLX, main.py's seed, a primed stream with no
  source EOS and no voice gate) counted the words written in 30 s of quiet after a FLEURS dev
  sentence or after the primer alone: rooms at −55 and −40 dBFS peaks, white −45, pink −40,
  door-and-cough events at −30, all through the level stage.
  - The teacher (GRPO update 1,400, 6 bits) wrote nothing on noise alone (20 runs); after a
    sentence, once the late end of its own translation.
  - The distilled student (update 8,000, bf16) wrote on noise alone in EN→VI 10 of 10 times
    ("Tổng thống Mỹ…", the English primer's topic, even after a 4 s settle; 55 tokens once on
    events), VI→EN 0 of 10. It stayed quiet through the settle's digital silence and wrote
    once noise came. The plain-SFT student (update 48,000) wrote on noise alone VI→EN 10 of 10.
  - GRPO's BLEU already teaches quiet after a source (the teacher's FLEURS dev rows ending more
    than 4 s after it: 136 → 27 VI→EN, 165 → 33 EN→VI), but in digital silence only.
- **Drawn** by `augment.room_tracks`: a background looped from a drawn offset, peaking at U[−65, −35]
  dBFS, and events every 15 s on average, each at most 3 s of a foreground file (up to 8 min long)
  peaking at U[−45, −25], cut edges ramped over 10 ms; then the level stage from the track's start:
  frame peaks at a median −24.8 dBFS (p90 −18.9, events up to −6; 24 rooms of 30 s, seed 44), −38.7
  at its first settings, when codebook 0 took a steady silence value on ~71% of room frames.
- **Encoded** by `streams.run_rooms`: GRPO 256 rooms, SFT 2,048 (a pass draws ~890,000 room streams),
  each as long as the silence track (GRPO's `--frames`, SFT's longest stream), from a fresh state on
  the run's GPU at its start, SFT's before its resume check (TF32 on, so residual codes may differ by
  GPU). `StreamDataset` draws one for a stream from (seed, index), the same on a resume; `assemble`
  puts it in the source's lead, gaps and after the EOS; the target keeps silence. `grpo_launch.sh`
  fetches the bank into `data/rirs_noises`; nothing does for SFT (`augment.fetch_bank`).
- **The level stage runs on the room alone**, so a gap right after speech is lifted at once,
  where live the envelope still falls 6 dB a second after speech and the room stays near its raw
  level for a few seconds: training's gaps are louder than live ones, erring toward quiet.
- **Held out:** every fifth file of each kind (26 backgrounds, 162 events; `quiet_noises(bank,
  held_out=True)`), from the rooms only: the noisy caches (`Bank`) mix every file under speech.
  Re-run the check on held-out rooms after a model's GRPO, and after an SFT with rooms (cold in
  `eval.py` the 3B models write on most, [the level-stage test](exposure-bias-runs.md#the-level-stage-test)).

## Noisy cache

`training.prep.noisy_cache` writes, for each `shard_<name>.pt` of a clean cache split, a shard
with the same rows in the same order, in two steps that share `--out-dir`:

```text
fetch  (network, CPU; runs beside training)  HF parquet ─► .sources/<name>.parquet
encode (--device cuda or mlx)                .sources + clean shard ─► shard_<name>.pt
```

- The source is the cache's (`direction`): `vi_audio` and `vi_frames` in a `vi-en` cache,
  `en_audio` and `en_frames` in an `en-vi` one (Common Voice's MP3 and Libriheavy's Opus decode
  through the same `audio.decode`).
- `fetch` reads the clips from the file each clean shard names as its `source` (`clip_file`): a
  `cache_codes` shard's EN-VI parquet, or a tranche `en-vi` shard's `<source>/en_audio/` part
  beside the segments it was built from (`anquachdev/en-vi-speech-20k`, rows in the segments'
  order; a tranche `vi-en` source is real speech and has no noisy copy). It downloads each file,
  the EN-VI parquets at `paths.EN_VI_REVISION`, 16 at a time, each tried 4 times, or reads it
  from `--dataset-dir`, a local copy of the dataset the cache was built from (the box's 2026-11-21 `en-vi` build of the remaining `en_speech` shards
  uses it to copy each batch before deleting its sources, [tranche-box](tranche-box.md#state)). It
  keeps the id and the source clip's bytes of the selected rows
  (`random.Random(f"{seed}:select:{id}").random() < --fraction`, 0.6), null for the others, in
  `.sources/` under the dataset's column name, and deletes the download: about a fifth of the
  parquet's bytes for a Vietnamese source (its audio is 26–33% of a parquet), ~40–45% for an
  English one (67–74%), so an `en-vi` split's sources take ~2.5 times the room. It pauses while the disk has
  less than `--min-free-gb` (50) free: a trainer on the same disk needs a checkpoint pair
  (37.5 GB) beyond its steady state while the sync still uploads the pair it pruned. Its first run
  writes `manifest.json` (seed, fraction, the cache's direction, the dataset, bank SHA-256); a
  later fetch with other values stops, and `encode` refuses a shard of another direction.
- `encode` re-encodes the rows whose source holds audio:
  - the clip is decoded by the builder's own code, `audio.decode` (PyAV), and resampled with
    soxr, so it has the samples and frame count the cache was built from. Any other decoder is
    wrong here ([Checks](#checks));
  - the clip goes under `Bank.draw(f"{seed}:{id}")`, stored as `source_conditions` (None for a
    clean row);
  - Mimi in float32 encodes the clips shortest first, in batches of up to `--batch-seconds` (640)
    of padded audio: on CUDA moshi's with TF32 matmuls and convolutions, as the clean build (the
    cache build's `cuda_mimi`, `CUDA_TF32`, and `mimi_batch`, [cache-build](cache-build.md#cuda);
    [cache-speed-cuda](cache-speed-cuda.md) has its speed and the codes it moves), right-padded, the
    conditions on the GPU; on MLX
    `mimi_mlx` one clip per forward, the conditions on the CPU while the GPU encodes the batch
    before;
  - the codes replace rows 17–32 (the source codebooks, either direction) up to the source's
    frame count.

  A clip whose frame count differs from the cache's, or a row without its codec EOS, stops the
  run. Text, target codes, frames, delays and both EOS positions are untouched.
- The shard gains `source_augment` (clean shard plus the manifest). The format string stays
  `hibiki_vn_grounded_cache_v2`, so the trainer reads the split as any cache.
- `encode` checks each source against its shard's `source` (the file, and its byte size or row
  count) and its rows' ids.
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
- Seconds per shard after the first (which adds ~10 s of start-up): serial with 24 workers ~14
  (~1,450x); pipelined ~10 with 24 workers, ~8 with 5–7 (~2,400x; the current loop, without a
  loader thread, at the default 6).

  `--workers` defaults to the CPU budget less one (6 here): the CPUs the process may run on,
  capped by the cgroup v2 or v1 quota. It keeps the throttle off (2 s summed over all threads in
  a run of six shards, against 21 s with 7 workers). Lower it when another CPU-bound job shares
  the container (two encodes split the budget: 3 workers each here); a fetch from the Hub is
  network-bound and needs no share, but one from `--dataset-dir` reads and slims 16 parquets at
  a time (37 s for an `en_speech` shard on the box). It reads the quota at the cgroup namespace's root, which a container has
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
- Either direction (the RTX PRO 6000 box, 2026-11-21, PR #153's validation):
  - `en-vi`, `cv27-00016` from `--dataset-dir` at seed 1 and two Libriheavy shards: the clean
    rows came out byte-identical; only rows 17–32 before `en_frames` changed; the selection and
    the conditions match the seeded draws on every row.
  - 48 rows a shard re-encoded with no conditions match the cached codes on codebook 0 for
    99.68–99.95% of frames, and on all 16 for 84–96% (the TF32 near-ties of the
    [cache build](cache-build.md#cuda)).
  - Seeds 1 and 0 selected 2,745 of the same 7,412 rows (37%; independent draws give ~36%),
    none of them under the same conditions.
  - `vi-en` from the Hub, against the published noisy shards made before the change: the same
    selection and conditions, codebook 0 on 100% and all 16 on 99.97–99.99%.
  - The default `--batch-seconds` 640 ran out of memory under a 15% memory cap of the 96 GB card
    (14.25 GiB); 320 fits.
- On MLX (the Mac, 2026-11-21): `vi-en` `cv27-00000` at seed 0 against the published CUDA shard,
  the same conditions on all 2,961 rows, 99.996% of source codes and every first codebook equal;
  the tranche's viVoice `en-vi` shard 0 at seed 1 from the Hub, text, target and EOS untouched and
  443 of 713 rows re-encoded. Its speed on the Mac (2 workers): [cache-build](cache-build.md#speed-on-the-mac).

## Evaluation

`training.prep.eval_sets --sets en-vi-val-noisy` writes `en-vi-val-noisy/{cv27,libriheavy}.jsonl`
from the built `en-vi-val` rows: every clip under `Bank.draw(f"0:{id}")` (always applied, unlike
training's 60%), as 24 kHz WAV, with the drawn `conditions` in the row. It resamples the clean
48 kHz clip with `sphn.resample`, as `eval.py` resamples the clean set, so the two sets differ by
the conditions alone. (Every evaluation set reaches Mimi through `sphn.resample`, training through
soxr: a residual-codebook difference that predates this set.) Scored against the clean set on the
same checkpoint it shows what the channel costs. It is not in the trainer's default
`--eval-pairs`; the second epoch passes it. `--sets en-vi-val-en-noisy` does the same from the
built `en-vi-val-en` (its English clips, `en_audio`; ids `en-vi-val-en-noisy/...`), for English to
Vietnamese. Its conditions differ from the Vietnamese set's because the draw keys on the clean row's
id (`0:en-vi-val-en/<id>` against `0:en-vi-val/<id>`).

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
