# YODAS3 in the tranche

Read when touching `training/tranche/yodas.py` (YODAS3's Vietnamese ingest, `ingest segments
yodas`) or running the YODAS pilot. The corpus's layout and captions are in
[vi-corpora](vi-corpora.md#yodas3); the stages after ingest in [real-vi-tranche](real-vi-tranche.md#pipeline).
Back to [CONTEXT.md](../../CONTEXT.md) · [Rows](#rows) · [Gate](#gate) · [Speed](#speed) ·
[Running](#running) · [Voice](#voice) · [Pilot](#pilot) · [Pilot results](#pilot-results)

More GigaSpeech 2 moved FLEURS by nothing in round 4, so the lever past ~41–42 FLEURS dev chrF is
more diverse real Vietnamese speech through the same SFT → GRPO → student pipeline (2026-11-25).
YODAS3's Vietnamese (12.6k h in 44k YouTube videos, none in YODAS v1/v2) is the largest such pool.

## Rows

- `ingest segments yodas` downloads each tar of `espnet/yodas3.1`@`3247e8c` `data/vi/audio/` (the
  next downloading while one is cut) and streams it video by video to the cutting processes.
- A video's caption words (YouTube ASR, timed word by word) only place its rows: runs of words
  ≤ 2 s apart with no tag (`[âm nhạc]`, `[ __ ]`) between them; the longest runs first up to
  20 min a video (many voices and topics rather than a few long videos). Videos where more than
  20% of the cues are tags alone are music; a censored `[ __ ]` inside a spoken cue only splits its run.
  Every run, the short ones too, bounds its neighbours' rows: a run that is cut at its lead or tail
  window, one that is not at its words.
- Each run is cut into 5–22.4 s rows (`ROW_S`), each ending at the widest pause between two words
  (≥ 0.33 s between their starts) that keeps the row in range, preferring cuts that strand no tail
  under 5 s, at the quietest 20 ms frame of that pause. A run starts at the quietest point of the
  0.5 s before its first word and ends at that of 0.25–1.2 s after its last word's start.
- Only the chosen runs are decoded (a seek to 1 s before each run's span; it lands within 10 s
  before it, one WebM cluster, on all 8,771 runs of tar 0002), and a run longer than what is left
  of the cap is cut at its last pause that fits (`capped`): a livestream can be one run of hours,
  which decoded whole held 15+ GB in one cutting process (48 kHz stereo) and cut hours of rows that
  the cap then dropped.
- Every row is heard by the ASR server (Qwen3-ASR-1.7B, `filters.Hearer`, 16 kHz WAV, language
  detected) and the transcript becomes the row's `text`; `<source>/asr_vi/<tar>` stores it again,
  its language and its CER against the caption over `spoken`. So `filters asr` has nothing to hear
  for yodas, and voice.py's and the cache build's `vi_gate` see the same CER as ingest did.
- `doc` is the video (`yodas/<tar>/<video>`), `speaker` none, `seq` the video's index in the tar
  × 10,000 + its row; rows are 24 kHz Opus (each source keeps its own rate).
- Tar 0000 (551 videos, 162 h): 60 music, 28 without caption words, 12 without a run of 5 s;
  14,993 rows cut (54 h). The pilot's music count took a tag-only cue twice when its tag also came
  as a word, and censored words as tags.

## Gate

`filters.vi_gate` at the shared `MAX_CER` 0.15, the caption standing as the transcript. On tar
0000's 14,993 rows: 1,122 (7.5%) not Vietnamese (Thai videos captioned as Vietnamese; the caption
is garbage there), 327 (2.2%) where the ASR hears a nghìn/triệu/tỷ the caption lacks (rice-price
talk: caption "800", ASR "tám ngàn một"), and the CER gate. Of the 13,544 other rows:

| CER ≤ | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rows | 43% | 63% | 75% | 84% | 89% | 92% | 96% |
| Hours (of 50.2) | 22.5 | 32.5 | 38.7 | 42.7 | 45.2 | 46.6 | 48.3 |

- Sampled by band: at 0.15–0.25 the ASR's transcript reads well and is usually better than the
  caption; at 0.25–0.4 mostly; above 0.4 a mix, some of it garbage (music, noise, overlap).
- Kept at 0.15: a looser cut for yodas needs a per-source threshold in ingest (which keeps only
  the rows the gate passes), voice.py and the cache build, and every tar ingested again; it gains
  nothing downstream (translation, voice and cache cost per kept hour, not per
  tar): 1,000 h is ~26 tars at 0.15 against ~22 at 0.25, ~13 min more of ingest, from a pool of ~80
  tars (~3,000 h at 0.15). Loosen it only if the pool runs short.

## Speed

On the RTX 5090 box (19.2 CPUs, 2026-11-25), one tar (8.7 GB, ~15k rows) through `tar_shard`:

| Run | Change | s a tar | Cutting CPU-s | ASR server CPU-s |
|---|---|---:|---:|---:|
| A | first version, 12 cutters, 64 requests in flight | 467 | | |
| B | a video settled as it finishes, not in order; 256 in flight | 294 | | |
| C | (B, measured) | 287 | 2,207 | 1,506 |
| D | runs decoded alone; the server at 1 OpenMP thread; 13 cutters | 193 | 1,592 | 426 |
| E | `ingest segments yodas --shards 0:3`, 16 cutters, a tar written while the next is cut | 196 | | |
| G | runs capped (tar 0002; its cutting 108 s, the ASR the bound) | 186 | 1,621 | 436 |

- A and B: the main loop waited on the oldest video, so one hours-long video idled the other
  cutters (GPU at 0% and 1.5 CPUs for stretches).
- The ASR server alone (`--max-num-seqs 256`, 4 API processes) hears ~104 rows/s (~1,390× real
  time, SM busy ~90%): 78 at 64 requests in flight, 104 at 256, no more at 512 seqs, 8k or 16k
  `--max-num-batched-tokens` (2,048 by default), 6 API processes or a compiled audio encoder. So
  a tar's ASR takes ~145 s, the bound once cutting keeps up. Those are bf16's; the server runs FP8
  from 2026-11-28 (1.39× on voicing's checks), which keeps ~0.5% fewer rows at the gate, mostly sung
  Vietnamese it tags Chinese ([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)).
- At 4 OpenMP threads the server spent 170 CPU-s per 3,000 rows, at 1 thread 71, at the same
  rows/s; `restart.sh asr` now runs it at 1.
- Cutting, one core alone: decoding 6.5 CPU-s per hour of video (ffmpeg's Opus decoder; libopus
  is slower and gives int16), the Opus encode of the rows ~5.7 per hour, scipy's resampling ~1
  (soxr would be 0.3). With every core busy each runs ~1.5× slower.
- After G the three stages a tar goes through take about the same: its download ~170 s (8.6 GB at
  ~50 MB/s), cutting ~110 s (16 processes, the CPU quota full), the ASR ~165 s (17k rows). So
  `segments` cuts a tar while the next downloads and the one before is heard to its end and
  written: ~3 min a tar, 1,000 h kept in ~22 tars. Peak memory: 1.8 GB a cutting process, 7.7 GB
  the main one (the WAVs queued for the ASR).
- Run E died on an ASR transcript holding a run of hundreds of digits (`num2words` overflows):
  `filters.spoken` now reads a run of more than 15 digits digit by digit.


## Running

On the box, the ASR server up first (`restart.sh asr`; for ingest alone it may take 0.8 of the GPU,
e.g. `--gpu-memory-utilization 0.8 --max-num-seqs 256 --max-model-len 4096`), from the repo root
with `HF_TOKEN_AN` in the environment:

```text
/workspace/envs/nanovox/bin/python -m training.tranche.ingest --root <root> segments yodas --shards 0:23
```

- It writes `<root>/yodas/asr_vi/<tar>.parquet`, then `<root>/yodas/segments/<tar>.parquet`, and
  pushes both folders after each tar, asr_vi first, in one thread. A rerun first pushes what an
  earlier run wrote but did not push, and skips every tar whose segments are written or on the Hub:
  cut again, a tar gives other rows (the ASR's batches vary) under English already made from the
  first. A tar with segments but no asr_vi, here or on the Hub, stops the run. So a check
  (`--no-push`, another `--root`) takes a tar not yet on the Hub (0023 on).
- A push uploads the part's whole folder: two boxes must not ingest the same tars.
  Translation and voice find shards on the Hub, so they follow the pushes (the 5090 box uploaded at
  ~2–5 MB/s, ~0.65 GB of segments a tar).
- `--workers` is the cutting processes, the CPU quota less 3 by default (16 of 19.2 on the 5090).

## Voice

Shards 0000–0003 by VoxCPM2 ([tranche-tts](tranche-tts.md)) on the RTX 5090 (2026-11-25/26), the
rest by VieNeu-TTS v3 Turbo, which `voice.py` now runs
([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)). VoxCPM2 alone on the GPU, 2,999 YODAS
rows (RTF is wall time over English voiced):

| Setup | RTF | The process's GPU memory |
|---|---:|---:|
| 512 sequences, `--gpu-memory 0.5` | 0.0059 | 29.6 GB |
| 1,024 sequences | 0.0062 | |
| the VAE's `BATCH_FRAMES` 4,000 | 0.0061 | 26.0 GB |
| expandable segments, 0.5 (762 KV blocks of 256 tokens) | 0.0058 | 20.5 GiB |
| expandable segments, 0.6 (1,119 blocks) | 0.0055 | 23.6 GiB |

- A YODAS row needs ~430 tokens of KV, 2–3 blocks. At 0.9 the VAE's warm-up ran out of memory.
- In production, beside the ASR server at 0.22 of the GPU, voice at `--gpu-memory 0.58`
  (0.57 under MPS); per shard, from the log's cumulative figures:

  | Run | Shard | English voiced | RTF |
  |---|---|---:|---:|
  | plain processes | 0001, the first (warm-up and its prompts' encode in) | 53.9 h | 0.0088 |
  | plain processes | 0002 | 46.1 h | 0.0075 |
  | under MPS (`nvidia-cuda-mps-control -d`) | 0000, the first | 46.5 h | 0.0075 |
  | under MPS | 0003 | 47.8 h | 0.0070 |

  The gap to the bench is the ASR server's share (~8% of the GPU), each shard's prompt encode (67–86 s
  at ~93% busy, not worth a thread) and each shard's tail of long rows. The GPU sat 93% busy at
  ~560 W, touching the card's 575 W limit; the CPU used 2.2 of 19.2. MPS let the two processes'
  kernels overlap: 7% on a warm shard, 15% on the first.
- 4.6–4.9% of each shard's rows stay broken (`MAX_TRIES` 1; the cache drops them).
- One row failed every translation try (`yodas/0020/dc7d71a6a9cd974feba4ce55fc036641/000`, a
  grammar lesson whose English must quote "sẽ"); its `text_en` was set to "" by hand, so voice
  skips it and the rest of shard 0020 can be voiced.
- Stopped 2026-11-26 00:55 UTC after shards 0000–0003 (46,501 rows, ~194 h of English, pushed):
  the YODAS scale-up (~20k h) needs a TTS at ~1,000× real time (RTF 0.001) or better, at
  VoxCPM2's accuracy, so the newer models are measured first (user, 2026-11-26). None reached
  it; optimized, VieNeu runs 3.1× VoxCPM2 and is chosen for the scale-up by the user's 3× rule
  ([tranche-tts-scaleup](tranche-tts-scaleup.md#finding)).
- VieNeu from shard 0004 on (2026-11-26 17:00 UTC): 0005–0021 in 2.33 h (919.9 h of English,
  ~395× real time end to end), 6.1–8.7% of rows still broken
  ([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)).

## Pilot

~1,000 h kept (tars 0000–0022), translated, voiced and cached on the RTX 5090 box, then measured
by two 20k-update SFT arms from `twoway_final`, with and without the YODAS caches, on the 5090 too
(user, 2026-11-26: "sft can run on 5090 too for your validation"). The voicing continues past the
pilot only if the arm with YODAS improves on the one without (user, 2026-11-26), and only once
the user has the pilot's results and says so, whatever they show (user, 2026-11-27: "either way
stop to report and merge after validation done, dont start 20k h yet").

- Caches: `restart.sh cache` builds both directions (`en-vi` second, deleting the parts it has
  read); every YODAS row needs its pair in the other direction, which times its source
  ([sft-training](sft-training.md#target-timing)). Then `streams --drop-shards` on
  `cache/yodas` and `cache/en-vi/yodas`. Built 2026-11-26 (2,129× and 2,180× real time, 44 min
  each way, 10.2 GB packed): vi-en kept 85.0% of VoxCPM2's rows (0000–0003) and 72.6% of
  VieNeu's, en-vi 83.8% and 71.2%. The gap is mostly length: VieNeu's ~12% longer English puts
  18.8% of its rows past the trainer's 280 frames against VoxCPM2's 8.7% (~10 of its 12 points);
  the rest is the ASR check, failed on 7.2% against 4.8%. For a scale-up on VieNeu, faster speech or shorter rows at ingest would win back
  ~10% of the rows it voices, beside the enrolment overlap's ~30% of its time
  ([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)).
- Arms: `restart.sh yodas-control`, then `restart.sh yodas-pilot`: `two-way-5090`'s rows and route
  (`--bf16-adamw`, 14 checkpointed layers, 4,608 frames), `--max-steps 20000` with the cosine
  over them, seed 47, validated at the end; the pilot adds `cache/yodas` and `cache/en-vi/yodas`:
  359,254 rows past the loader (3,399 unpaired, 29,307 over 280 frames and 177 on a reassigned
  slot left out), 3.5% of its 10,243,579, and 1,273.6 source hours, 6.5% of its 19,586.8 (the
  control's 18,313.2). 20,000 updates are ~7% of an epoch (285,576 batches; the control's
  268,096), so an arm reads ~1,370 h, the pilot's ~90 h of them YODAS. The control trained 3.55 h
  at 0.64 s/update, then validated for 8 min (2026-11-26 21:35 to 11-27 01:17 UTC).
- Judged on the test sets as [the level-stage test](exposure-bias-runs.md#the-level-stage-test)
  scored them (`eval.py` seed 43, batch 64, one call a model, paired bootstrap seed 43), the two
  arms and `twoway_final` together.

The card of `anquachdev/en-vi-speech-20k` lists `yodas/` with its config and YODAS v3's terms
(CC BY 3.0 audio, attributed; 2026-11-25). Ingest and translation are done (264,665 rows, 1,012 h;
[tranche-translate](tranche-translate.md#running)); all 23 shards are voiced ([Voice](#voice)).

## Pilot results

Tested 2026-11-27 on box 62513802: each arm's `model_step020000` (also its best) and
`twoway_final`, `eval.py --text-only` seed 43, batch 64, through the input level stage; every set
passes the health gates for all three. Raw outputs, logs and scripts:
`/Volumes/data/hibiki-zero-evals/box-62513802/yodas-pilot-2026-11-27/`; the models are in the
recovery repo's `yodas_control/` and `yodas_pilot/`. The pilot trained 3.55 h (01:19 to 04:52
UTC), as the control did.

- On real Vietnamese, what YODAS is for, the pilot showed no gain: pilot − control +0.05 [−0.30,
  +0.41] on the mean of FLEURS test, both MultiMed-ST sets and `giga-heldout` (−0.06 with FLEURS
  dev added, which itself fell 0.53). The control's 20,000 updates on the old mix gave those four
  +0.51 [+0.13, +0.88] over `twoway_final`, the pilot's +0.56 [+0.19, +0.90].
- The VI→EN mean's +0.64 [+0.19, +1.10] is all on the four VieNeu-voiced `en-vi-val*` sets,
  +1.23 [+0.36, +2.17]; its largest, `en-vi-val/cv27` (+2.12), is the control's dip (−2.05 from
  `twoway_final`; the pilot +0.08).
- EN→VI: pilot − control −0.51 [−1.50, +0.46] on its three test sets; against `twoway_final` the
  pilot is worse on FLEURS-en test (−1.48 [−2.59, −0.35]) and dev (−1.78) and
  `en-vi-val-en-noisy/libriheavy` (−1.62), each interval below 0.
- The mean of the 11, +0.33 [−0.07, +0.78], is no clear win, and the validation's selecting
  chrF prefers the control (37.76 against 36.85).
- One seed an arm: the intervals cover row sampling only, the arms read different streams of the
  old mix, and the control's own updates moved single sets by ±2 chrF (−2.05, +1.81). At ~90 h
  read, YODAS shows no effect on real Vietnamese.
- Both arms finish later than `twoway_final`, End Offset up 0.4–1.6 s on every set but
  `giga-heldout` (0.1–0.2 s), the pilot within 0.5 s of the control on all but
  `en-vi-val-en-noisy/cv27` (+1.2 s): the extra SFT moved it, not YODAS.
- The group means above are `pilot_groups.py`'s (`logs/pilot_groups.txt`), bootstrapped as
  `pilot_compare.py` does. Both runs record commit `5ee7f64`, this branch before its rebase onto
  #200 and #201; its `restart.sh` is this branch's.
- Not acted on: the user decides on the scale-up (2026-11-27).

Test sets (chrF / BLEU / median End Offset s / length ratio):

| Set | `twoway_final` | control | pilot |
|---|---|---|---|
| fleurs-vi/test | 39.90 / 12.10 / 2.70 / 1.15 | 40.39 / 12.29 / 3.28 / 1.13 | 40.68 / 12.16 / 3.36 / 1.17 |
| multimed-vi/select | 37.83 / 11.95 / 3.87 / 1.23 | 38.08 / 12.27 / 4.40 / 1.21 | 37.99 / 11.95 / 4.70 / 1.23 |
| multimed-vi/final | 37.73 / 11.84 / 4.14 / 1.20 | 38.25 / 12.40 / 4.70 / 1.18 | 38.51 / 12.32 / 4.94 / 1.21 |
| giga-heldout/test | 63.63 / 38.48 / 2.46 / 1.13 | 64.39 / 39.30 / 2.54 / 1.11 | 64.14 / 38.71 / 2.64 / 1.12 |
| en-vi-val/cv27 | 46.16 / 18.44 / 3.22 / 1.11 | 44.12 / 18.87 / 4.03 / 1.10 | 46.24 / 19.56 / 3.91 / 1.13 |
| en-vi-val/libriheavy | 43.87 / 19.57 / 3.97 / 1.03 | 44.42 / 19.61 / 5.06 / 1.02 | 45.28 / 20.70 / 5.19 / 1.01 |
| en-vi-val-noisy/cv27 | 40.74 / 13.76 / 3.43 / 1.15 | 41.09 / 13.96 / 3.83 / 1.13 | 42.85 / 14.98 / 4.08 / 1.18 |
| en-vi-val-noisy/libriheavy | 41.98 / 17.26 / 4.02 / 1.02 | 42.41 / 18.30 / 5.16 / 1.01 | 42.59 / 18.33 / 4.86 / 0.99 |
| fleurs-en/test | 32.71 / 13.93 / 3.78 / 1.00 | 31.90 / 13.28 / 4.83 / 0.95 | 31.23 / 13.20 / 4.89 / 0.92 |
| en-vi-val-en/cv27 | 34.06 / 16.65 / 1.20 / 1.08 | 35.00 / 16.25 / 2.26 / 1.07 | 34.34 / 17.45 / 2.10 / 1.03 |
| en-vi-val-en/libriheavy | 39.32 / 18.50 / 3.79 / 1.04 | 38.64 / 18.30 / 5.42 / 1.02 | 38.43 / 18.38 / 5.31 / 1.01 |
| fleurs-vi/dev | 41.06 / 11.60 / 2.78 / 1.21 | 42.05 / 12.49 / 3.33 / 1.15 | 41.53 / 12.11 / 3.77 / 1.19 |
| fleurs-en/dev | 32.62 / 14.13 / 3.66 / 0.98 | 31.99 / 13.76 / 4.83 / 0.93 | 30.84 / 13.05 / 4.99 / 0.94 |
| en-vi-val-en-noisy/cv27 | 21.99 / 8.86 / 2.54 / 0.97 | 23.80 / 10.14 / 2.96 / 0.90 | 22.37 / 9.05 / 4.16 / 0.98 |
| en-vi-val-en-noisy/libriheavy | 33.20 / 15.03 / 4.54 / 0.96 | 31.90 / 13.79 / 5.01 / 0.94 | 31.58 / 13.30 / 5.51 / 0.92 |
| **mean of the 11 test sets** (first 11 rows) | 41.63 | 41.70 | 42.02 |
| **mean of the 8 VI→EN test sets** | 43.98 | 44.14 | 44.78 |

Paired bootstrap of corpus chrF (seed 43, 1,000 resamples, 95% interval):

| Set | pilot − control | control − `twoway_final` | pilot − `twoway_final` |
|---|---|---|---|
| fleurs-vi/test | +0.29 [−0.43, +1.05] | +0.50 [−0.31, +1.28] | +0.79 [+0.07, +1.56] |
| multimed-vi/select | −0.09 [−0.57, +0.41] | +0.25 [−0.26, +0.73] | +0.16 [−0.34, +0.66] |
| multimed-vi/final | +0.26 [−0.21, +0.69] | +0.52 [+0.01, +1.00] | +0.78 [+0.25, +1.30] |
| giga-heldout/test | −0.25 [−1.24, +0.79] | +0.76 [−0.24, +1.84] | +0.51 [−0.53, +1.55] |
| en-vi-val/cv27 | +2.12 [+0.06, +4.23] | −2.05 [−4.26, +0.32] | +0.08 [−2.30, +2.38] |
| en-vi-val/libriheavy | +0.86 [−0.57, +2.28] | +0.56 [−1.06, +2.28] | +1.41 [−0.04, +2.75] |
| en-vi-val-noisy/cv27 | +1.76 [−0.26, +3.68] | +0.34 [−2.04, +2.69] | +2.10 [−0.02, +4.27] |
| en-vi-val-noisy/libriheavy | +0.17 [−1.48, +1.80] | +0.43 [−1.15, +1.96] | +0.61 [−1.01, +2.17] |
| fleurs-en/test | −0.67 [−1.84, +0.40] | −0.81 [−1.90, +0.26] | −1.48 [−2.59, −0.35] |
| en-vi-val-en/cv27 | −0.65 [−2.85, +1.54] | +0.93 [−1.34, +3.11] | +0.28 [−1.99, +2.39] |
| en-vi-val-en/libriheavy | −0.21 [−1.88, +1.50] | −0.68 [−2.10, +0.66] | −0.89 [−2.37, +0.68] |
| fleurs-vi/dev | −0.53 [−1.48, +0.50] | +0.99 [−0.05, +2.07] | +0.47 [−0.70, +1.66] |
| fleurs-en/dev | −1.15 [−2.54, +0.36] | −0.64 [−2.07, +0.81] | −1.78 [−3.22, −0.41] |
| en-vi-val-en-noisy/cv27 | −1.44 [−3.24, +0.25] | +1.81 [+0.24, +3.38] | +0.37 [−1.28, +2.03] |
| en-vi-val-en-noisy/libriheavy | −0.32 [−2.00, +1.39] | −1.30 [−2.99, +0.31] | −1.62 [−3.21, −0.14] |
| **mean of the 11 test sets** | **+0.33 [−0.07, +0.78]** | +0.07 [−0.42, +0.56] | +0.40 [−0.01, +0.86] |
| **mean of the 8 VI→EN test sets** | **+0.64 [+0.19, +1.10]** | +0.16 [−0.39, +0.71] | +0.80 [+0.33, +1.33] |

The arms' final validations (update 20,000, seed 47; chrF / BLEU / length ratio / End Offset s;
health passes on every set); selecting chrF (the FLEURS dev pair) control 37.76, pilot 36.85:

| Set | control | pilot |
|---|---|---|
| vi-en validation loss (semantic, content, acc) | 2.6229 (1.7213, 1.6642, 0.669) | 2.6214 (1.7204, 1.6620, 0.669) |
| en-vi validation loss (semantic, content, acc) | 2.3239 (1.4923, 1.5795, 0.683) | 2.3256 (1.4930, 1.5839, 0.682) |
| fleurs-vi/dev | 42.34 / 13.19 / 1.15 / 3.2 | 41.83 / 12.08 / 1.20 / 3.7 |
| fleurs-en/dev | 33.18 / 14.17 / 0.97 / 4.7 | 31.88 / 13.61 / 0.93 / 5.0 |
| multimed-vi/select | 38.07 / 12.12 / 1.19 / 4.4 | 38.47 / 12.11 / 1.23 / 4.9 |
| en-vi-val/cv27 | 44.95 / 16.54 / 1.09 / 4.2 | 45.95 / 18.25 / 1.14 / 4.4 |
| en-vi-val/libriheavy | 44.44 / 20.06 / 1.02 / 6.1 | 44.78 / 20.56 / 1.01 / 6.0 |
| en-vi-val-en/cv27 | 36.28 / 19.32 / 1.07 / 2.6 | 35.30 / 18.11 / 1.04 / 3.1 |
| en-vi-val-en/libriheavy | 38.87 / 18.83 / 1.02 / 5.0 | 39.10 / 19.25 / 1.00 / 4.6 |
