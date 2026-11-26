# YODAS3 in the tranche

Read when touching `training/tranche/yodas.py` (YODAS3's Vietnamese ingest, `ingest segments
yodas`) or running the YODAS pilot. The corpus's layout and captions are in
[vi-corpora](vi-corpora.md#yodas3); the stages after ingest in [real-vi-tranche](real-vi-tranche.md#pipeline).
Back to [CONTEXT.md](../../CONTEXT.md) · [Rows](#rows) · [Gate](#gate) · [Speed](#speed) ·
[Running](#running) · [Voice](#voice) · [Pilot](#pilot)

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
  a tar's ASR takes ~145 s, the bound once cutting keeps up.
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
- VieNeu from shard 0004 on (2026-11-26 17:00 UTC): a shard every ~8 min, ~440× real time, 7.2–7.6%
  of rows still broken ([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)).

## Pilot

~1,000 h kept (tars 0000–0022), translated, voiced and cached on the RTX 5090 box, then measured
by two 20k-update SFT arms from `twoway_final` (with and without the YODAS cache) on an 80 GB GPU.
The card of `anquachdev/en-vi-speech-20k` lists `yodas/` with its config and YODAS v3's terms
(CC BY 3.0 audio, attributed; 2026-11-25). Ingest and translation are done (264,665 rows, 1,012 h;
[tranche-translate](tranche-translate.md#running)); voice stopped at four shards ([Voice](#voice)).
