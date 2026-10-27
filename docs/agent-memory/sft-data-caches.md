# SFT grounded caches

Read when touching `training/prep/cache_codes.py`, `training/prep/mimi_mlx.py`,
`training/prep/text_timing.py` or `training/paths.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Flow](#flow) · [Builder](#builder) · [Speed](#speed) ·
[PhoMT cache](#phomt-cache) · [Row layout](#row-layout)

## Flow

```text
/Volumes/data/datasets/en_vi_speech/data/<split>/<name>.parquet
  ─► cache_codes (Mac, MLX) ─► <out-dir>/<split>/shard_<name>.pt
       workers: PyAV decode; soxr to 24 kHz (Mimi) and 16 kHz (CTC)
       GPU: frame cap ─► text groups ─► wav2vec2 CTC (text_timing) ─► Mimi (mimi_mlx)
HF phomt-grounded-v2-5s/ ─► data/cache/phomt-grounded-v2-5s (restored, see PhoMT cache)
<out-dir>/train + PhoMT ─► training.finetune.train --cache-dir (several)
<out-dir>/validation ─► --val-cache-dir
```

- The source is the EN-VI dataset ([en-vi-speech-dataset](en-vi-speech-dataset.md)), read
  straight from its parquet shards, the split taken from the folder. No cache is built from it
  yet. FLEURS is dropped.
- `--shards` picks `.parquet` source shards by glob under `data/` (default: every train and
  validation shard, `cv-anquach` included). `cv-anquach` rows have 16 kHz English targets
  ([en-vi-speech-dataset](en-vi-speech-dataset.md#cv-anquach-rows)); to keep them optional, pass
  the other globs and build them into their own `--out-dir`. Put `--out-dir` on `/Volumes/data`: the codes come to
  ~70 GB and the internal disk is too small.
- Grounded-v2 (`hibiki_vn_grounded_cache_v2`) is the only cache format; `CachedCodeDataset`
  rejects anything else:
  - Mimi at 24 kHz / 12.5 Hz;
  - English audio delayed by a draw from U[0, 0.5 × Vietnamese duration], seeded by
    `1234:{split}:{id}` with the duration rounded to the 0.01 s it stores, and inserted as left
    silence;
  - English text word-timed by CTC alignment with `facebook/wav2vec2-base-960h`; rows that
    fail alignment or score below 0.5 are rejected.
- `training/paths.py` holds the repo defaults (`weights/` artifacts), file guards and `durable`.

## Builder

- Gates run cheapest first, and a row is rejected, with its reason, at the first one it fails:
  a decoded clip more than 0.02 s off its stored duration (PyAV decodes a truncated or damaged
  clip without error, only shorter); more frames than `--max-frames` (280, the trainer's cap),
  counted from the decoded lengths as max(English frames with the delay, Vietnamese frames) + 1;
  the text grouping; CTC alignment; token timing. Only kept rows reach Mimi.
- Only data faults reject a row (`ValueError`, and PyAV errors for undecodable bytes). A dead
  decode worker, an MLX error, or a shard whose every row fails stops the run before the shard
  is written, so a rerun rebuilds it rather than skipping lost rows.
- A shard holds `samples` (the row fields plus `source_row`, `data/<split>/<file>#r<index>`),
  `rejected` (`id`, `reason`), `source` (file, bytes) and `max_frames`. Its values are Python
  types and tensors only: the trainer's `torch.load` keeps the `weights_only` default, which
  refuses NumPy scalars.
- A shard is written as `.tmp`, flushed with `F_FULLFSYNC`, then renamed. A rerun skips shards
  that exist, without checking their source again; `--overwrite` rebuilds. A lock
  (`.cache_codes.lock`) keeps a second run off the same `--out-dir`. Run a multi-day build under
  `caffeinate -i -s`.
- Decoding: PyAV for every stream (the Opus clips and cv27's MP3); anything but mono `fltp`
  is rejected, and every clip in the dataset decodes at 48 kHz. soxr resamples from 48 kHz straight to each rate.
  `sphn.read` cannot decode Opus, `sphn.read_opus_bytes` keeps the 312-sample Opus pre-skip, and
  `sphn.resample` delays its output (256 samples at 48 → 24 kHz).
- Mimi: `mimi_mlx.MimiEncoder`, an encode-only MLX port that loads the PyTorch checkpoint. Each
  clip is encoded whole and alone, padded to a multiple of 16 frames (so the compiled encoder
  sees at most 18 shapes; [mlx-inference](mlx-inference.md#memory)), then cut to
  ceil(samples / 1920) frames. Against moshi 0.2.13 PyTorch on CPU in float32, 99.98% of frames
  are equal on all 16 codebooks and every first codebook (65,237 frames of 504 clips; 12,704
  frames of 40 built rows); the rest are near-ties in the residual codebooks.
- CTC: `text_timing.EnglishCTCAligner` runs mlx-audio's wav2vec2 in float32, one clip at a time
  and uncompiled, then a numpy Viterbi (the same frames as the former torch loop). On 252 rows it
  rejects the same rows as PyTorch CPU float32 and gives the same spans on 6,180 of 6,181 words.
- Rejected after measurement (2026-10-27):
  - upstream moshi_mlx 0.3.0 Mimi: no attention mask and tanh GELU, 30% of codes equal on whole
    clips (with both fixed it matches, which `mimi_mlx` does);
  - rustymimi 0.4.1: an attention window of 251 frames, not 250, so codes differ past 10 s;
  - streaming or chunked encoding: differs past 250 encoder frames (10 s);
  - float16 / bfloat16 Mimi: 89% / 45% of frames equal; float16 wav2vec2: 99.3% of word spans;
  - padded wav2vec2 batches: change the emissions, since the base model has no attention mask.
- Upstream (checked 2026-10-27): moshi 0.2.13, rustymimi 0.4.1 and moshi_mlx 0.3.0 are the
  newest releases; the Mimi stack of moshi 0.2.13 equals kyutai's main, and the local Mimi
  checkpoint is the file upstream Hibiki-Zero loads (sha256 `09b782f0…`).

## Speed

Measured on the M4 Pro Mac (10 performance + 4 efficiency cores, 48 GB), 2026-10-27.

- The builder runs ~200x real time on its Mimi audio (both clips plus the delay): a 1,738-row
  validation shard took 170–190 s for 9.97 h, at a 3.4 GB peak footprint. GPU time was
  ~100 s Mimi and ~75 s wav2vec2; the Viterbi took 8 s and waiting for decodes ~1 s.
- The dataset's ~20k h of Mimi audio after the frame cap projects to ~100 h.
- Alone: Mimi 440–460x (PyTorch CPU 80x; PyTorch MPS 300–350x, but only with length buckets:
  per-clip shapes grow MPS memory to 28–30 GB); wav2vec2 241x (PyTorch CPU 86x, MPS 193x). Its 12
  layers already run at ~3.9 TFLOP/s in float32.
- PyAV decodes Opus at ~1,430x real time per core (soundfile 413x); two workers keep the GPU fed.
- Tried without gain: batching Mimi clips (MLX and MPS: none to −10%, with more memory); a 4 GB
  MLX cache; grouping the GPU work by stage; wav2vec2 in CPU processes beside the GPU. With the
  GPU busy, four CPU processes fell from 123x to 20x in total and Mimi slowed ~45%: the CPU and
  GPU share this laptop's power.

## PhoMT cache

- Pooled with the `en_speech` caches by passing it as one more `--cache-dir`. It holds 804 hours
  of English speech (~770 h of them 5–15 s clips), on top of the corpus's 10,486 train hours
  ([en-speech-corpus](en-speech-corpus.md#rules)).
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
