# Vietnamese voice-transfer TTS

Read when touching `training/prep/tts_en_speech.py`, `training/prep/vieneu_mlx.py` or the
`vi_audio` of `/Volumes/data/datasets/en_vi_speech`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Purpose](#purpose) · [Recipe](#recipe) ·
[Voice transfer](#voice-transfer) · [Engine](#engine) · [Speed and memory](#speed-and-memory) ·
[Running](#running)

## Purpose

Step 3 of phase 1 in [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md): each `en_speech` row of
the EN-VI dataset ([en-vi-speech-dataset](en-vi-speech-dataset.md)) gets its Vietnamese line
(`text_vi`) spoken in the voice of its own English clip. The user chose VieNeu-TTS v3 Turbo on
the Mac on 2026-10-23, after Fish Drama 3 on OpenRouter came to ~$1,430 for the corpus (715M
UTF-8 bytes at its discounted $2 per million) and no cheaper API cloned voices.

## Recipe

- VieNeu-TTS v3 Turbo, `update/` weights at `61b85e3`, in MLX (`vieneu_mlx.py`). The `vieneu`
  3.8.3 package (installed with `--no-deps`) supplies text normalization, chunking,
  phonemization, the frame caps and the x-vector ONNX. The codec is MOSS-Audio-Tokenizer-Nano at
  `6aa02b0` through mlx-audio 0.6.1. Also needed in the conda base Python: mlx 0.32.4,
  `sea-g2p`, `kaldi-native-fbank`, `soxr`, `onnxruntime` and `av` (PyAV, for Opus).
- The reference is the row's own English clip, edge silence trimmed, first 8 s: it gives the
  x-vector (the speaker anchor added to every input) and the in-context reference codes.
  Upstream's denoiser is skipped: 3.6 CPU-seconds a clip, and no better voice match or CER.
- Text is cut into upstream's chunks of at most 256 characters (a longer single word stays one
  chunk), joined with upstream's pauses. Sampling, caps and retries are upstream's:
  temperature 0.8, top-k 25, top-p 0.95, repetition penalty 1.2 over 64 frames. A chunk stops
  at EOS (kept) or at upstream's `max_expected_frames`, at most 300: `24 + 2 x phoneme chars`,
  but 13 frames plus 5 per syllable after the first for chunks of up to 4 syllables, and 13 for
  a lone emotion cue. One under 0.25 frames a phoneme character is generated again, twice at most, keeping the
  longest. Upstream's babble guard for chunks of a few syllables is not ported.
- The reference keeps the whole clip when trimming would leave under 1 s (a lone click; a
  ~30 ms reference gave a NaN x-vector and a buzz). A text with nothing speakable, a chunk past
  the model's 1024 positions or a non-finite x-vector stops the run, and a rerun stops on it
  again. No row of the corpus does (checked 2026-10-24: the longest chunk
  needs 704 positions).
- Output: the dataset shard itself, rewritten with `vi_audio{bytes,path}` (48 kHz mono Ogg Opus,
  libopus complexity 3, ~32 kbps, ~13 MB an hour) and `vi_duration_s` filled; the timbre flag
  stays empty until speaker similarity is measured. One local copy: the user asked on 2026-10-24
  for the TTS output to go up continuously, so it fills the published columns instead of a
  separate folder, and the 88 shards written to one before were folded in. Vietnamese runs
  shorter than the English: 0.68x on Common Voice, 0.93x on Libriheavy validation.

## Voice transfer

Measured 2026-10-24 on 120 validation rows (60 Common Voice, 60 Libriheavy speakers):
WeSpeaker ResNet34 cosine to the English clip, and Whisper large-v3-turbo CER against `text_vi`.
Another clip of the same speaker scores 0.77–0.79, a different speaker 0.09–0.11.

| Reference | Similarity CV / LH | CER CV / LH | Rows over 15% CER |
|---|---|---|---|
| own clip, fp16 codec encoder (the recipe) | 0.501 / 0.555 | 0.123 / 0.030 | 10.9% |
| own clip, fp32 codec encoder | 0.515 / 0.547 | 0.087 / 0.036 | 11.8% |
| own x-vector, nearest preset's codes | 0.467 / 0.518 | 0.069 / 0.025 | 8.4% |
| nearest preset alone | 0.126 / 0.144 | 0.042 / 0.039 | 5.9% |

Sampling noise is large at this size: the two own-clip rows differ only in 3% of reference
codes, yet their Common Voice CER means differ by 0.036 (their medians favour fp16).

- VieNeu's only voices are its 25 Vietnamese presets (x-vector plus ~90 reference frames); the
  nearest to an English speaker averages cosine 0.39 by x-vector. A preset reference trades
  voice match for intelligibility, so the recipe keeps the row's own clip. VieNeu's Vietnamese
  speech sets for a bigger bank are gated: 140 h (24 GB) and 1000 h (142 GB).
- Another clip of the same speaker as the reference matched worse (0.48) than the row's own.
- High-CER rows: idioms translated freely ("A chip of the old block"), English names, a few
  chunks that repeat themselves before stopping.

## Engine

- Backbone: 12 Qwen3 layers (768 wide, 12 query and 4 KV heads); the acoustic decoder samples
  16 MOSS codebooks a frame with one layer; bf16. In fp32 with greedy sampling, its codes equal
  upstream's PyTorch batched engine (`V3TurboBatchEngine`) frame for frame, refills included.
- `Generator` batches continuously: 128 slots; each slot's KV cache row holds its prompt from
  column 0 and one column per frame. Every 8 frames the host collects stopped chunks and
  prefills queued prompts into the freed slots. Static batches kept 47% of slots busy, since
  nearly every cap is 300 frames and a batch ran until its longest chunk ended.
- Prompts are ~295 positions (phonemes, then 100 reference frames), so prefill is about a third
  of generation time; chunks are queued sorted by prompt length so prefill groups pad less.
- Codec: mlx-audio builds a (T, T) mask and attends every key, though each layer sees only its
  last 4–10 s. `WindowedAttention` computes the band block by block: decoding 2.7x faster on
  20 s rows at half the memory, audio within 2e-5 and reference codes identical.
- The codec transformers run in fp16 (quantizer fp32): decoding 1.6x faster, audio 53 dB from
  fp32's (bf16: 35 dB); encoding 1.5x faster with 96.8% of reference codes the same and no
  change in similarity or CER beyond noise (table above).
- MLX keeps freed buffers; the codec's shapes change every call, so its cache reached 27 GB
  while 2 GB was in use. `Synthesizer` caps the cache at 2 GB; 6 GB was no faster.
- Tried without gain: batch 256 (+5%, twice the KV memory), the codec on a second MLX stream
  beside generation (the GPU serialized them), utility QoS for the CPU workers (they fell
  behind: 41x), `nice` (no effect).

## Speed and memory

- Bench (1,500 Common Voice and 600 Libriheavy validation rows, 4.2 h of Vietnamese) on the
  M4 Pro with 48 GB, nothing else busy, 2026-10-24: 131x real time. Static batches with
  full-mask fp32 codec calls, shard by shard, ran 43x on it.
- It is GPU-bound. GPU time on Libriheavy splits ~53% generation, ~28% decoding, ~19% reference
  encoding; Common Voice references last about as long as their Vietnamese, so encoding weighs
  more there.
- The thread feeding the GPU competes for performance cores: six CPU-bound processes beside it
  slowed generation 22%. Runs beside the publisher's `upload` (hashing) measured 35–67x.
  libsndfile's Opus (complexity 10, 41x real time a core) made the workers the bottleneck;
  libopus at complexity 3 encodes at 250x for the same size (log-mel distance 1.41 dB to the
  source, against 1.24 dB).
- Footprint: the main process ~4–6 GB with its Metal buffers (`footprint -p`; `ps` RSS omits
  them), the six workers ~0.5 GB each. Waiting in memory: the dataset shard being read, at most
  256 enrolled rows (1.5 MB of reference audio each), a decode batch or two of float32 audio
  (~130 MB each, queued behind the enrollments) until encoded, and the Opus bytes of each shard
  not yet written.
- `kill` (SIGTERM) exits through the pool's shutdown, so the workers stop too; queued rows are
  dropped, and only unwritten shards are lost.
- Rewriting a shard (read, fill, write, read back, flush) takes ~1.5 s on the main thread, once a
  shard; the publisher's `upload --follow` beside the run (1.5 GB) left it at ~95x.

## Running

Commands are in [training/prep/README.md](../../training/prep/README.md). A shard is rewritten
whole once its last row is encoded (through the publisher's `write_shard`, so a crash never
leaves half a shard), and a rerun skips the shards whose `vi_duration_s` has no nulls. It holds
`.tts.lock` in the dataset folder. At ~100–130x, the corpus's ~9k hours of Vietnamese would take
the Mac alone 3–4 days.

Two machines share the work (user's plan, 2026-10-24): the Mac goes top down through the
dataset's shard paths in sorted order (`data/train/cv27-*`, `data/train/libriheavy-*`, then
`data/validation/*`), and another person's own hardware goes bottom up. The Mac's first run
(validation, then a shuffled train order) had voiced 88 shards before, scattered through the
list.

```text
data/train/cv27-00000 ... ─► Mac, top down
                     ... meet ...
... data/validation/*     ◄─ other machine, bottom up
```

- The other machine reads the shards from the Hub, skips those whose `vi_duration_s` has no
  nulls there, and commits each voiced shard back as soon as it is done: the same schema, only
  `vi_audio` and `vi_duration_s` filled, every other column (`cross_lingual_timbre_matched`
  included, which stays empty) as it was.
- The Mac's `upload --follow` downloads a Hub shard voiced there when the local copy is
  unvoiced, after checking that it keeps that contract, and never commits an unvoiced shard. The
  TTS checks each shard again before reading it, `fill` keeps a shard pulled while it was being
  voiced, and a pull keeps a shard voiced here while it downloaded, so the first voicing to reach
  the Mac stays, there and on the Hub. Only a pull and a rewrite (~1.5 s) landing within
  milliseconds of each other can swap that order, either way; both copies still end the same,
  since the next pass commits whatever the Mac holds. Near the meeting point a few shards, ~3–5
  on each side, may be voiced twice: each machine sees the other's work only once it is
  committed, after the 10 min wait plus a pass of 4 GB commits (2–7 min each).
