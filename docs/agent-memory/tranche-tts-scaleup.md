# VieNeu's speed on the 5090

Read when speeding up VieNeu-TTS v3 Turbo or its ASR checks on the RTX 5090 (`voice.Voicer`, `vieneu_frame.py`,
`decode_attention.py`, `onnx_graph.py`), or weighing another TTS against it.
Back to [CONTEXT.md](../../CONTEXT.md) · part of [tranche-tts](tranche-tts.md) ·
[Candidates](#candidates) · [Method](#method) · [Results](#results) ·
[Optimizing VieNeu](#optimizing-vieneu) · [In production](#in-production) · [Finding](#finding)

The user's bar (2026-11-26): at least ~1,000× real time (RTF 0.001) on the 5090, at VoxCPM2's
accuracy, chosen by the measures below without a listening test. VoxCPM2's production speed there:
[tranche-yodas](tranche-yodas.md#voice).

## Candidates

- [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) (Apache-2.0, 0.8B MoE with ~0.1B active,
  15 Hz frames of 10 codebooks, 44.1 kHz, 43 languages with Vietnamese; clones from a 192-d speaker
  embedding or a raw reference prefix; Zyphra's offline `TTSLLM` on Mini-SGLang, repo commit a50e141).
- [VieNeu-TTS v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) (Apache-2.0, ~130M,
  48 kHz, trained from scratch on ~10k h of English and Vietnamese; clones from a speaker embedding
  plus MOSS-Audio-Tokenizer-Nano reference codes; the `vieneu` 3.7.1 SDK's batch engine).
- Not measured: Confucius4-TTS (its MaskGCT S2A and BigVGAN stages are multi-step, far from
  1,000×), Audio8-TTS-Realistic (Seed-TTS en similarity 0.663), OmniVoice (CC-BY-NC weights).

## Method

- 2,000 random rows of YODAS shard 0000 that VoxCPM2 voiced in production (7.6 h of Vietnamese
  clips), each cloned from its own clip to speak its `text_en`, alone on the GPU.
- Scored as production checks a row (Qwen3-ASR-1.7B, Whisper's normalizer; broken at WER over 0.2
  or pace off 2×): VoxCPM2's outputs rescored this way agree with their stored `en_broken` on 99.2%
  of rows (3.9% broken against 3.65%). Voice match is the bake-off's WeSpeaker ResNet293-LM cosine
  against the row's own Vietnamese clip; UTMOS22 strong (SpeechMOS) stands in for naturalness.
- Scripts, logs, scores and every output (as 64 kbps Opus):
  `/Volumes/data/hibiki-zero-evals/box-62513802/ttsbench-2026-11-26/`; the optimization round's
  in `ttsbench-opt-2026-11-26/` beside it (its fast VieNeu path: `vieneu_fast.py`,
  `vieneu_cb.py`, `onnx_torch.py`).

## Results

Paired differences are against VoxCPM2 on the rows neither broke, ± standard error:

| System | Broken | Kept WER median | Voice match | UTMOS | Speed on the 5090 |
|---|---:|---:|---:|---:|---|
| VoxCPM2 (production outputs) | 3.9% | 0.025 | 0.687 | 2.59 | 182× bench, ~140× beside the ASR server |
| ZONOS3, speaker embedding | 31.2% | 0.015 | 0.594 (−0.087 ± 0.003) | 3.39 | LM 315× at 256 rows; 184× with embeddings and decode |
| VieNeu v3 Turbo | 7.1% | 0.035 | 0.604 (−0.081 ± 0.002) | 3.18 | generation 326× at 128 rows a batch; 12× with its CPU enrolment |

- ZONOS3 breaks by leaving English mid-line ("Adi tine jam province ay la to comments sa trang…"): a Vietnamese
  voice pulls the multilingual model off English. Its shipped Vietnamese-accent steering direction, or a raw
  reference prefix, brought broken rows to 3–5% and took the voice with them (−0.14 to −0.23 of voice match).
  It was not optimized further (its bench's sm_120 fixes are in the records' scripts).
- VieNeu's batch engine runs each frame as one CUDA graph (`vieneu.v3_turbo_serve.fused`), its repetition penalty
  on the GPU inside it; its batches are static, grouped by phoneme length.
- Both new models speak cleaner audio (higher UTMOS) and take ~10–20% longer to say the same line
  (9.1–9.8 h against VoxCPM2's 8.2 h).
- VieNeu's enrolment (denoise, speaker encoder, codec encode) runs on the CPU in ONNX at 1.3 s a
  clip. Its generation read 92% busy at 270 W, ZONOS3's LM 73% at 246 W, against the card's 575 W.

## Optimizing VieNeu

The user's rule (2026-11-26): optimize both heavily, and VieNeu is chosen if it is at least 3×
VoxCPM2. VoxCPM2 (its VAE compiled in fp16, its LM in nano-vllm's CUDA graphs, 100% busy at 515–535 W) ran
11.29 rows/s at production's 8 steps and 512 concurrent, 11.45 at 768; fewer steps broke more rows (6: 5.8%, 4:
14.6%, against 3.8%) and without CFG rows never stopped. Same 2,000 rows, each system alone on the GPU, timed
from clips in memory to waveforms (load, compiles and warm-up excluded; VoxCPM2's 48 → 16 kHz resample too,
which production runs in its decoder processes); rows/s, since VieNeu says the same lines ~12% longer. Every
speed-up was scored before it counted.

VieNeu, each step adding to the one above:

| Step | Rows/s | Broken | Voice match | UTMOS |
|---|---:|---:|---:|---:|
| SDK batch engine (enrolment on the CPU) | 0.77 | 7.1% | 0.604 | 3.18 |
| enrolment on the GPU, MOSS decode batched and chunked | 14.8 (torch 2.15) | | | |
| continuous batching, 256 slots | 20.6 | 7.2% | 0.603 | 3.18 |
| denoiser compiled | 27.8 | | | |
| frame compiled, top-k-first sampler | 31.4 | 6.8% | 0.604 | 3.17 |
| 512 slots | 32.0 | 7.4% | 0.606 | 3.18 |
| codec encode in fp16, prompt embeddings built ahead | 34.1 | 7.3% | 0.604 | 3.18 |
| denoiser autotuned | 35.5 | 7.7% | 0.604 | 3.18 |

- Enrolment on the GPU: the SDK's ONNX denoiser and speaker encoder in PyTorch (onnx2torch in the
  bench, `training/tranche/onnx_graph.py` in production: on the speaker encoder cosine ≥ 0.99997 to
  onnxruntime, on the denoiser bit-exact to the bench's runner), its Kaldi fbank batched on the GPU
  (`training/prep/kaldi.py`, shared with the WeSpeaker scores: 0.019 ms a clip, embeddings at cosine 0.999999 to the SDK's
  kaldi-native-fbank), clips trimmed to whole 0.25 s and batched by length (81% are the full 8 s).
  Without the denoiser VieNeu runs 38.1 rows/s at the continuous step but breaks 11.9%: it stays.
- MOSS decode and encode run whole: every attention layer of the codec is causal with a window
  (4–10 s), which `voice.window_attention` runs through `flex_attention` with a sliding-window block
  mask, and both paths are compiled. The SDK's whole batch decode builds a dense T² bias and runs out
  of memory, and its chunked `batch_decode` (2 s chunks, a ring cache a layer) ran 4.60 s a 512
  chunks against 0.60 s now, 54.6 dB from fp32's one-row decode (the chunked fp16 55.1; bf16 41).
  Encode 98 → 35 ms for 64 references, 98.4% of the codes equal.
- Shapes: Dynamo specializes a batch of one and a codec stage that fits one flex_attention block
  (under 33 code frames), and each such shape recompiles for 25–50 s. So the encode runs at one
  static shape (64 references padded to 8 s; causal layers keep the padding out of real frames, and
  a clip keeps its whole frames, as `batch_encode`'s lengths), and decode batches pad to at least two
  rows and 33 frames.
- Continuous batching (the bench's `vieneu_cb.py`, `training/tranche/vieneu_frame.py` in production): the SDK's fused CUDA-graph frame with each row's own cache
  position, frame counter and repetition window, a finished row's slot refilled; the SDK's static
  batches run to their longest row (37% of row-frames real). Graph-captured writes need device
  tensors: a host scalar fails capture ("operation not permitted when stream is capturing").
- Denoiser compiled: onnx2torch's modules read shapes from tensors, which Dynamo cannot trace, and
  onnx-simplifier's folding does not fix that. `onnx_graph.py` (the bench's `onnx_torch.py`) plans the graph's shape arithmetic
  in NumPy for each input shape and compiles the tensor ops: 21.3 → 8.9 ms a clip in fp16 (68 dB
  from fp32), a graph per reference length (32; ~14 min the first time, ~1 min from Inductor's
  cache), batches padded to 16 with copies (instance norms keep clips apart).
- Frame compiled: each layer's projections, norms, rotary and MLP, and the acoustic decoder with
  its sampling (attention as math), through Inductor; the cache writes stay eager, since compiling
  the whole frame copies the 1.5 GB cache and runs out of memory. 10.1 → 6.0 ms a frame at 256
  slots, 9.9 ms at 512. The top-k-first sampler draws from the 25 top-k values, the SDK's
  distribution without its sort over 1,024 codes.
- The backbone's attention (`training/tranche/decode_attention.py`): a Triton kernel over each
  slot's live keys only, a program per slot and KV head, its 3 query heads padded to `tl.dot`'s 16
  rows; the SDK's attends all 1,024 positions under a mask. 0.404 → 0.170 ms a layer, the frame
  9.7 → 6.0 ms at 512 slots. The SDK's prompt cache is left-padded: `SlotBackbone.load` (`vieneu_frame.py`) moves
  each row's keys to its slot's start in one gather.
- Codec encode in fp16: 3.13 → 1.51 ms a clip, 97.9% of codes as fp32's (codebook 0 99.9%; bf16
  76%).
- Denoiser autotuned (`mode="max-autotune-no-cudagraphs"`): 8.9 → 7.4 ms a clip, but its first
  compile takes 54 min for the 32 lengths; later runs load it from Inductor's cache.
- The denoiser's 40 InstanceNorms as fp32 sums reduced one axis at a time
  (`onnx_graph.instance_norm`): Inductor reduces `F.instance_norm`'s 841 × T plane in one block a
  channel. 8.0 → 7.2 ms a clip in the default mode.
- Broken rows across VieNeu's steps (6.8–7.7%) and VoxCPM2's two concurrencies (3.8%, 4.5%) differ
  by sampling noise (±0.6 points on 2,000 rows).
- No gain: two processes on the GPU (25.5 rows/s against one's 27.8: without MPS they take
  turns), a second CUDA stream beside the frames (overlaps ~25% of the shorter stage), denoiser
  batches of 32, `cudnn.benchmark`, longer decode chunks or bigger decode batches, and TensorRT
  11.5 for the denoiser (fp16 no faster than the compiled graph and its output wrong, −0.1 dB from
  fp32). Not done: the speaker encoder compiled in fp16 (0.59 → 0.18 ms a clip, but 1.6 s of a
  bench run and 32 more compiled shapes) and the frame's sampler and prefill (~0.7 ms a frame).
- At 35.5 rows/s (56.3 s): enrolment 26.7 s (denoise 16.6, fbank and speaker encoder 3.3, codec
  encode 3.2, trim and resampling 3.5), frames 13.9 s, decode 14.7 s, joining 0.9 s.

Then production's own `voice.Voicer` (2026-11-28, records in
`/Volumes/data/hibiki-zero-evals/box-62513802/ttsbench-5090opt-2026-11-28/`: `vbench.py` feeds it the
same 2,000 rows and times its stages), each step adding to the one above; levers that change the
output were kept only within the bench's noise on all four scores:

| Step | Rows/s | Real time | Broken | Kept WER | Voice match | UTMOS |
|---|---:|---:|---:|---:|---:|---:|
| main (torch 2.8) | 29.6 | 485× | 7.3% | 0.035 | 0.594 | 3.19 |
| torch 2.15 + cu134, the speaker fbank on the GPU, the frame's attention in Triton, MOSS whole and compiled | 50.2 | 827× | 7.3% | 0.032 | 0.593 | 3.19 |
| the denoiser's instance norms by axis | 52.4 | 861× | 6.9% | 0.034 | 0.591 | 3.19 |
| clean references skip the denoiser, the encode at one shape | 58.3 | 962× | 7.2% | 0.035 | 0.595 | 3.18 |
| rejected: 5 s references (`REF_S`) | 69.5 | 1,145× | 6.9% | 0.035 | 0.572 | 3.19 |
| rejected: 5 s references and the gate | 76.8 | 1,262× | 7.0% | 0.035 | 0.575 | 3.19 |

- Paired against VoxCPM2 on rows neither broke, voice match −0.0909 ± 0.0024 on main, −0.0901 with
  the gate, −0.1130 with 5 s references: those lose 0.02 of voice match, ~9 standard errors.
- The gate (`CLEAN_SPREAD_DB`): a reference whose 20 ms frame levels spread at least 40 dB (95th
  minus 10th percentile, `voice.level_spread`) is clean enough to skip the denoiser: ~30% of YODAS's.
  Chosen before its run from the 2026-11-26 bench's runs with and without the denoiser, each row
  taken from the run the gate would pick (levels by `snr_est.py`): broken 6.75% against 6.80%.
- At 58.3 rows/s (34.3 s): denoise 11.2 s, speaker encoder 1.7, codec encode 1.4, admission 2.4,
  frames 7.6, decode 6.8; peak 13.0 GiB allocated. At start-up the denoiser compiles its 32 lengths
  in ~80 s from Inductor's cache and the codec in ~45 s.

## In production

`training/tranche/voice.py` (env `vieneu`, torch 2.15 + cu134) runs the production `Voicer` of the
table above, streamed:

- The loader's processes trim, cut and resample each shard's references and phonemize its lines a
  shard ahead: 217 rows/s with 4 processes. A shard's references are ~17 GB (8 s at 44.1 kHz in
  float32 a row), so the next shard starts loading only once the voicer starts on the one before:
  two at most (holding three, the first production run filled the box's 127 GB and was killed,
  2026-11-26).
- Enrolment is streamed: the voicer takes a shard's rows 256 at a time (`ENROL_PIECE`), shortest
  references first, enrolling each piece as it takes it and dropping its references, so the frames
  and the checks keep running between pieces. Enrolling a whole shard at once held the frames for
  128–156 s a shard (~30% of its time), with the ASR idle. A reference under one 0.25 s bucket is
  padded with silence to one, so the 32 compiled lengths cover them all.
- The SDK's own onnxruntime denoiser and speaker encoder, built by its constructor, are dropped
  (each holds a thread per core); PyTorch runs with expandable segments, as the bench did.
- Rows enter as slots free, from a queue of 512 chunks whose prompt embeddings a thread builds ahead.
  A chunk made again (too short, or suspected of babble) goes to the queue's front. Chunks are
  decoded once 512 wait, sorted by length into batches of 64, or all that wait once nothing generates.
- The checks: 32 processes, each hearing 8 rows at once on threads (`CHECK_ROWS`), so 256 requests
  are in flight, the server's `--max-num-seqs`. The server spends ~92% of its time in decode steps
  (0.67 s a request against 0.06 s of prefill; ~53 transcript tokens a row), each mostly a read of
  its weights, so a bigger batch costs it little more a step. Runaway transcripts (to the 512-token
  cap) are 0.6% of its decode tokens: no lower cap.

Main's code, measured on box 62513802 (2026-11-26):

- Alone, 2,000 YODAS rows of shard 0004 (the smoke test, enrolment in): 30.7 rows/s, 499× real
  time; pace off 2× on 1.1%. The denoiser's first compile took 681 s, 66–77 s from Inductor's
  cache after; VieNeu loads and captures its frame in 10 s.
- In production, beside the ASR server (0.25 of the GPU) under MPS: a shard every ~469 s between
  enrolments, ~26 rows/s, ~440× real time (16.8 s of English a row, RTF ~0.0023): 3.1–3.3×
  VoxCPM2's production RTF (0.0070–0.0075, [tranche-yodas](tranche-yodas.md#voice)). Still broken
  after the check: 7.6% of shard 0004 (9,839 rows), 7.2% of 0005 (12,299), 1.00 tries a row.
- One process over shards 0005–0021, loading and enrolment in: 197,947 rows, 919.9 h of English
  in 2.33 h, 23.7 rows/s, RTF 0.00253 (~395× real time, 2.8–3.0× VoxCPM2's); still broken 6.1–8.7%
  a shard.

End to end (2026-11-28, box 62513802): `voice.run` on YODAS shards 0000–0001 (21,930 rows, ~100 h of
English) beside the ASR server under MPS, nothing pushed (`prod_e2e.py` in the records), timed from
the voicer's start-up (compiles out) to the last shard written:

| Code | Rows/s | Real time | Still broken (0000, 0001) |
|---|---:|---:|---|
| main | 19.9 | 328× | 6.7%, 8.4% |
| this round's voicer, 32 requests in flight | 32.2 | 532× | 7.2%, 8.7% |
| and 256 in flight (`CHECK_ROWS`) | 36.6 | 602× | 7.3%, 8.5% |

- On main the voicer stops 128–151 s a shard to enrol it whole. With 32 requests in flight the
  checks fell ~5,000 rows behind the voicer (~40 rows/s); with 256 they keep up.
- At 256 the run is the GPU's (~97% busy, ~440 W): the voicer needs ~17 ms of it a row (58.3
  rows/s alone) and the ASR ~12.5 ms (below), ~34 rows/s back to back; MPS overlaps a little more.
  The voicer's main process peaked at ~40 GB (the rows waiting for checks, two shards' references).
- Main's denoiser compile on torch 2.8 took 727 s again (Inductor's cache is per torch version).
- The ASR alone on those rows (their Opus at 16 kHz, 256 in flight, `asr_ab.py`): 80.0 rows/s in
  bf16, so ~12.5 ms of the GPU a row against the voicer's ~17. With `--quantization fp8` (vLLM's
  online FP8; weights 3.87 → 2.55 GiB) 111.2 rows/s, 1.39×: the broken flag agrees on 98.7% of
  rows, broken 7.61% against 7.66%, kept WER median 0.036 against 0.037.
- FP8 is the server's (`restart.sh asr`, 2026-11-28), after bf16, FP8 and bf16 again on the same rows
  (`asrcheck/` in the records). Per row it disagrees ~5× as often as bf16 with itself; every total
  holds. On the bench's 2,000 voiced rows the broken flag agrees with bf16 on 98.30% (bf16 again:
  99.65%), broken 6.65% against 6.55%, kept WER 0.0501 against 0.0498. On 5,000 rows of YODAS tar
  0030 cut before the gate, with their captions, ingest's gate agrees on 98.02% (bf16 again:
  99.60%), keeps 75.88% against 76.42% (14.64 against 14.71 h), CER against the caption 0.116
  against 0.120, and the rows both keep 0.0500 against 0.0499. Its one systematic difference: it
  tags sung Vietnamese as Chinese (24 rows of 11 videos, all song lyrics, transcribed right; bf16
  2), which the gate drops on language.
- Not done (the plan's B1, B3): vLLM past 0.14 (`qwen-asr` pins it), and checks that verify
  `text_en` as a draft in one forward pass, which the server's HTTP API cannot take (it builds the
  prompt itself); an in-process vLLM would.

## Finding

Optimized, VieNeu ran 35.5 rows/s (583× real time) against VoxCPM2's 11.3–11.45 (~168×): 3.1×,
3.0× on rows that do not break. By the user's rule VieNeu is the scale-up's TTS (2026-11-26). It
keeps its SDK's quality: 7.7% broken against VoxCPM2's 3.8–4.5%, voice match 0.60 against 0.69,
UTMOS 3.18 against 2.7. `voice.py` runs VieNeu from YODAS shard 0004 on
([In production](#in-production)). Its voicer now runs 58.3 rows/s alone (962× real time) and the
whole job 36.6 rows/s beside its ASR checks (602×, 2026-11-28): 20k h of YODAS rows in ~33
5090-hours, against ~51 at production's 395× before. The ASR's checks now cost the GPU ~12.5 ms a
row against the voicer's ~17. The user heard five rows of it against main's (each with its
Vietnamese source, one whose reference skipped the denoiser; `listen/` in the records): "my listen
test is good" (2026-11-28).

The optimization plan's ~2,000× end to end (its Phase 3, ~122 rows/s: ~8.2 ms of the GPU a row) was a projection
from levers not yet measured. Measured: the voicer needs ~17 ms a row and the ASR's check alone ~12.5 (FP8 ~9),
over that whole budget. Its stage targets missed (at 58.3 rows/s: enrolment 12.9 s against ~8, MOSS 8.2 against
~6, frames 10.0 against ~5), its biggest levers did not hold (TensorRT for the denoiser: wrong output, no faster;
5 s references: −0.02 voice match; vLLM past 0.14 and draft verification: not possible on `qwen-asr`'s server;
FP8 or NVFP4 frames and denoiser: not tried), and it started from 34.1 rows/s alone and 23.7 in production where
main measured 29.6 and 19.9. The round gave 1.97× alone and 1.84× end to end against its ~4.5×.
