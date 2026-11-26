# The TTS for YODAS's scale-up

Read when choosing or speeding up the English TTS for YODAS's ~20k h, or running ZONOS3-Mini or
VieNeu-TTS v3 Turbo on the RTX 5090.
Back to [CONTEXT.md](../../CONTEXT.md) · part of [tranche-tts](tranche-tts.md) ·
[Candidates](#candidates) · [Method](#method) · [Results](#results) · [ZONOS3 settings](#zonos3-settings) ·
[Running them on the 5090](#running-them-on-the-5090) · [Finding](#finding)

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
  `/Volumes/data/hibiki-zero-evals/box-62513802/ttsbench-2026-11-26/`.

## Results

Paired differences are against VoxCPM2 on the rows neither broke, ± standard error:

| System | Broken | Kept WER median | Voice match | UTMOS | Speed on the 5090 |
|---|---:|---:|---:|---:|---|
| VoxCPM2 (production outputs) | 3.9% | 0.025 | 0.687 | 2.59 | 182× bench, ~140× beside the ASR server |
| ZONOS3, speaker embedding | 31.2% | 0.015 | 0.594 (−0.087 ± 0.003) | 3.39 | LM 315× at 256 rows; 184× with embeddings and decode |
| VieNeu v3 Turbo | 7.1% | 0.035 | 0.604 (−0.081 ± 0.002) | 3.18 | generation 326× at 128 rows a batch; 12× with its CPU enrolment |

- Both new models speak cleaner audio (higher UTMOS) and take ~10–20% longer to say the same line
  (9.1–9.8 h against VoxCPM2's 8.2 h).
- VieNeu's enrolment (denoise, speaker encoder, codec encode) runs on the CPU in ONNX at 1.3 s a
  clip. Its generation read 92% busy at 270 W, ZONOS3's LM 73% at 246 W, against the card's 575 W.

## ZONOS3 settings

Every 7th row (286; VoxCPM2 there: 2.8% broken, voice match 0.686):

| Setting | Broken | Voice match against VoxCPM2 |
|---|---:|---:|
| speaker embedding | 35.0% | −0.079 ± 0.007 |
| embedding moved by −0.25 × the Vietnamese direction | 16.4% | −0.102 ± 0.007 |
| −0.5 × | 4.2% | −0.142 ± 0.007 |
| −0.5 ×, speaker guidance 1.5 (twice the compute) | 8.0% | −0.088 ± 0.007 |
| −1.0 × | 2.8% | −0.232 ± 0.007 |
| temperature 0.5 | 33.6% | −0.087 ± 0.007 |
| raw reference prefix | 4.5% | −0.174 ± 0.006 |

- ZONOS3 breaks by leaving English mid-line ("Adi tine jam province ay la to comments sa trang…"):
  a Vietnamese voice pulls the multilingual model off English. Its shipped speaker-embedding
  direction `assets/steering/speaker_accents_mect60k_x10/vie.npy` (Vietnamese minus English speakers,
  applied as its steering does: subtracted, then rescaled to the embedding's norm) keeps it in
  English, and takes the voice with it; so does the raw reference.

## Running them on the 5090

- ZONOS3's `servers/sgl/scripts/env.sh` takes compute capability ≥ 9 for Hopper: on sm_120 set
  `FLA_TILELANG=0 ZONOS3_USE_TRAINING_SONIC_EXPERTS=0` (the A100 kernels).
- No attention kernel serves its head_dim 32 on sm_120: FlashInfer has no decode for 32 and
  sgl-kernel's FA3 is sm_90 only. The bench swapped the `fa` backend's `_fa_sgl_impl` for PyTorch
  (gather the page-size-1 cache, masked softmax; decode stays in the CUDA graphs) with
  `ZONOS3_PREFILL_CUDA_GRAPH=0` and `max_seq_len_override` 2048.
- Its speaker encoder runs on the CPU unless `ZONOS3_SPEAKER_EMBEDDER_DEVICE=cuda:0` (then 24 ms a
  clip, one at a time); the raw-reference encoder runs a 600M w2v-BERT teacher on 20 s chunks, one
  clip at a time (0.27 s a clip once loaded).
- `TTSLLM.generate` decodes each row alone after the batch. A row that never stops can fail the
  codec's check ("Invalid matched codec tokens"), inside `generate` too: the bench decodes rows
  itself and counts such a row broken (1 of 2,000).
- VieNeu's `repetition_penalty` (1.2) turns off its CUDA graphs; its batches are static, grouped by
  phoneme length.

## Finding

Neither model reaches 1,000×, and neither matches VoxCPM2: VieNeu breaks 3.2 points more rows,
ZONOS3 holds English only by giving up voice match, and both lose ~0.08 voice match at best. Their
generation is ~1.8× VoxCPM2's before their conditioning and decode are batched. Throughput is
bought with boxes instead: 20k h at VoxCPM2's production RTF ~0.007 is ~140 5090-hours (~$72 at
$0.51/h, ~1.2 days on 5 boxes), against ~20 hours at 1,000×. The choice is the user's.
