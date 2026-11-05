# Tranche English TTS

Read when touching `training/tranche/bakeoff.py` or `envs.sh`, or choosing, tuning or running the
English TTS that voices the real-Vietnamese tranche's targets in each speaker's voice.
Back to [CONTEXT.md](../../CONTEXT.md) · part of [real-vi-tranche](real-vi-tranche.md)

Kyutai's TTS 1.6B, the paper's, is out: its voice conditioning is closed and only precomputed
embeddings ship ([card](https://huggingface.co/kyutai/tts-1.6b-en_fr)). The finalists, with
Apache code and weights, at least 24 kHz, cloning from any reference clip:

| Model | Published English numbers | Notes |
|---|---|---|
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (2B) | Seed test-en WER 1.84, similarity 75.3; 48 kHz | Vietnamese is one of its 30 languages, so a Vietnamese reference is in distribution; clones from the reference alone; batched serving through vLLM-Omni. Output varies between runs (its authors suggest 1–3 tries) |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | WER 1.24, similarity 0.775; 24 kHz | No Vietnamese; its best mode needs the reference's transcript |
| [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | WER 2.24, similarity 71.8; 24 kHz | A cross-lingual mode; the smallest and fastest |
| [dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar) (Apache-2.0; post-trained from dots.tts-base) | WER 1.30, similarity 77.1 | New; no serving stack |
| [LongCat-AudioDiT-3.5B](https://github.com/meituan-longcat/LongCat-AudioDiT) (MIT) | WER 1.50, similarity 78.6 | New; English and Chinese only |

- Ruled out:
  - non-commercial weights (F5-TTS, MaskGCT, Llasa, Spark-TTS, XTTS-v2, OpenAudio S1-mini, Fish S2);
  - licences that bar training other models on the output (IndexTTS-2 §3.4(c), Higgs Audio v2);
  - VibeVoice (research only, an audible disclaimer in every output);
  - MegaTTS 3 (its cloning encoder is unreleased);
  - OpenVoice v2 (22.05 kHz).
- Kyutai's `tts-0.75b-en-public` (CC BY 4.0, clones by an audio prefix, English only) is a fourth
  candidate if time allows: it is the closest to the paper's setup.
- From the cards and repos (2026-11-04): VoxCPM2 lists Vietnamese among its languages and
  dots.tts benchmarks it; Qwen3-TTS, CosyVoice3 and LongCat do not. LongCat is out: it needs the
  reference's transcript and was trained on Chinese and English only. No licence of the other
  four restricts training on their outputs. Only CosyVoice3 (Triton/TRT-LLM) and VoxCPM2 (Nano-vLLM
  or vLLM-Omni) document batched serving; their Python APIs, like dots.tts's, run one request at
  a time, and Qwen3-TTS takes lists.
- None publishes Vietnamese→English speaker similarity, so a bake-off chooses, on the H100
  (`training/tranche/bakeoff.py`, each model in its own uv env under `/workspace/envs`):
  1. real Vietnamese clips, one pair per speaker: PhoAudiobook readers, viVoice channels and
     GigaSpeech 2 recordings, each clip with another 5–16 s clip of the same speaker as its
     reference, and an `en-vi-val` English line to speak;
  2. each model's cloning modes at fixed seeds (VoxCPM2 reference-only and with the reference's
     transcript; dots.tts and Qwen3-TTS x-vector and with the transcript; CosyVoice3
     cross-lingual), every output round-tripped through Mimi (16 codebooks), as training sees it.
     Each model runs its best release: `dots.tts-soar` (post-trained; `-base` is its pretraining
     baseline) and CosyVoice3's RL-tuned LLM (`llm.rl.pt`, ranked above `llm.pt` on its card);
  3. scores:
     - speaker similarity (WeSpeaker ResNet293-LM: the seed-tts-eval WavLM-Large checkpoint is
       only on Google Drive) between the row's own Vietnamese clip and the English output,
       anchored by the clip against its reference (ceiling; rows under 0.4 are not one speaker)
       and against another speaker's clip (floor);
     - Whisper-large-v3 WER on the English, and the share of broken clips: WER over 20%, or a
       duration per character off the median by 2× either way;
     - speed: wall time per second of audio, one model at a time, its rows shared out to 8
       processes on the GPU through CUDA MPS, timed once every process has loaded. The bake-off
       calls each model one request at a time, which leaves the GPU two-thirds idle; MPS lets the
       processes' kernels overlap, where plain processes take turns. Qwen3-TTS on 64 rows: RTF
       0.89 with one process, 0.40 with 4 without MPS, 0.29 with 4 and 0.17 with 8 under MPS
       (2026-11-05). dots.tts gained far less (RTF 0.81 alone, 0.52–0.64 with 8), with its 8
       processes given 16 torch threads each on a box whose container quota is 15.36 CPUs
       (`nproc` says 128): workers now take the quota's share (`training.paths.cpu_budget`).
       This speed compares the models; production cost is the winner's batched runtime
       (vLLM-Omni serves Qwen3-TTS and VoxCPM2 with CUDA graphs, CosyVoice3 has TensorRT-LLM),
       measured after. No Qwen3-TTS is newer than the January 2026 release;
  4. production verifies every output with Whisper and voices a broken one again, so a mode's
     broken share is its retry cost, not a bar: take the highest voice match whose kept rows'
     WER is near the best, then listen blind to 20 pairs; the winner must reach ~50 H100-hours
     per 1,000 h (RTF 0.05, retries included) on its batched runtime, or the runner-up is
     measured the same way. (The first rule, broken ≤ 2% and WER ≤ 3%, passed only CosyVoice3,
     which barely clones, and Qwen3-TTS x-vector, at half the others' voice match.)
- The run of 2026-11-05: 290 rows, 8 workers per model. Whisper's normalizer deletes text in
  brackets, which scored correct readings of a parenthetical as broken; `score` now unbrackets
  first, and leaves out the 7 lines most systems break (Latin, rare names like "Cloak N Dagga",
  one mis-encoded line). Broken share and voice match (Mimi, 0 = another speaker, 1 = the
  speaker's own other clip; ceiling 0.742, floor 0.266):

  | Mode | Broken | Voice match | RTF |
  |---|---:|---:|---:|
  | CosyVoice3 cross-lingual | 0.0% | 0.12 | 0.18 |
  | Qwen3-TTS x-vector | 1.8% | 0.22 | 0.13 |
  | VoxCPM2 reference only | 3.9% | 0.29 | 0.08 |
  | Qwen3-TTS with transcript | 4.9% | 0.32 | 0.14 |
  | VoxCPM2 with transcript (`ult`) | 8.5% | 0.35 | 0.08 |
  | dots.tts with transcript | 9.5% | 0.40 | 0.31 |
  | dots.tts x-vector | 8.8% | 0.43 | 0.20 |

  The broken rows of the cloning modes are sampling glitches a retry fixes: dots.tts babbling
  "Nhi, Nhi…", silence or "Subtitles by the Amara.org community", VoxCPM2 garbling names.
  VoxCPM2 with transcript is the pick, pending the blind listening: the fastest model one
  request at a time, served batched by vLLM-Omni. dots.tts keeps the voice a little better
  but has no batched runtime and runs 2.5–4× slower.
- Before the full tranche run, the chosen TTS model is optimized for throughput on the box and
  profiled on a slice (user, 2026-11-05): batched serving, CUDA graphs, batch size and
  concurrency swept to the GPU's limit, quality checked unchanged against the bake-off's
  outputs. The full run starts only from the tuned setup.
- `training/tranche/envs.sh` builds those envs.
- `synth` checks for the MPS daemon with `nvidia-cuda-mps-control` itself: on driver 615 its
  socket is seqpacket, and a stream connect fails with EPROTOTYPE while the daemon runs.
- The MPS daemon outlives the bake-off, and every later CUDA job on the box runs as its client:
  stop it with `echo quit | nvidia-cuda-mps-control` before the trainer.
