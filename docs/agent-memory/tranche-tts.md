# Tranche English TTS

Read when touching `training/tranche/bakeoff.py`, `voice.py` or `envs.sh`, or choosing, tuning or running the
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
- The run of 2026-11-05 (rerun on Japan 59219222, same seeds): 290 rows, 8 workers per model.
  Whisper's normalizer deletes text in brackets, which scored correct readings of a parenthetical
  as broken; `score` now unbrackets first, and leaves out the 9 lines most systems break (Latin,
  rare names like "Cloak N Dagga", one mis-encoded line). 211 of the other 281 rows are one
  speaker. Broken share and voice match (Mimi, 0 = another speaker, 1 = the speaker's own other
  clip; ceiling 0.741, floor 0.268):

  | Mode | Broken | Voice match | RTF |
  |---|---:|---:|---:|
  | CosyVoice3 cross-lingual | 0.0% | 0.11 | 0.19 |
  | Qwen3-TTS x-vector | 1.4% | 0.21 | 0.16 |
  | VoxCPM2 reference only | 3.6% | 0.29 | 0.08 |
  | Qwen3-TTS with transcript | 5.0% | 0.31 | 0.16 |
  | VoxCPM2 with transcript (`ult`) | 8.2% | 0.34 | 0.08 |
  | dots.tts with transcript | 8.9% | 0.40 | 0.32 |
  | dots.tts x-vector | 8.2% | 0.43 | 0.20 |

  The broken rows of the cloning modes are sampling glitches a retry fixes: dots.tts babbling
  "Nhi, Nhi…", silence or "Subtitles by the Amara.org community", VoxCPM2 garbling names.
- Blind listening (user, 2026-11-05): 20 one-speaker rows none of the three broke (7
  PhoAudiobook, 10 viVoice, 3 GigaSpeech 2), the Vietnamese clip against Mimi-round-tripped
  VoxCPM2 `ult`, dots.tts x-vector and Qwen3-TTS x-vector in shuffled order. Closest voice:
  VoxCPM2 11, Qwen3-TTS 7, dots.tts 2; viVoice went 8–1–1 to VoxCPM2, PhoAudiobook 5–2 to
  Qwen3-TTS. dots.tts was marked unusable twice, the others never. The ear and the speaker
  embedding disagree: the highest voice match lost most often, so the embedding score ranks
  modes but does not decide. **VoxCPM2 with transcript is the TTS**: the ear's pick and the
  fastest model, served batched by vLLM-Omni.
- Before the full run, the chosen model is tuned for throughput on the box, with quality checked
  unchanged against the bake-off's outputs (user, 2026-11-05). `training/tranche/voice.py` runs
  nano-vllm-voxcpm 2.0.5's engine in its own process (env `nanovox`): LM and diffusion steps as
  CUDA graphs, up to 512 rows at once. Measured on the bake-off's 290 rows, repeated (H100,
  2026-11-05; RTF is wall time over audio voiced):

  | Setup | RTF |
  |---|---:|
  | voxcpm 2.0.3, 8 processes under MPS (the bake-off) | 0.084 |
  | nano-vllm's server, 64 / 128 / 512 requests at once | 0.0080 / 0.0069 / 0.0065 |
  | its engine without per-step audio decode, rows decoded in fp32 batches | 0.0043 |
  | the same, VAE decoder in fp16 under torch.compile, fixed batch shapes | 0.0032 |

  - Profiling (a stack sampler in the process; ptrace is blocked in the container) put ~70% of
    the server's loop in its streaming VAE decoder, which loops over every stream in Python in
    each causal conv layer. `voice.py` decodes each row once it ends, as voxcpm's non-streaming
    path does. The VAE is causal, so right-padding a batch changes no row.
  - The fp16 decoder is 57 dB SNR from fp32 and 3.4x faster compiled. cuDNN spends ~0.9 s
    planning each new conv shape, so batch widths round up to 64 frames, each width has one
    batch size, and every shape up to 1,024 frames (41 s) is warmed at start (~30 s).
  - Gained nothing: a decode thread on its own CUDA stream (the engine keeps the GPU busy).
    Left: fewer diffusion steps or FP8, which change the output.
  - Quality on the bake-off's 281 rows (its 9 hard lines out), Mimi round trip: broken 8.2%,
    11.0%, 11.0% over three batched passes against voxcpm's 8.2%, and voice match 0.339–0.350
    against 0.338. Only 6 rows broke in all three passes.
- `voice.py run` voices the tranche (see its docstring). Each row is prompted with its own clip
  and Vietnamese transcript, as the paper conditions on the source speaker. The bake-off cloned
  from another clip of the speaker only to score similarity fairly. Every output is heard by
  Qwen3-ASR-1.7B (`qwen-asr-serve` from env `asr`, `to_language=en` to force English) and voiced
  again while broken (WER over 20% under Whisper's normalizer, or pace off by 2x), up to 3
  tries. Two viVoice shards: 1.05 tries a row, 1.5% still broken. Those rows are mostly the ASR
  mishearing Vietnamese names and numbers; `en_broken` marks them for the cache build.
  PhoAudiobook: 1.02–1.06 tries a row, ~1–1.6% still broken. Each check asks for at most 512
  transcript tokens (`max_completion_tokens`, as `checks.py`): uncapped, ASR decoding is greedy, so
  an output it loops on ran to the model's 65k context past any timeout, every try (one in the
  first ~8,000). Capped, it comes back long and scores broken; a timeout or HTTP error then means
  the server is in trouble and stops the run.
- The full run shares the GPU with the ASR server (`--gpu-memory-utilization 0.2`, ~25 GB with
  its graphs; the TTS holds ~55 GB at `--gpu-memory 0.45`, ~1.5 GB left). It runs at RTF ~0.007:
  a 1,299-row PhoAudiobook shard every ~2 min, ~13 h for the tranche (started 2026-11-05).
  Decoding and checking run in spawned processes: as threads, their Python held the GIL and
  starved the engine loop (GPU at 2%). One ASR API process featurizes ~11 requests/s on the
  CPU, so the server runs 8.
- The box's container allows 6,656 threads. A pool sized by `nproc` (224) exceeds it, so threads
  follow `cpu_budget()`. ffmpeg defaults to one decoder thread per host CPU, so PyAV codecs get
  `thread_count = 1`. PyAV's `AudioResampler` builds a filter graph holding a thread per host CPU
  while it lives (4,075 threads from 20 decoding threads), so `voice.py` resamples with scipy. Its
  Opus encoder builds one too, so a semaphore shared by the checking processes lets at most 4
  encode at once.
- A crashed run drops its queued checks rather than running them, so its GPU memory frees within
  a few minutes. Stop a run by its PID file (`/workspace/logs/voice-run.pid`), never `pkill -f`:
  its spawned workers look like vLLM's own processes.
- `training/tranche/envs.sh` builds those envs.
- `synth` checks for the MPS daemon with `nvidia-cuda-mps-control` itself: on driver 615 its
  socket is seqpacket, and a stream connect fails with EPROTOTYPE while the daemon runs.
- The MPS daemon outlives the bake-off, and every later CUDA job on the box runs as its client:
  stop it with `echo quit | nvidia-cuda-mps-control` before the trainer.
