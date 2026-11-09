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
  4. production verifies every output with an ASR (Qwen3-ASR) and voices a broken one again, so a mode's
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
  fastest model, run batched on nano-vllm-voxcpm's engine.
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
    batch size, and every decode shape up to 1,024 frames (41 s) and every prompt shape up to
    `ROW_S`'s 22.4 s is warmed at start (below: "The VAE").
  - Gained nothing: a decode thread on its own CUDA stream (the engine keeps the GPU busy).
    Left: fewer diffusion steps or FP8, which change the output.
  - Quality on the bake-off's 281 rows (its 9 hard lines out), Mimi round trip: broken 8.2%,
    11.0%, 11.0% over three batched passes against voxcpm's 8.2%, and voice match 0.339–0.350
    against 0.338. Only 6 rows broke in all three passes.
- `voice.py run` voices the tranche (see its docstring), except the shards held out for
  evaluation (`ingest.SOURCES[...].heldout`, [giga-heldout](sft-evaluation.md#evaluation-sets)).
  Each row is prompted with its own clip
  and Vietnamese transcript, as the paper conditions on the source speaker. The bake-off cloned
  from another clip of the speaker only to score similarity fairly. Every output is heard by
  Qwen3-ASR-1.7B (`qwen-asr-serve` from env `asr`, `to_language=en` to force English) and voiced
  again while broken (WER over 20% under Whisper's normalizer, or pace off by 2x), up to 3
  tries. Two viVoice shards: 1.05 tries a row, 1.5% still broken. Those rows are mostly the ASR
  mishearing Vietnamese names and numbers; `en_broken` marks them for the cache build.
  PhoAudiobook: 1.01–1.06 tries a row, 0.2–2.2% still broken per shard. GigaSpeech 2: 1.35–1.40
  tries a row, 10.2–11.9% still broken per shard. On its shard 0, 2,007 of the 2,085 broken rows fail
  on WER, 78 on pace; clips under 2 s break most (23%, against 7% from 3 to 8 s), where one
  Vietnamese name the ASR spells otherwise ("Binh Dinh" heard as "Benin") is over 20% of the words.
  Of 12 sampled, 5 look like that ASR noise and 7 are real faults: a word repeated or dropped, a
  number changed ("177" spoken as "100"), a Vietnamese clause. Ignoring spaces and punctuation would
  pass only 23 of the 2,085, so the rule stays and the cache drops them. The full run voiced 2,138.1
  h in 13.43 h (done 2026-11-06 12:17). Prompted with the
  Vietnamese transcript, VoxCPM2 sometimes speaks a Vietnamese clause before the English; under
  ~20% of the words it passes the WER check (~0.05% of rows, e.g. pho/00003/00588), so the cache
  build also drops rows whose `en_asr` has Vietnamese letters absent from `text_en`. The TTS
  overshoots full scale on 3% of PhoAudiobook rows and 15% of viVoice rows (at most 0.23% of
  samples, hard-clipped by the Opus encode). Opus decodes at 48 kHz whatever the 24 kHz input,
  so readers resample. Each check asks for at most 512
  transcript tokens (`max_completion_tokens`, as `checks.py`): uncapped, ASR decoding is greedy, so
  an output it loops on ran to the model's 65k context past any timeout, every try (one in the
  first ~8,000). Capped, it comes back long and scores broken. The server still stalls past a 60 s
  try now and then and recovers within minutes; with 3 tries that ended a run three times in two
  days (2026-11-06/07), each losing the shards in flight, so a check now tries 10 times (`ASR_TRIES`)
  before a timeout stops the run; an HTTP error stops it at once.
- The full run shares the GPU with the ASR server (`--gpu-memory-utilization 0.2`, ~25 GB with
  its graphs; the TTS holds ~55 GB at `--gpu-memory 0.45`, ~1.5 GB left). It runs at RTF ~0.007:
  a 1,299-row PhoAudiobook shard every ~2 min, ~13 h for the tranche (started 2026-11-05).
- The scale-up (GigaSpeech 2 from archive 13, 2026-11-06) runs one try a row (`MAX_TRIES` 1): time,
  not source rows, bounds it. On six of the first tranche's GigaSpeech 2 shards (131,669 rows,
  158 h), counting each row's tries:

  | Tries | Rows kept | Kept Vietnamese h per h of English voiced |
  |---:|---:|---:|
  | 1 | 77.1% | 0.660 |
  | 2 | 85.8% | 0.607 |
  | 3 | 89.0% | 0.571 |

  Clips under 2 s are 13.4% of the rows and 4.8% of the hours, 24.8% broken at 1.72 tries a row.
  A first test of fewer flow-matching steps (2,000 rows of shard 0, the ASR server beside, one
  try; mean WER over all rows, broken ones included): 10 steps RTF 0.00524, 22.2% broken, mean
  WER 0.188; 6 steps 0.00448, 24.0%, 0.317; 4 steps 0.00384, 33.9%, 0.373. It kept 10 steps; by
  cost per kept row (below) even it favoured fewer (6 steps 0.875, 4 steps 0.863 of 10's).
  Decoding and checking run in spawned processes: as threads, their Python held the GIL and
  starved the engine loop (GPU at 2%). One ASR API process featurizes ~11 requests/s on the
  CPU, so the server runs 8 (a third of the CPU quota: `restart.sh asr`).
- Quality against speed (2026-11-08; the user: changing the audio is fine if quality stays good).
  The diffusion head is ~92% of a decode step: 141 ms for 512 rows at 10 steps (9 estimator calls
  at double batch for guidance; the first step is a zero init), 83 ms at 6. The same 2,000 kept rows
  of archive 147, one pass each, the production check, voice match (WeSpeaker, raw outputs, against
  the row's Vietnamese clip; paired difference ± standard error); cost per kept row is RTF over the
  share not broken:

  | Variant | RTF (generation) | Broken | Kept WER median | Voice match | Cost per kept row |
  |---|---:|---:|---:|---:|---:|
  | 10 steps | 0.00407 | 16.2% | 0.035 | 0.618 | 1 |
  | 8 steps | 0.00368 | 17.2% | 0.038 | −0.001 ± 0.001 | 0.915 |
  | 6 steps | 0.00331 | 19.7% | 0.043 | −0.008 ± 0.001 | 0.849 |
  | No reference audio (prompt only) | 0.00389 | 15.8% | 0.035 | −0.020 ± 0.001 | 0.951 |
  | Guidance on the first 4 calls only | 0.00351 | 30.3% | 0.056 | −0.053 ± 0.002 | 1.04 |
  | Guidance on the first 2 calls only | 0.00321 | 58.1% | 0.077 | −0.113 ± 0.002 | 1.58 |

  `TIMESTEPS` is 8 from 2026-11-08 ~00:40 UTC+7 (the shards voiced before then ran 10, and the
  figures above them were measured at 10): within noise of 10 (broken +1.0 point, kept WER median
  +0.003, voice match unchanged). 6 steps costs ~3.5 points of broken rows and 1% of voice match;
  dropping the reference costs 3% of voice match; a guidance interval breaks rows.
- Measured, not shipped (2026-11-07/08): Qwen3-ASR-0.6B as the check served 122 requests/s against
  the 1.7B's 138 (the server is bound by its API processes' CPU featurizing, not the GPU) and agreed
  with the 1.7B on broken 89% of rows (the 1.7B with itself on the Opus round trip: 95%), flagging 3
  points more. The VAE's residual units fused for torch.compile (the depthwise conv as shifted
  multiply-adds, exact to 1e-6) gave 5% in a 3,000-row profile, mostly on partial batches (below).
  A full batch takes the compiled fp16 decoder 173 ms in the engine as alone (179 ms with the
  copies in and out; eager 340 ms); cuDNN autotune (`cudnn.benchmark`) changed neither the decoder
  nor the encoder, nor did max-autotune the static decoder (131 ms at width 192).
- Skipped rows (2026-11-07): the cache build drops a row on its Vietnamese side alone (language,
  CER over 0.15, a nghìn/triệu/tỷ the transcript lacks: `filters.vi_gate` on asr_vi) whatever its
  English, 5.5% of GigaSpeech 2's rows and 6.6–6.9% of its hours (archives 147, 148). `voice.py`
  no longer voices them: `en_tries` 0, the other English columns null. So a shard waits for its
  asr_vi as for its text_en: run `filters.py` beside voice, ahead of it (a run logs how many
  translated shards wait for their asr_vi). Its per-shard "still broken" now counts voiced rows
  only, so it reads lower than the ~17.7% of the shards voiced before (many skipped rows broke).
- Speed on one H100 (2026-11-07, 5,000 rows of GigaSpeech 2's archive 147, 7.8 h voiced, the GPU
  to itself: no ASR server, no checks): RTF 0.0037. Of the engine's time, decode steps 45% (141 ms
  for 512 rows), the VAE decode 30%, prefill 19%, Python 5%. A short profile overstates the VAE:
  its last flush decodes each width's partial batch padded to a full one; a long run decodes full
  batches (a rare width waits up to `MAX_WAIT_S`).
  Submitting only once 128 slots are free (prefill in larger steps) changed nothing (RTF 0.0037);
  1,024 sequences at 0.6 of the GPU gained 5% (decode 260 ms for 910 rows: compute-bound). In
  production, beside the ASR server and filters, GigaSpeech 2 ran at RTF 0.0058–0.0063. Filters
  beside voice (the same ASR server and CPU quota) slows it ~20%: a run's first archive (startup
  included) ran RTF 0.0062 without filters and 0.0077–0.0081 beside them at 10 steps, 0.0070–0.0073
  beside them at 8 (archives 145–151). Archive to archive (8 steps, the fp16 VAE, startup
  excluded): RTF 0.0049–0.0057 (mean 0.0052) beside filters (archives 157–173), 0.0044 once filters
  had finished (2026-11-08 06:38 UTC+7; archives 177–190): ~8.6 min an archive of ~21k rows.
  The feed keeps up: 10 min of `nvidia-smi` at 100 ms (2026-11-08 14:18–14:28 UTC+7, archives
  230–232 beside the ASR server) read 93.4% busy on average, 94–95% while generating, and 0% for
  1.2% of samples. The dips are the prompt encode at each archive's start (38–51 s with the engine
  paused, 81% busy with a ~5 s stall; it is GPU work, 1.92 ms a prompt alone, so encoding ahead
  in a thread would recover only its gaps), the shard write in the main thread (~2 s at 0%), and
  sub-second dips to 50–80% between engine steps. At most ~1–3% is left there: not worth changing.
- The VAE (2026-11-08): both halves run compiled in fp16 with static shapes, a graph per batch
  shape (`shape()` gives each width one), all warmed at start. Static graphs run faster than one
  dynamic graph: the decoder 132 ms a full batch against 174 (24% less, output identical), the
  prompt encode 1.92 ms a prompt against 2.98 dynamic and 4.13 for the fp32 eager encoder voxcpm
  runs (3,000 prompts of archive 147, the GPU alone). The warm costs ~3 min a run (decoder 140 s for
  16 widths, encoder ~50 s for 9; dynamic took ~45 s), repaid within ~3 archives. Its inputs are made
  in inference mode as the runs' are: Dynamo's guards tell inference tensors apart, so a warm on
  other tensors would compile every width again mid-run. Past `recompile_limit` (64 a compiled
  module) new shapes run eager, the cached ones still compiled. The fp16 encoder's latents are
  51 dB from fp32's (the engine's own bf16 cast of them: 56 dB); on the quality test's 2,000 rows
  at 8 steps: broken 17.2% (fp32 encoder 17.2%), kept WER mean 0.054 (0.053), voice match
  −0.0007 ± 0.0014. The encode had taken ~85 s an archive of ~19k prompts, the engine's steps waiting meanwhile;
  a run logs it per shard. In production from 2026-11-08 03:32 UTC+7 (beside filters):
  archive 156, a run's first, RTF 0.0067 against 0.0070–0.0078 for the six first archives before it
  on the old VAE; the encode 52–58 s an archive (~2.8 ms a prompt beside the ASR server), the warm
  61 s (Inductor's on-disk cache held the graphs from the benchmarks).
- CPU: process pools take shares of the container's quota (`cpu_budget()`, cgroup `cpu.max`;
  26.88 CPUs on the Japan box, `nproc` 224): voice's clip decoders a quarter, filters' a sixth.
  The 32 checkers stay fixed: they are the ASR requests in flight, not CPU work.
- The box's container allows 6,656 threads. A pool sized by `nproc` (224) exceeds it, so threads
  follow `cpu_budget()`. ffmpeg defaults to one decoder thread per host CPU, so PyAV codecs get
  `thread_count = 1`. PyAV's `AudioResampler` builds a filter graph holding a thread per host CPU
  while it lives (4,075 threads from 20 decoding threads), so `voice.py` resamples with scipy. Its
  Opus encoder builds one too, so a semaphore shared by the checking processes lets at most 4
  encode at once.
- A crashed run drops its queued checks rather than running them, so its GPU memory frees within
  a few minutes, except behind a check still trying a stalled server (up to `ASR_TRIES` × ~70 s):
  a server that hangs is surfaced ~20 min after it does, one that refuses connections at once. Stop a run by its PID file (`/workspace/logs/voice-run.pid`), never `pkill -f`:
  its spawned workers look like vLLM's own processes.
- `training/tranche/envs.sh` builds those envs.
- `synth` checks for the MPS daemon with `nvidia-cuda-mps-control` itself: on driver 615 its
  socket is seqpacket, and a stream connect fails with EPROTOTYPE while the daemon runs.
- The MPS daemon outlives the bake-off, and every later CUDA job on the box runs as its client:
  stop it with `echo quit | nvidia-cuda-mps-control` before the trainer.
