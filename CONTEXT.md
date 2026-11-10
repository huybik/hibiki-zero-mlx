# Hibiki MLX — project context

Last verified: 2026-11-10

## Purpose

This repo runs Hibiki-Zero simultaneous speech translation:

- MLX inference on Apple Silicon (upstream q4 for FR/ES/PT/DE→EN, or staged BF16 checkpoints);
- full-model Vietnamese→English voice-preserving SFT on one H100, on multi-sentence streams
  joined from the 5–22.4 s clips of the English read-speech corpus `en_speech` (~10.5k hours;
  the ~0.8k-hour PhoMT cache is a later ablation);
- the roadmap toward VI→EN parity with the upstream languages.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero),
[paper](https://arxiv.org/abs/2602.11072).

## Map

```text
training/  (CUDA, Linux H100; prep/ translates, voices, publishes and caches the EN-VI dataset on the Mac)
  en_vi_speech (the en_speech corpus; text_vi from prep/translate_en_speech)
    ─► prep/tts_en_speech (fills vi_audio) ─► prep/publish_en_vi_speech upload ─► HF common-vi-en
    ─► prep/cache_codes (on prep/cache_build, MLX or CUDA; Mac top down, a second machine bottom up)
         ─► <out-dir>/{train,validation}
    ─► prep/publish_caches --follow ─► HF cache repo <out-dir name>/ ─► data/cache/ on the pod
    ─► prep/mimi_cache_cuda (CUDA, the pod: Vietnamese source re-encoded under prep/augment) ─► *-noisy
    ─► h100.sh run → finetune/train.py ─► runs/, HF recovery repo
  HF PhoMT cache ─► data/cache/phomt-grounded-v2-5s ─────────┘ (one more --cache-dir, ablation)
inference/ (Apple Silicon, MLX)
  checkpoints ─► tools/convert_mlx_{q4,bf16}.py ─► weights/… ─► main.py · hibiki_mlx/ · macos/
                                                            └─► swift/ (MLX Swift: CLI, iOS bench app)
  Mimi checkpoint ─► tools/export_mimi_coreml.py (from tools/mimi_step.py) ─► mimi-decoder.mlmodelc ─► swift/
```

- `inference/` is the Apple Silicon runtime: `main.py`, the `hibiki_mlx/` runtime, the
  vendored `moshi-mlx/`, `tools/`, `assets/leon.wav`, the `macos/` test app and `swift/`, the
  native MLX Swift runtime with its iOS benchmark app.
- `training/` holds `h100.sh`, shared `paths.py`, `prep/` (the EN-VI dataset publisher, the
  Vietnamese translation of its English text, the Vietnamese TTS in MLX, the grounded cache build
  every cache builder shares (`cache_build`, MLX or CUDA) and the EN-VI dataset's builder on it,
  the evaluation-set builder, the noisy source re-encode), `finetune/` (the generic SFT trainer, GRPO, evaluation
  and HF recovery sync), `tranche/` (the real-Vietnamese tranche on the box: the English
  TTS bake-off, the ASR check, each model in its own uv env, the ingest of the corpora's
  segments to the tranche repos, their translation, the English TTS, the row filters and its cache builder) and `vi_tokenizer/` (the Vietnamese tokenizer swap, on the box's
  CPUs). Its modules run from the repo root as
  `python -m training.<package>.<module>`; everything except the translation, the TTS, the dataset
  publisher, the cache builder, the evaluation-set builder and the tokenizer swap runs on CUDA only.
- `docs/` holds plans, analyses and reports.
- Gitignored: `weights/` (shared by both sides), `data/` (pairs, caches), `runs/` (training
  runs and eval outputs) and `translations/` (inference outputs).

## Where things stand

- The last run, on PhoMT and FLEURS (`grounded_v2_full_direct_voice_5epoch`), is analysed up
  to step 135k in `docs/analysis/`: teacher-forced loss improves, but free-running translation
  fails. FLEURS and MultiMed-ST now serve only as real-Vietnamese evaluation sets
  ([sft-evaluation](docs/agent-memory/sft-evaluation.md#evaluation-sets)).
- Two stream runs are done on a 700 W H100 SXM: `en_speech_streams_epoch1b` (one epoch of
  `en-vi-grounded-v2/train`) and `en_speech_noisy_epoch2` (one epoch of its noisy copy, from the
  first's final model). FLEURS test chrF went 24.7 → 29.5 ([plan](docs/vi_en_parity_plan.md#order)).
- The third, `tranche_epoch1b` (2026-11-06), trained one epoch of the real-Vietnamese tranche's
  caches from run 2's final model: FLEURS test chrF 29.5 → 38.6, MultiMed-ST 24.9 → 34.6
  ([plan](docs/vi_en_parity_plan.md#order)).
- The fourth, `full_epoch4b` (2026-11-10, the H200 box), one epoch of every tranche cache plus
  150 noisy `en_speech` shards (6,737 h) from round 3's model: MultiMed-ST 34.7 → 37.4,
  `giga-heldout` 51.3 → 61.8, FLEURS test 38.2 → 37.9 (inside its noise), the held-out TTS sets
  −0.9. `full_epoch4e` repeats it with 500 `en_speech` shards (29% of 8,456 h) in round 3's
  4,608-frame batches (2026-11-10, an H100) ([plan](docs/vi_en_parity_plan.md#order)).
- [docs/vi_en_parity_plan.md](docs/vi_en_parity_plan.md) is the active plan (2026-11-01):
  Hibiki-Zero 3B from training to the phone. For the data,
  the `en_speech` corpus is built and all its transcripts are translated to Vietnamese:
  the first tranche and all Common Voice on Claude
  ([en-vi-translation](docs/agent-memory/en-vi-translation.md)),
  2,000 h of Libriheavy on Gemini ([en-vi-gemini](docs/agent-memory/en-vi-gemini.md)) and the
  rest on Muse ([en-vi-muse](docs/agent-memory/en-vi-muse.md)). The corpus now lives, with its
  translations and the already voiced Common Voice rows of `anquachdev/common-vi-en`
  (`cv-anquach`), in the local EN-VI dataset, published as `anquachdev/common-vi-en`, the one
  copy used ([en-vi-speech-dataset](docs/agent-memory/en-vi-speech-dataset.md)). VieNeu-TTS voiced the
  Vietnamese lines in each row's own English voice, filling the dataset's `vi_audio` shard by
  shard, the Mac top down and anquachdev's hardware bottom up, meeting through the Hub; all
  1,842 shards were voiced on 2026-10-27 ([vi-tts](docs/agent-memory/vi-tts.md)).
  `cache_codes` builds the training caches from it in MLX;
  the full build ran on the Mac from 2026-10-27 to 2026-11-02: 1,835 main shards (3.45M rows) and
  7 `cv-anquach` shards, all on the Hub ([sft-data-caches](docs/agent-memory/sft-data-caches.md#builds)). `publish_caches`
  commits each shard to the cache repo as it is built, and anquachdev's machine may build bottom
  up, meeting the Mac through the Hub as for the TTS
  ([sft-data-caches](docs/agent-memory/sft-data-caches.md#publishing)).
- Next (user, 2026-11-04): a learning-rate test, then a third VI→EN epoch only if a higher rate
  won (none did: no third epoch), then the real-Vietnamese tranche (its SFT at 3e-6)
  ([real-vi-tranche](docs/agent-memory/real-vi-tranche.md)) and two-way speech in one model,
  prepared in parallel; GRPO last ([plan](docs/vi_en_parity_plan.md#order)). Then (user,
  2026-11-10): two-way before GRPO, so after round 4e the two-way pilot and SFT, then one GRPO on
  both directions, and the phone student cut from the two-way model. GRPO is built in
  `training/finetune/grpo.py` on the SFT trainer's run directory, loss kernel and validation, and
  is profiled on the box phase by phase before its run ([grpo](docs/agent-memory/grpo.md)).

## Essential invariants

- Profile every job before its full run (a cache build, an encode, a training or GRPO run, an
  evaluation pass) and optimize it until its costliest resource stays busy; only then start the
  real run. Time each stage on a few
  shards, sample the GPU (`nvidia-smi` every 0.5 s) and know the CPU budget (a container's quota
  in `/sys/fs/cgroup/cpu.max`, not `nproc`; `h100.sh cpus` makes new shells follow it); compare variants on the same rows. The user's rule
  since the first noisy encode ran ~3 h with its GPU idle half the time →
  [noisy-source](docs/agent-memory/noisy-source.md#speed),
  [cache-build](docs/agent-memory/cache-build.md#speed-on-the-mac).
  Optimize every job heavily before its full run, not during it: each job's levers differ, so
  profile where its own time goes and judge changes by its throughput per GPU-hour beside the box's
  other jobs; a busy GPU is no proof (voicing read ~94% busy while mid-run changes still sped it
  up; user, 2026-11-08: "we didn't optimize voicing enough and wasted lot's of time and money").
- One implementation per job: every grounded cache runs Mimi and wav2vec2 through
  `training/prep/cache_build.py` → [cache-build](docs/agent-memory/cache-build.md). Before adding
  a device, model or speed lever, search where that model already runs (`MimiEncoder`,
  `get_mimi`, `Wav2Vec2`) and extend that code.
- Never print, log or commit `.env` values (`HF_TOKEN`, `HF_TOKEN_AN`, `MDC_API_KEY`, the Gemini
  and OpenRouter keys) →
  [environment](docs/agent-memory/environment.md).
- Every data-generation job from 2026-11-04 on (ASR, alignment, translation calls, TTS, Mimi
  caches, tokenizer passes) runs on the H100 box, written and optimized for CUDA; so do training
  and evaluation. The user games on the Mac and doesn't want it slowed: local runs stay small (a
  check, a sample), single-process and under `nice -n 19`, and subagents get the same rule (user,
  2026-11-04). Overnight in the user's time (UTC+7) the Mac (M4 Pro, 48 GB) may also take a share
  of a box job when that shortens it, such as MLX translation or TTS of some shards, or the
  downloads; size each night job to finish by morning (user, 2026-11-04). The Mac keeps the MLX
  inference work; the existing MLX TTS and cache builder produced the current data →
  [sft-training](docs/agent-memory/sft-training.md).
- Pick the newest version of any third-party model (ASR, TTS, LLM, aligner) the pipeline uses,
  for its efficiency and accuracy (user, 2026-11-04).
- Production runs go only through `training/h100.sh run`, which ties a run to its commit and
  HF recovery prefix → [sft-recovery](docs/agent-memory/sft-recovery.md).
- Cache rows are `[33, T]`: text, 16 target codebooks, then 16 source codebooks (English
  target, Vietnamese source in a `vi-en` cache; the other way in `en-vi`), with fixed EOS and -1
  positions. The trainer joins rows into streams and never rewrites the cache →
  [sft-data-caches](docs/agent-memory/sft-data-caches.md).
- Resume only from the newest complete model and optimizer pair of the same run →
  [sft-recovery](docs/agent-memory/sft-recovery.md).
- In MLX, give each thread its own `rustymimi.Tokenizer` and pass NumPy arrays, not MLX arrays,
  between threads → [mlx-inference](docs/agent-memory/mlx-inference.md).
- `mx.compile` only functions that see a small, fixed set of input shapes; fed a new shape per call,
  a compiled function grows memory without bound (61 GB on wav2vec2) →
  [mlx-inference](docs/agent-memory/mlx-inference.md#memory).
- q4 means 4-bit, group size 32, on layers whose input width divides by 32. A BF16 model
  directory needs `weight_dtype: bfloat16` → [mlx-inference](docs/agent-memory/mlx-inference.md).

## Topic Index

| Read when working on… | Note | Main paths |
|---|---|---|
| MLX inference, conversion, benchmarks, macOS app | [mlx-inference](docs/agent-memory/mlx-inference.md) | `inference/` |
| Swift runtime, iOS app, on-device tests | [swift-runtime](docs/agent-memory/swift-runtime.md) | `inference/swift/` |
| Mimi on neural accelerators (Neural Engine now, Qualcomm later) | [mimi-npu](docs/agent-memory/mimi-npu.md) | `inference/tools/mimi_step.py`, `inference/tools/export_mimi_coreml.py`, `inference/swift/Sources/HibikiCore/NeuralDecoder.swift` |
| SFT trainer: streams, defaults, precision, objective | [sft-training](docs/agent-memory/sft-training.md) | `training/finetune/` |
| SFT H100 launcher, checkpoints, resume, HF recovery sync | [sft-recovery](docs/agent-memory/sft-recovery.md) | `training/h100.sh`, `training/finetune/hf_sync.py`, `training/finetune/checkpoints.py` |
| SFT checkpoint selection, `eval.py` and `validate.py`, evaluation sets | [sft-evaluation](docs/agent-memory/sft-evaluation.md) | `training/finetune/eval.py`, `training/finetune/validate.py`, `training/prep/eval_sets.py`, `data/eval/` |
| GRPO: rollouts, process rewards, the update, its profiler and launcher | [grpo](docs/agent-memory/grpo.md) | `training/finetune/grpo.py`, `training/finetune/grpo_profile.py`, `training/finetune/rewards.py` |
| SFT speed: the `lm_training` fast path, benchmarks, choosing a GPU pod | [sft-speed](docs/agent-memory/sft-speed.md) | `training/finetune/lm_training.py` |
| Noisy Vietnamese source: simulated rooms, noise and bands, the noisy cache, `en-vi-val-noisy` | [noisy-source](docs/agent-memory/noisy-source.md) | `training/prep/augment.py`, `training/prep/mimi_cache_cuda.py` |
| The real-Vietnamese tranche: its decision and storage, models, pipeline and ingest, the ASR check, the row filters | [real-vi-tranche](docs/agent-memory/real-vi-tranche.md) | `training/tranche/` other than the bake-off, the translation and the cache |
| The tranche's VI→EN translation on the box: the model and its bake-off, the reply checks, speed, bursts beside voice | [tranche-translate](docs/agent-memory/tranche-translate.md) | `training/tranche/translate.py` |
| The tranche's caches: its row gates and their thresholds, running the build on the box or the Mac | [tranche-cache](docs/agent-memory/tranche-cache.md) | `training/tranche/cache.py` |
| The tranche's English voice-cloning TTS: candidates, bake-off, scores, tuning | [tranche-tts](docs/agent-memory/tranche-tts.md) | `training/tranche/bakeoff.py`, `training/tranche/voice.py`, `training/tranche/envs.sh` |
| The tranche's jobs on a rented box: bring-up, job order, the restart/watchdog/pause scripts, round 4's inputs | [tranche-box](docs/agent-memory/tranche-box.md) | `training/tranche/box/` |
| Vietnamese speech corpora: licences, file layouts, labels and order, measured samples | [vi-corpora](docs/agent-memory/vi-corpora.md) | `training/tranche/ingest.py`, the tranche repos |
| Vietnamese tokenizer swap for two-way text: build, English check, embedding init map | [vi-tokenizer](docs/agent-memory/vi-tokenizer.md) | `training/vi_tokenizer/` |
| Two-way speech: directions in caches, streams and evaluation, the Vietnamese aligner, the pilot | [two-way](docs/agent-memory/two-way.md) | `--direction`, `text_timing.ALIGNERS`, `training/prep/cache_rows.py`, `training/finetune/streams.py` |
| Every grounded cache build: rows, gates, Mimi and wav2vec2 on MLX or CUDA, each device's speed; Mimi or wav2vec2 run anywhere | [cache-build](docs/agent-memory/cache-build.md) | `training/prep/cache_build.py`, `training/prep/cache_rows.py`, `training/prep/mimi_mlx.py`, `training/prep/wav2vec2_mlx.py`, `training/prep/text_timing.py`, `training/prep/audio.py` |
| The EN-VI dataset's caches, their builds and publishing, the PhoMT cache, the row layout | [sft-data-caches](docs/agent-memory/sft-data-caches.md) | `training/prep/cache_codes.py`, `training/prep/publish_caches.py`, `training/paths.py` |
| English read-speech corpus (Common Voice, Libriheavy rows) | [en-speech-corpus](docs/agent-memory/en-speech-corpus.md) | `cv27-*` and `libriheavy-*` shards of `/Volumes/data/datasets/en_vi_speech` |
| Roadmap, GRPO plan, analyses, reports | [docs-and-plans](docs/agent-memory/docs-and-plans.md) | `docs/` |
| Python, secrets, external storage and corpora, pushing | [environment](docs/agent-memory/environment.md) | `.env`, `weights/`, `data/`, `runs/`, `/Volumes/data` |
| EN→VI transcript translation | [en-vi-translation](docs/agent-memory/en-vi-translation.md) | `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text` |
| EN→VI translation on the Gemini Batch API, claims | [en-vi-gemini](docs/agent-memory/en-vi-gemini.md) | the `claim` and `gemini` steps of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/gemini` |
| EN→VI translation on Muse through OpenRouter | [en-vi-muse](docs/agent-memory/en-vi-muse.md) | the `muse` step of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/muse` |
| Vietnamese voice-transfer TTS (VieNeu in MLX) | [vi-tts](docs/agent-memory/vi-tts.md) | `training/prep/tts_en_speech.py`, `training/prep/vieneu_mlx.py`, `vi_audio` of `/Volumes/data/datasets/en_vi_speech` |
| EN-VI speech dataset, its upload and the cache repo | [en-vi-speech-dataset](docs/agent-memory/en-vi-speech-dataset.md) | `training/prep/publish_en_vi_speech.py`, `/Volumes/data/datasets/en_vi_speech`, `huybik/en-vi-speech`, `anquachdev/common-vi-en`, `huybik/hibiki-zero-vi-full-sft` (dataset) |

## Canonical resources

- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
- Run recovery repo (`HIBIKI_HF_REPO`): https://huggingface.co/huybik/hibiki-zero-vi-full-sft
- Training caches (the dataset repo; the model repo of the same name holds runs):
  https://huggingface.co/datasets/huybik/hibiki-zero-vi-full-sft
- EN-VI speech and text pairs: https://huggingface.co/datasets/anquachdev/common-vi-en
- Real-Vietnamese tranche: https://huggingface.co/datasets/anquachdev/en-vi-speech-20k (public,
  gated; also PhoMT's pairs, formerly `anquachdev/PhoMT-en-vi-speech`) and its caches,
  https://huggingface.co/datasets/anquachdev/mimi-en-vi-speech-20k (gated). The first tranche's
  old repos: `huybik/hibiki-zero-vi-real` and `huybik/hibiki-zero-vi-real-restricted` (private;
  never public)

## Maintaining context

```text
CLAUDE.md (imports this file)
  └─ CONTEXT.md: orientation, invariants, Topic Index
       └─ docs/agent-memory/<topic>.md: one note per aspect (what is true and how to work on it),
          read when work touches its paths
```

- Record current facts checked against the code, and guidance on how to work (decisions,
  feedback) with its reason. This replaces each agent's local memory: save here what you would
  have saved there, so everyone shares it. Link to existing docs instead of copying, keep one
  owner per fact, and keep no history.
- Update the owning note in the same commit as the change. Knowledge with no owner gets a new
  note, added to the Topic Index.
- A note owns a job, not a file: one note lists every path that does a job (runs a model, builds
  a cache), so whoever opens any of them is sent there.
- Change this file only when orientation, an invariant or the index changes, and update
  `Last verified` when you do.
- Keep this file under ~1,500 words and each note under 300 lines. Split a subject into a new note
  before one grows past that.
- Each note opens with `Read when touching …`, a link back here, and links to its sections.
