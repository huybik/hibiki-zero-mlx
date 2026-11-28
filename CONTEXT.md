# Hibiki MLX — project context

Last verified: 2026-11-28

## Purpose

This repo runs Hibiki-Zero simultaneous speech translation:

- MLX inference on Apple Silicon (upstream's 3B for FR/ES/PT/DE→EN, or any staged checkpoint,
  quantized or BF16);
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
    ─► prep/noisy_cache (CUDA or MLX: the source, either language, re-encoded under prep/augment) ─► *-noisy
    ─► h100.sh run → finetune/train.py ─► runs/, HF recovery repo
  HF PhoMT cache ─► data/cache/phomt-grounded-v2-5s ─────────┘ (one more --cache-dir, ablation)
inference/ (Apple Silicon, MLX)
  checkpoints ─► tools/convert_mlx.py (q6/q4/bf16) ─► weights/… ─► main.py · hibiki_mlx/ · macos/
                                                    │       └─► swift/ (MLX Swift: CLI, iOS bench app)
                                                    └─► tools/gate.py (scored as training scores)
  Mimi checkpoint ─► tools/export_mimi_coreml.py (from tools/mimi_step.py) ─► mimi-decoder.mlmodelc ─► swift/
  heads or depformer_parallel student (bf16) ─► tools/export_lm_coreml.py (from tools/lm_step.py) ─► lm-trunk, lm-heads or lm-depformer-k .mlmodelc ─► swift/
```

- `inference/` is the Apple Silicon runtime: `main.py`, the `hibiki_mlx/` runtime, the
  vendored `moshi-mlx/`, `tools/`, `assets/leon.wav`, the `macos/` test app and `swift/`, the
  native MLX Swift runtime with its iOS benchmark app.
- `training/` holds `h100.sh`, shared `paths.py`, `prep/` (the EN-VI dataset publisher, the
  Vietnamese translation of its English text, the Vietnamese TTS in MLX, the grounded cache build
  every cache builder shares (`cache_build`, MLX or CUDA) and the EN-VI dataset's builder on it,
  the evaluation-set builder, the noisy source re-encode), `finetune/` (the generic SFT trainer, GRPO, evaluation
  and HF recovery sync), `tranche/` (the real-Vietnamese tranche on the box: its envs
  (`box/envs.sh`), the ingest of the corpora's segments to the tranche repos, their translation,
  VieNeu's TTS both ways (the tranche's English, the EN-VI dataset's Vietnamese re-voiced), the row
  filters and its cache builder) and `vi_tokenizer/` (the Vietnamese tokenizer swap, on the box's
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
- The fifth, `twoway_sft` (2026-11-11 to 13, an H100), trained both directions of round 4e's rows
  for two epochs from round 4's final model on the Vietnamese tokenizer swap. Scored with the
  input level stage: EN→VI FLEURS-en 7.8 → 33.7 chrF, VI→EN FLEURS dev 41.3 → 39.6 and
  MultiMed-ST 37.3 → 37.7 ([two-way](docs/agent-memory/two-way.md#two-way-run)).
- [docs/vi_en_parity_plan.md](docs/vi_en_parity_plan.md) is the active plan (2026-11-01):
  Hibiki-Zero 3B from training to the phone. For the data,
  the `en_speech` corpus is built and all its transcripts are translated to Vietnamese:
  the first tranche and all Common Voice on Claude
  ([en-vi-translation](docs/agent-memory/en-vi-translation.md)),
  2,000 h of Libriheavy on Gemini ([en-vi-gemini](docs/agent-memory/en-vi-gemini.md)) and the
  rest on Muse ([en-vi-muse](docs/agent-memory/en-vi-muse.md)). The corpus now lives, with its
  translations, in the EN-VI dataset, published as `anquachdev/common-vi-en`, the one copy used ([en-vi-speech-dataset](docs/agent-memory/en-vi-speech-dataset.md)). VieNeu-TTS voiced the
  Vietnamese lines in each row's own English voice, filling the dataset's `vi_audio` shard by
  shard, the Mac top down and anquachdev's hardware bottom up, meeting through the Hub; all
  1,835 shards were voiced on 2026-10-27 ([vi-tts](docs/agent-memory/vi-tts.md)).
  `cache_codes` builds the training caches from it in MLX;
  the full build ran on the Mac from 2026-10-27 to 2026-11-02: 1,835 shards (3.45M rows), all on the
  Hub ([sft-data-caches](docs/agent-memory/sft-data-caches.md#builds)). `publish_caches`
  commits each shard to the cache repo as it is built, and anquachdev's machine may build bottom
  up, meeting the Mac through the Hub as for the TTS
  ([sft-data-caches](docs/agent-memory/sft-data-caches.md#publishing)).
- Next (user, 2026-11-04): a learning-rate test, then a third VI→EN epoch only if a higher rate
  won (none did: no third epoch), then the real-Vietnamese tranche (its SFT at 3e-6)
  ([real-vi-tranche](docs/agent-memory/real-vi-tranche.md)) and two-way speech in one model,
  prepared in parallel; GRPO last ([plan](docs/vi_en_parity_plan.md#order)). Then (user,
  2026-11-10): two-way before GRPO, so after round 4e one two-way run on all its rows, then one GRPO on
  both directions, and the phone student cut from the two-way model. GRPO is built in
  `training/finetune/grpo.py` on the SFT trainer's run directory, loss kernel and validation, and
  is profiled on the box phase by phase before its run ([grpo](docs/agent-memory/grpo.md)).

## Essential invariants

- Profile every job before its full run (a cache build, an encode, a training or GRPO run, an
  evaluation pass) and optimize it until its costliest resource stays busy; only then start the
  real run. Time each stage on a few
  shards, profile the GPU kernels of a few steps (`torch.profiler`: stage times hid generation's
  whole-cache attention through 1,600 GRPO updates; user, 2026-11-25), sample the GPU (`nvidia-smi`
  every 0.5 s) and know the CPU budget (a container's quota
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
- Record every validation and test in full: each set's chrF, BLEU, End Offset, length ratio and
  health (and LAAL when it ran with `--laal`), with the model, update, seed and date. Copy their raw outputs off a box before
  it ends, to `/Volumes/data/hibiki-zero-evals/` (user, 2026-11-15) →
  [sft-evaluation](docs/agent-memory/sft-evaluation.md#recording-results).
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
  [sft-training](docs/agent-memory/sft-training.md). The EN→VI data of the next two-way run is
  the exception: the user put it on the Mac and its external drive (2026-11-21) →
  [envi-data](docs/agent-memory/envi-data.md).
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
- Source audio reaches Mimi only through the input level stage (`inference/hibiki_mlx/level.py`,
  Swift's `Audio.Leveller`): in the MLX and Swift runtimes, `eval.py` and the trainers' validation
  (`common.read_source`). Mimi codes quiet speech so differently that the model translated it
  worse, and under −35 dBFS often not at all. The training caches are coded without it: their
  sources are loud → [sft-evaluation](docs/agent-memory/sft-evaluation.md#input-level).
- In MLX, give each thread its own `rustymimi.Tokenizer` and pass NumPy arrays, not MLX arrays,
  between threads → [mlx-inference](docs/agent-memory/mlx-inference.md).
- `mx.compile` only functions that see a small, fixed set of input shapes; fed a new shape per call,
  a compiled function grows memory without bound (61 GB on wav2vec2) →
  [mlx-inference](docs/agent-memory/mlx-inference.md#memory).
- A model directory's `config.json` names its MLX weights (`moshi_name`), tokenizer and sampling
  (`lm_gen_config`), and records any quantization per layer (`quantization`; `--bits 6`, what
  ships, is 6 bits on the transformer linears, 8 on the embeddings and heads), which both
  runtimes load by, strictly. A Core ML LM's directory (`lm_coreml`, the heads student) names no
  MLX weights, and only the Swift runtime runs it →
  [lm-npu](docs/agent-memory/lm-npu.md#export). The gate feeds a file as training's evaluation does (source EOS,
  20 s of silence, stop at the text EOS after it), so Mac scores compare with the box's (rows
  whose EOS comes before the source ends aside); main.py's file mode also stops after 5 s
  without a word →
  [mlx-inference](docs/agent-memory/mlx-inference.md#model-directories).

## Topic Index

| Read when working on… | Note | Main paths |
|---|---|---|
| MLX inference, conversion and quantization, the quantization gate, benchmarks, macOS app | [mlx-inference](docs/agent-memory/mlx-inference.md) | `inference/`, `inference/tools/convert_mlx.py`, `inference/tools/gate.py` |
| Swift runtime, iOS app, on-device tests | [swift-runtime](docs/agent-memory/swift-runtime.md) | `inference/swift/` |
| Mimi on neural accelerators (Neural Engine now, Qualcomm later) | [mimi-npu](docs/agent-memory/mimi-npu.md) | `inference/tools/mimi_step.py`, `inference/tools/export_mimi_coreml.py`, `inference/swift/Sources/HibikiCore/NeuralDecoder.swift` |
| A phone student's LM on the Neural Engine (the heads, or `depformer_parallel`'s slices): export, runtime, precision, phone speed | [lm-npu](docs/agent-memory/lm-npu.md) | `inference/tools/lm_step.py`, `inference/tools/export_lm_coreml.py`, `inference/swift/Sources/HibikiCore/NeuralLM.swift` |
| SFT trainer: streams, defaults, precision, objective, schedules, checkpoint averaging | [sft-training](docs/agent-memory/sft-training.md) | `training/finetune/` |
| SFT H100 launcher, checkpoints, resume, HF recovery sync | [sft-recovery](docs/agent-memory/sft-recovery.md) | `training/h100.sh`, `training/finetune/hf_sync.py`, `training/finetune/checkpoints.py` |
| SFT checkpoint selection, `eval.py` and `validate.py`, evaluation sets, the input level stage | [sft-evaluation](docs/agent-memory/sft-evaluation.md) | `training/finetune/eval.py`, `training/finetune/validate.py`, `training/prep/eval_sets.py`, `data/eval/`, `inference/hibiki_mlx/level.py` |
| Voice scores of every decoded translation: speaker similarity to the source, clarity, the speech cut, what models score | [voice-scores](docs/agent-memory/voice-scores.md) | `training/finetune/voice_scores.py`, `training/prep/kaldi.py`, `hibiki_helpers.decode_wavs` |
| GRPO: rollouts, process rewards, the update, its profiler and launcher | [grpo](docs/agent-memory/grpo.md) | `training/finetune/grpo.py`, `training/finetune/grpo_profile.py`, `training/finetune/rewards.py` |
| SFT speed: the `lm_training` fast path, benchmarks, choosing a GPU pod | [sft-speed](docs/agent-memory/sft-speed.md) | `training/finetune/lm_training.py` |
| Noisy source (Vietnamese and English): simulated rooms, noise and bands, the noisy caches, `en-vi-val-noisy` and `en-vi-val-en-noisy`, the rooms where nobody speaks (GRPO's, SFT's `--room-share`) | [noisy-source](docs/agent-memory/noisy-source.md) | `training/prep/augment.py`, `training/prep/noisy_cache.py`, `streams.run_rooms` |
| The real-Vietnamese tranche: its decision and storage, models, pipeline and ingest, the ASR check, the row filters | [real-vi-tranche](docs/agent-memory/real-vi-tranche.md) | `training/tranche/` other than the TTS, the translation and the cache |
| The tranche's VI→EN translation on the box: the model and its bake-off, the reply checks, speed, bursts beside voice | [tranche-translate](docs/agent-memory/tranche-translate.md) | `training/tranche/translate.py` |
| The tranche's caches: its row gates and their thresholds, running the build on the box or the Mac | [tranche-cache](docs/agent-memory/tranche-cache.md) | `training/tranche/cache.py` |
| YODAS3's Vietnamese in the tranche: rows cut from caption runs, the ASR in ingest, its CER gate, ingest and voice speed on the 5090, the pilot | [tranche-yodas](docs/agent-memory/tranche-yodas.md) | `training/tranche/yodas.py` |
| VieNeu TTS both ways on the box: the tranche's English, the EN-VI dataset's Vietnamese re-voiced where broken (`run en_speech`, its `vi_check` parts), the ASR check, how the TTS was chosen, which TTS voiced what | [tranche-tts](docs/agent-memory/tranche-tts.md) | `training/tranche/voice.py`, `training/tranche/vieneu_frame.py`, `training/tranche/decode_attention.py`, `training/tranche/onnx_graph.py`, `training/tranche/box/envs.sh` |
| VieNeu's speed on the 5090: its benches against VoxCPM2 and ZONOS3-Mini, each optimization, production and end-to-end speed, the FP8 ASR server, why the plan's 2,000× was not reached | [tranche-tts-scaleup](docs/agent-memory/tranche-tts-scaleup.md) | `/Volumes/data/hibiki-zero-evals/box-62513802/ttsbench-*` |
| The tranche's jobs on a rented box: bring-up, job order, the restart/watchdog/pause scripts, round 4's inputs | [tranche-box](docs/agent-memory/tranche-box.md) | `training/tranche/box/` |
| Vietnamese speech corpora: licences, file layouts, labels and order, measured samples | [vi-corpora](docs/agent-memory/vi-corpora.md) | `training/tranche/ingest.py`, the tranche repos |
| Vietnamese tokenizer swap for two-way text: build, English check, embedding init map, the runtimes on it | [vi-tokenizer](docs/agent-memory/vi-tokenizer.md) | `training/vi_tokenizer/` |
| The phone student cut from the two-way model: the cut, training and distilling a student, checks | [phone-student](docs/agent-memory/phone-student.md) | `training/student/`, `training/finetune/streams.py` (`model_rows`) |
| The parallel-heads student: its delays, 16 codebooks with 8–15 at one delay, speed, runs and validations | [student-heads](docs/agent-memory/student-heads.md) | `training/student/heads.py`, `restart.sh student-heads` |
| The slices student: the heads student's trunk with depformer slices (9 of 512 × 2, the last for codebooks 8–15), its cut, training and runs, the wide 1024 × 3 retrain | [student-slices](docs/agent-memory/student-slices.md) | `training/student/slices.py`, `train.py --freeze-trunk`, `restart.sh student-slices-5090`, `inference/moshi-mlx/moshi_mlx/models/lm.py` |
| The slices student's phone release: its fused Core ML export, Speech Live, its voice on the phone against the teacher | [slices-phone](docs/agent-memory/slices-phone.md) | `release/student_slices_coreml6`, onevoice-ai-challenge `apps/ios` |
| Training the heads student against the Core ML palettization: the method, `--park-states`, the 6-bit run, its tests and Neural Engine agreement | [student-palettize](docs/agent-memory/student-palettize.md) | `training/student/palettize.py`, `train.py --palettize-bits` |
| The slices student's on-policy run (`both_slices_z`): its profile and sizes, the run, its 6-bit finish and export to the phone | [slices-on-policy](docs/agent-memory/slices-on-policy.md) | `grpo_launch.sh both_slices_z`, `runs/both_slices_z` |
| The phone student's GRPO run, its evaluations and tests, the shipped update 2,000 | [student-grpo](docs/agent-memory/student-grpo.md) | `runs/grpo_student`, `/Volumes/data/hibiki-zero-evals/` |
| On-policy distillation and centered advantages in GRPO; the student's recipe experiments | [on-policy-distillation](docs/agent-memory/on-policy-distillation.md) | `training/finetune/grpo.py`, `training/finetune/rewards.py` |
| The heads student's on-policy runs: the probes' results, `both_heads_z`, `both_heads_z2` and its pre-run scores | [heads-on-policy-runs](docs/agent-memory/heads-on-policy-runs.md) | `runs/both_heads_z*`, `training/tranche/box/grpo_launch.sh` |
| Exposure bias and the 3B's recipe: scratch or fine-tune, scheduled sampling in SFT, GRPO anchored to its start, their experiments | [exposure-bias](docs/agent-memory/exposure-bias.md) | `train.py --sampled-text`, `grpo.py --teacher-weight` |
| Those experiments' runs: both SFT arms, their GRPO, the anchored probes, the quiet-clip loss (a level effect), noise | [exposure-bias-runs](docs/agent-memory/exposure-bias-runs.md) | `/Volumes/data/hibiki-zero-evals/box-62170174/` |
| The 3B teacher's next round: the full two-way SFT on the new EN→VI data, its gates, decay branches and GRPO, time, disk and card | [teacher-next](docs/agent-memory/teacher-next.md) | `restart.sh two-way`, `grpo_launch.sh` |
| Two-way speech: directions in caches, streams and evaluation, the run's tokenizer checks, the Vietnamese aligner, the two-way run and its box jobs | [two-way](docs/agent-memory/two-way.md) | `--direction`, `--tokenizer`, `common.run_tokenizer`, `text_timing.ALIGNERS`, `training/prep/cache_rows.py`, `training/finetune/streams.py` |
| The next two-way run's EN→VI data: `en_speech` both ways, the noisy copies of every English source, where each is on the Hub, the Mac run that builds them | [envi-data](docs/agent-memory/envi-data.md) | `/Volumes/data/datasets/envi_mac/envi_mac.py`, `en-vi/*-noisy` on the Hub |
| Every grounded cache build: rows, gates, Mimi and wav2vec2 on MLX or CUDA, the Mac's speed; Mimi or wav2vec2 run anywhere | [cache-build](docs/agent-memory/cache-build.md) | `training/prep/cache_build.py`, `training/prep/cache_rows.py`, `training/prep/mimi_mlx.py`, `training/prep/wav2vec2_mlx.py`, `training/prep/wav2vec2_coreml.py`, `training/prep/text_timing.py`, `training/prep/audio.py` |
| A cache build or noisy copy on a GPU: speed and memory levers, measured per GPU (H100, H200, RTX 5090), the window and small-RAM boxes | [cache-speed-cuda](docs/agent-memory/cache-speed-cuda.md) | `CUDAModels` in `training/prep/cache_build.py`, `noisy_cache encode --device cuda` |
| The EN-VI dataset's caches, their builds and publishing, the PhoMT cache, the row layout | [sft-data-caches](docs/agent-memory/sft-data-caches.md) | `training/prep/cache_codes.py`, `training/prep/publish_caches.py`, `training/paths.py` |
| English read-speech corpus (Common Voice, Libriheavy rows) | [en-speech-corpus](docs/agent-memory/en-speech-corpus.md) | `cv27-*` and `libriheavy-*` shards of `/Volumes/data/datasets/en_vi_speech` |
| Roadmap, GRPO plan, analyses, reports | [docs-and-plans](docs/agent-memory/docs-and-plans.md) | `docs/` |
| Python, secrets, external storage and corpora, pushing | [environment](docs/agent-memory/environment.md) | `.env`, `weights/`, `data/`, `runs/`, `/Volumes/data` |
| EN→VI transcript translation | [en-vi-translation](docs/agent-memory/en-vi-translation.md) | `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text` |
| EN→VI translation on the Gemini Batch API, claims | [en-vi-gemini](docs/agent-memory/en-vi-gemini.md) | the `claim` and `gemini` steps of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/gemini` |
| EN→VI translation on Muse through OpenRouter | [en-vi-muse](docs/agent-memory/en-vi-muse.md) | the `muse` step of `training/prep/translate_en_speech.py`, `/Volumes/data/datasets/en_speech_vi_text/muse` |
| Vietnamese voice-transfer TTS in MLX on the Mac, which voiced the EN-VI dataset (the box's: tranche-tts) | [vi-tts](docs/agent-memory/vi-tts.md) | `training/prep/tts_en_speech.py`, `training/prep/vieneu_mlx.py`, `vi_audio` of `/Volumes/data/datasets/en_vi_speech` |
| EN-VI speech dataset, its upload and the cache repo | [en-vi-speech-dataset](docs/agent-memory/en-vi-speech-dataset.md) | `training/prep/publish_en_vi_speech.py`, `/Volumes/data/datasets/en_vi_speech`, `huybik/en-vi-speech`, `anquachdev/common-vi-en`, `huybik/hibiki-zero-vi-full-sft` (dataset) |

## Canonical resources

- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
- Run recovery repo (`HIBIKI_HF_REPO`): https://huggingface.co/huybik/hibiki-zero-vi-full-sft
  (its `release/twoway_grpo1400/`: the shipping two-way model, self-contained, with a model card;
  `release/student_heads_coreml6/` and `release/student_slices_coreml6/`: the phone's heads and
  slices students in Core ML, with cards)
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
