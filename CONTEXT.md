# Hibiki MLX — project context

This repository maintains two paths only:

1. q4 MLX inference for Hibiki-Zero on Apple Silicon;
2. base-start, full-model Vietnamese-to-English SFT on CUDA.

The mobile model track now has the frozen 12-layer contract, strict CUDA AR and
parallel-head training, BF16 qualification/export gates, deterministic
PyTorch/MLX parity tooling, and strict MLX q4 group-size-32 pack
conversion/inference. Student packs use an explicit raw pre-undelay
previous-head frame; stock Swift still lacks `parallel_v1` and must not be
called compatible.

Historical design, vision, and report documentation is retained under `docs/`.
Generated FLEURS data is excluded from both the active tree and Git history.

## Inference

- `main.py` is the CLI for file and microphone translation.
- `hibiki_mlx/pipeline.py` owns model loading and the three-thread
  Mimi-encode → LM → Mimi-decode pipeline.
- `moshi-mlx/` is the minimal vendored MLX language-model runtime. Its required
  Hibiki deltas are GQA (`kv_repeat`), configurable `hidden_scale`,
  `rope_concat`, per-slice depformer output LayerNorm, and the compact
  one/two-pass `parallel_v1` head.
- Supported weights are q4 with `group_size=32`. `3b` resolves to `weights/`;
  legacy custom directories remain supported, while student directories require
  the complete hash-validated pack contract.
- Each codec thread owns a separate `rustymimi.Tokenizer`. Queues carry NumPy
  arrays because lazy MLX graphs cannot move across creating threads.
- File inference adds an 8-second silence tail and stops after 12 sustained PAD
  frames to flush translation lag without extended hallucination.
- Runtime gates are `scripts/verify_mlx_q4.py` and
  `scripts/bench.py --model 3b --silence`; student packs additionally use
  `scripts/verify_student_parity.py`.
- `scripts/check_swift_compat.py` requires strict q4 group-size-32 reload plus
  valid config, tokenizer, and Mimi sidecars, and deliberately rejects
  `parallel_v1` as unsupported by stock Swift.

## Training

- Start every new-agent or new-pod training session with
  `docs/finetune.md`. It is the copy-paste handoff for fresh setup, artifact
  restore, preflight/smoke, the stop-before-launch boundary, and exact recovery.
- `finetune/train.py` always trains every model parameter. There is no LoRA or
  adapter path; isolated pilot-only warm-start/replay modes remain while the
  final full-training receipt is being locked.
- Training starts from `weights/hibiki-pytorch-77f82164@110.safetensors`.
  `--resume-checkpoint` is only for interruption recovery within the same run.
- CUDA uses fp32 master weights, bf16 autocast, fused AdamW, causal SDPA, fixed
  length-sorted 16-frame buckets, and `--max-frames 280`; 80 GB H100s run batch
  8 with two accumulation steps.
- `finetune/h100.sh` pins the pod environment, verifies staged artifacts and
  evaluation audio, selects the 80/94 GB batch recipe, and gates training on a
  save/eval/resume smoke.
- `HIBIKI_RECIPE=grounded-v2` selects isolated CTC word-timed caches, 95/5
  PhoMT/FLEURS sampling, conservative cosine SFT, eligibility-only best saves,
  and paired source-dependence evaluation. `HIBIKI_PILOT=1` forces separate
  `*_grounded_v2_pilot` caches, smoke/run directories, and HF prefix; exactly
  104 evenly sampled PhoMT shards feed a frozen 50k-row membership for 1,000
  steps with 100-step warmup. Pilot target-audio inputs are masked and audio
  loss is zero. Full grounded keeps 1,000-step warmup and rejects pilot limits.
  PhoMT is pinned and CTC rows below 0.5 are rejected. Legacy remains default.
- `finetune/common.py` owns cached data, losses, schedules, exact full-model
  checkpoint I/O, free-running generation, paired metrics, RNG isolation, and
  frozen duration-matched evaluation/training derangements. Complete checkpoint
  pairs are published atomically and pre-rotated to avoid transient disk spikes;
  loading rejects missing and unexpected tensor keys.
- `finetune/cache_phomt_stream.py` builds the published PhoMT cache directly
  from parquet with bounded download prefetch and Hugging Face Xet. MPS runs keep
  the CTC dynamic program on-device with one result transfer, release each batch,
  and bound concurrent workers by row and audio-sample budgets. Its H100 profile
  batches CTC Viterbi across rows; `h100.sh cache-grounded` supervises bounded
  workers and builds all grounded-v2 PhoMT/FLEURS caches.
  `finetune/publish_grounded_cache.py` validates and checksum-publishes
  the complete cache under an isolated dataset prefix. `remote_dataset/download_fleurs_vi_en.py` →
  `finetune/build_pairs.py` → `finetune/cache_codes.py` builds FLEURS inputs.
  `remote_dataset/download_covost2.py` materializes the pinned healthy FR→EN
  evaluator control.
- The grounded-v2 PhoMT rebuild is running on the H100 from the SHA-verified,
  contiguous 90-shard Mac prefix (`shard_00000.pt` through `shard_00089.pt`).
  It contains 52,939 accepted rows and has aggregate manifest SHA
  `7b76432f7034284d440c27a433e48c791ca4e1f5c6daa6e92e8c23bcef2b4e56`.
  Five CUDA workers run from commit `8e310ff` at a measured 7.0 shards/min with
  about 5 GiB GPU headroom; the detached pipeline validates all 1,377 shards,
  builds grounded FLEURS caches, then publishes and verifies the isolated
  dataset `grounded-v2/` prefix. The Mac copy remains until remote publication
  succeeds.
- `finetune/validate.py` is teacher-forced diagnostics only. `finetune/eval.py`
  evaluates correct and shuffled sources at fixed-seed text temperature 0.4,
  writing condition and consolidated artifacts. Promotion requires correct-source
  health plus calibrated BLEU/chrF gaps, then ranks by `(BLEU, chrF)`.
- `finetune/hf_sync.py` maintains two recovery pairs plus the best model under
  `full_run/` in the public `huybik/hibiki-zero-vi-full-sft` model repo;
  final sync also preserves run configuration, pilot membership, and compact
  paired-evaluation CSV/JSON artifacts. `h100.sh` verifies a shared run identity,
  supervises sync, and protects the local resume point before training restarts.

After the `docs/finetune.md` handoff, use `docs/training_plan.md` for the exact
recipe and `docs/validation_plan.md` for qualification thresholds.
Paired controls lock text temperature 0.4 and source-gap gates of 1.0 BLEU plus
5.0 chrF. Healthy French passed at 23.08/38.80; Vietnamese base and phase-1
failed source dependence at -0.07/1.23 and 0.01/1.03. Phase-1's 19.57 absolute
chrF was therefore mostly target-side modeling. Treat current early text timing
as a diagnostic. The corrected ordinary pilot failed promotion at every
0/250/500/750/1,000 milestone; final health was 126/128 nonempty, 116 EOS, and
24 repeated-4gram failures, with BLEU/chrF gaps -0.07/0.22. Its exact 50k
manifest SHA is `52ef91a79dc09fb6c00a6f800bf087f2228b7c0842ecb2705ac873d3ef3a458f`.
The high-delay retry is explicit `HIBIKI_HIGH_DELAY_PILOT=1`, uses deterministic
uniform ratios `[0.75, 1.0]`, and owns isolated `*_pilot_high_delay` artifacts.
It reconstructs that membership exactly, hard-gates training at 480 frames,
preserves production order, and uses physical batch 8 / accumulation 2.
Teacher-forced validation retains all rows under a separate 704-frame cap at
batch 1. The exact high-delay retry also failed every promotion gate: at step
1,000 it produced BLEU/chrF 0.03/9.34, gaps 0.01/0.72, 31 EOS, and 111
repetition failures. Delay alone is rejected; the next isolated pilot adds a
duration-matched shuffled-source margin loss before considering acoustic
preadaptation. `HIBIKI_CONTRASTIVE_PILOT=1` reuses the verified high-delay cache
but owns `*_pilot_high_delay_contrastive` smoke/run/HF artifacts. It freezes a
no-duplicate-ID donor permutation, preserves each target's source duration/EOS,
and adds weight-1 `relu(0.5 + correct_nll - shuffled_nll)` over English content.
The sequential correct/shuffled forwards use physical batch 4 / accumulation 4
on the 94 GB H100 while preserving effective batch 16. That pilot also failed:
by step 1,000 its teacher-forced shuffled-minus-correct NLL gap reached 1.04,
but free-running BLEU/chrF gaps remained 0.04/0.61 with 69 repetition failures
and mean length ratio 2.95. Contrastive text ranking is rejected; Vietnamese
acoustic preadaptation is the next diagnostic before any full SFT. The bounded
diagnostic is `HIBIKI_ASR_PREADAPT=1`: reuse the exact high-delay 50k cohort,
discard English targets, keep Vietnamese source codes through source EOS, then
supervise the Vietnamese transcript with target audio absent. It owns the
`*_grounded_v2_pilot_vi_asr_preadapt` namespace, uses physical batch 4 /
accumulation 4 on the 94 GB H100, and hard-gates train/validation at 672/640
frames. Paired Vietnamese ASR must pass normal health, 1.0 BLEU / 5.0 chrF
source gaps, correct chrF at least 50, and WER at most 0.60. This tests whether
the temporal backbone can learn Vietnamese acoustics; it is not Kyutai's
multilingual audio pretraining. The 1,000-step pilot saw only 16,000/50,000
ordered positions. At step 1,000 it passed health and source dependence
(BLEU/chrF gaps 1.18/7.64) but failed absolute ASR (chrF 18.31, WER 0.678).
It proves learnable Vietnamese routing but cannot initialize translation. The
exact raw-Vietnamese one-epoch retry also failed promotion. At step 3,125 it
passed health (128/128 nonempty and EOS, two repetition failures, 0.79 length
ratio) and source dependence (6.65 BLEU / 15.43 chrF gaps), but correct chrF was
26.72. Its recorded WER 0.639 used the old ASCII-only normalizer; corrected
diacritic-insensitive WER is 0.775. The tokenizer itself is the bottleneck for
this diagnostic: raw Vietnamese costs 4.14 pieces/word and 110/128 hypotheses
contain invalid-byte replacement characters. Deterministic ASCII Vietnamese
costs 1.87 pieces/word. The corrected `HIBIKI_ASR_ASCII=1` one-epoch run
qualified at step 3,125: 127/128 nonempty, 128 EOS, zero repetition failures,
BLEU/chrF 27.85/53.26, WER 0.514, and source gaps 27.74/34.48. Its promoted
parent SHA is `d37d69103bff8f128b9b69fc9634a018d8ab5c5c58dbb0b5cc98ecf5a26f92ca`.
The retired `HIBIKI_ASR_TRANSLATION_PILOT=1` experiment used a fresh optimizer,
ordinary-timing exact 50k cohort, physical batch 16, masked target audio, zero
audio loss, and 1,000 translation steps initialized from that exact parent. Its
isolated `*_grounded_v2_pilot_vi_asr_warmstart` run failed: at step
1,000, correct-source BLEU/chrF was 0.06/8.44, source gaps were 0.01/-0.36, and
24 rows failed the repetition gate, so no best checkpoint was promoted. A plain
fresh-optimizer switch from qualified ASCII ASR to English translation is
rejected. `HIBIKI_ASR_REPLAY_TRANSLATION_PILOT=1` then jointly trained the same
translation objective with a deterministic batch-4, weight-1 ASCII-ASR replay
forward. The memory-safe run peaked near 80 GiB and completed, but failed every
paired milestone. At step 1,000, correct-source BLEU/chrF was 0.13/7.73 and the
source gaps were -0.04/-0.24; no checkpoint qualified. Joint ASR replay is
rejected. The final bounded diagnostic is
`HIBIKI_POST_SOURCE_EOS_TRANSLATION_PILOT=1`: reconstruct the exact ordinary 50k
manifest, initialize the exact qualified ASCII-ASR parent, delete target-audio
inputs at the shared dataset boundary, retain Vietnamese codes through source
EOS, then supervise only the English sentence. It uses no ASR replay, owns
isolated `*_grounded_v2_pilot_vi_post_source_eos_translation` artifacts,
hard-gates transformed train/validation lengths at 400/480, uses physical batch
8 / accumulation 2, 100-step warmup, deterministic ascending transformed
validation, and paired evaluation at 0/250/500/750/1,000 with a 24-second
generation tail. Smoke reverses validation to exercise its observed longest row.
The frozen policy persists tokenizer, ordered English-text hash, row count, and
observed cohort maximum; recovery checkpoints are retained at steps 500 and
1,000. Do not start full SFT until this receipt is decided and the complete
grounded cache is published and verified.

## Canonical resources

- Published training caches: https://huggingface.co/datasets/huybik/hibiki-zero-vi-full-sft/tree/main
- Training checkpoints and recovery artifacts: https://huggingface.co/huybik/hibiki-zero-vi-full-sft
- Source PhoMT Vietnamese–English speech dataset: https://huggingface.co/datasets/anquachdev/PhoMT-en-vi-speech

## Environment

- Local Python work uses `/opt/homebrew/Caskroom/miniconda/base/bin/python`.
- The ignored `.env` contains the HF credential used for downloads and recovery
  sync. Source it when needed; never print, log, or commit its value.
- Inference requires MLX 0.31+, NumPy, rustymimi, sentencepiece, sphn, and
  sounddevice for microphone mode.
- Training additionally requires a CUDA-compatible PyTorch build, `moshi`
  0.2.13, safetensors, sacrebleu, datasets, soundfile, pyarrow, and
  huggingface-hub.
