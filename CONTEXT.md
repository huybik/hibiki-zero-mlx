# Hibiki MLX — project context

This repository maintains two paths only:

1. q4 MLX inference for Hibiki-Zero on Apple Silicon;
2. base-start, full-model Vietnamese-to-English SFT on CUDA.

The mobile target is a future distilled 1B Hibiki-Zero student with parallel
target-codebook heads. It is not implemented yet.

Historical design, vision, and report documentation is retained under `docs/`.
Generated FLEURS data is excluded from both the active tree and Git history.

## Inference

- `main.py` is the CLI for file and microphone translation.
- `hibiki_mlx/pipeline.py` owns model loading and the three-thread
  Mimi-encode → LM → Mimi-decode pipeline.
- `moshi-mlx/` is the minimal vendored MLX language-model runtime. Its required
  Hibiki deltas are GQA (`kv_repeat`), configurable `hidden_scale`,
  `rope_concat`, and per-slice depformer output LayerNorm.
- Supported weights are q4 with `group_size=32`. `3b` resolves to `weights/`;
  custom staged Hibiki-Zero directories can be passed explicitly.
- Each codec thread owns a separate `rustymimi.Tokenizer`. Queues carry NumPy
  arrays because lazy MLX graphs cannot move across creating threads.
- File inference adds an 8-second silence tail and stops after 12 sustained PAD
  frames to flush translation lag without extended hallucination.
- Runtime gates are `scripts/verify_mlx_q4.py` and
  `scripts/bench.py --model 3b --silence`.
- `scripts/check_swift_compat.py` requires strict q4 group-size-32 reload plus
  valid config, tokenizer, and Mimi sidecars.

## Training

- Start every new-agent or new-pod training session with
  `docs/finetune.md`. It is the copy-paste handoff for fresh setup, artifact
  restore, preflight/smoke, the stop-before-launch boundary, and exact recovery.
- `finetune/train.py` always trains every model parameter. There is no LoRA,
  warm-start adapter, replay sampler, or alternate batch scheduler.
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
  frozen duration-matched derangements. Complete checkpoint pairs are
  published atomically and pre-rotated to avoid transient disk spikes; loading
  rejects missing and unexpected tensor keys.
- `finetune/cache_phomt_stream.py` builds the published PhoMT cache directly
  from parquet with bounded download prefetch and Hugging Face Xet. MPS runs keep
  the CTC dynamic program on-device with one result transfer, release each batch,
  and bound concurrent workers by row and audio-sample budgets. Its H100 profile
  batches CTC Viterbi across rows; `h100.sh cache-grounded` supervises four
  workers and builds all grounded-v2 PhoMT/FLEURS caches.
  `finetune/publish_grounded_cache.py` validates and checksum-publishes
  the complete cache under an isolated dataset prefix. `remote_dataset/download_fleurs_vi_en.py` →
  `finetune/build_pairs.py` → `finetune/cache_codes.py` builds FLEURS inputs.
  `remote_dataset/download_covost2.py` materializes the pinned healthy FR→EN
  evaluator control.
- The grounded-v2 PhoMT rebuild resumed on an H100 from the SHA-verified,
  contiguous 90-shard Mac prefix (`shard_00000.pt` through `shard_00089.pt`).
  Four CUDA workers run from commit `45a1327`; the detached pipeline validates
  all 1,377 shards, builds grounded FLEURS caches, then publishes and verifies
  the isolated dataset `grounded-v2/` prefix. The Mac retains the ignored
  90-shard recovery copy until remote publication succeeds.
- `finetune/validate.py` is teacher-forced diagnostics only. `finetune/eval.py`
  evaluates correct and shuffled sources at fixed-seed text temperature 0.4,
  writing condition and consolidated artifacts. Promotion requires correct-source
  health plus calibrated BLEU/chrF gaps, then ranks by `(BLEU, chrF)`.
- `finetune/hf_sync.py` maintains two recovery pairs plus the best model under
  `full_run/` in the public `huybik/hibiki-zero-vi-full-sft` model repo;
  it also preserves run configuration and pilot membership metadata. `h100.sh`
  verifies a shared run identity, supervises sync, and protects the local resume
  point before training restarts.

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
preadaptation.

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
