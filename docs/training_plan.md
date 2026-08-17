# Vietnamese base-start training recipe

Train the full Hibiki-Zero 3B model from the upstream base weight on the frozen
PhoMT and FLEURS caches. Do not initialize from another finetuned checkpoint.
Use `--resume-checkpoint` only to recover this same run after interruption.

## Preflight

Stage the repository, weights, caches, pair manifests, and FLEURS validation/test
audio on a pod with one visible H100, NVIDIA driver 580 or newer, at least 128 GB
host RAM, and 300 GB disk. CUDA 13.x minor-version compatibility supports the
pinned CUDA 13.2 wheel on the 580 driver series. Then run:

```bash
./finetune/h100.sh setup
./finetune/h100.sh preflight
./finetune/h100.sh smoke
```

The launcher pins Torch 2.13.0/CUDA 13.2 and Moshi 0.2.13, verifies artifact
hashes and evaluation audio, uses 16-frame CUDA buckets, and chooses batch 16 on
a 94 GB H100 NVL or batch 8 with two accumulation steps on an 80 GB H100. The
smoke must complete finite training, save, standalone free-running evaluation,
exact resume, and a VRAM headroom check before it writes
`finetune/runs/h100_smoke/SMOKE_OK`.

## Run

Authenticate the pod and select the existing public checkpoint model repo:

```bash
./.venv/bin/hf auth login
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
```

```bash
./finetune/h100.sh train
```

Recover an interrupted run in its original directory with:

```bash
./finetune/h100.sh resume finetune/runs/vi_base_full/trainer_stepNNNNNN.pt
```

If the pod volume was lost, inspect `full_run/checkpoints/` in the repo and copy the
newest matching model/trainer filenames into `finetune/runs/vi_base_full/`
before running the same resume command. Restore the run identity and current
best files too; resume verifies them before it uploads the newest local pair:

```bash
recovery_dir="$(mktemp -d)"
./.venv/bin/hf download "$HIBIKI_HF_REPO" \
  full_run/run.json \
  full_run/metadata/run_config.json \
  full_run/checkpoints/model_stepNNNNNN.safetensors \
  full_run/checkpoints/trainer_stepNNNNNN.pt \
  full_run/best/best_stepBBBBBB.safetensors \
  full_run/best/best_stepBBBBBB.json \
  --local-dir "$recovery_dir"
cp "$recovery_dir/full_run/run.json" finetune/runs/vi_base_full/run_id.json
cp "$recovery_dir/full_run/metadata/run_config.json" \
  finetune/runs/vi_base_full/run_config.json
cp "$recovery_dir/full_run/checkpoints/model_stepNNNNNN.safetensors" \
  finetune/runs/vi_base_full/
cp "$recovery_dir/full_run/checkpoints/trainer_stepNNNNNN.pt" \
  finetune/runs/vi_base_full/
cp "$recovery_dir/full_run/best/best_stepBBBBBB.safetensors" \
  finetune/runs/vi_base_full/
cp "$recovery_dir/full_run/best/best_stepBBBBBB.json" \
  finetune/runs/vi_base_full/best.json
```

`h100.sh train` expands to the recipe below using the hardware-selected batch
size and accumulation factor.

```text
epochs=2, max_frames=280, effective_batch=16
lr=1e-4@0,3e-5@0.5, warmup_steps=500
text_weight=5@0,2@0.6, text_pad_loss_weight=0.5
val_every=2000, eval_every=9000, save_every=3000, keep_checkpoints=2
```

Checkpoint rotation keeps the newest two complete model/trainer pairs. Before a
new save it removes the oldest pair while retaining one valid recovery point,
so checkpoint storage peaks at two pairs (about 70 GiB) rather than three. The
trainer file is published only after its model and acts as the remote pair's
commit marker; interrupted writes are ignored and cleaned at the next save. The
setup also disables pip's package cache.

The launcher runs `hf_sync.py` beside training. It uploads 9,000-step evaluation
checkpoints, the best model, and the newest pair when training exits under
`full_run/` in the public checkpoint repo. It preserves the existing phase-1
artifacts and keeps two recovery pairs plus one best model in the current tree.
Because model repos retain commit history, pruning old files does not immediately
reclaim their underlying storage. Hard-link
staging prevents local rotation from invalidating an active upload without
copying checkpoint bytes, though a slow or interrupted upload can temporarily
pin one additional pair on disk. Training waits for a verified final sync before
the launcher exits. If sync fails three consecutive times, the launcher stops
training instead of continuing without disaster recovery.

## Stop conditions

Stop immediately on non-finite loss, checkpoint mismatch, corrupt output, or a
logged LR that differs from `run_config.json`. Teacher-forced loss cannot select
a model. Promotion always requires the paired health and calibrated source-gap
gates in [validation_plan.md](validation_plan.md). Qualified checkpoints rank by
correct-source BLEU, with chrF as the secondary key. The rolling repo preserves
the complete `best.json` metadata with the selected model.

## Experimental grounded-v2 recipe

The legacy recipe remains the default. `grounded-v2` full training is isolated
behind `HIBIKI_RECIPE`. Its pilot is a separate explicit mode; for pilot setup:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh setup
```

Do not set `HIBIKI_HF_PREFIX` in pilot mode; the launcher forces
`grounded_v2_pilot`. It also forces the pilot cache directories
`phomt_grounded_v2_pilot`, `train_grounded_v2_pilot`, and
`validation_grounded_v2_pilot`, smoke directory
`h100_smoke_grounded_v2_pilot`, and run directory `vi_grounded_v2_pilot`.

Build CTC word-timed caches; grounded mode refuses to mix them with legacy
contiguous-text shards. PhoMT is pinned to one dataset revision. All pairs are
used: earlier pairs are gender-consistent but independently voiced, while source
indexes at or above 345,600 have additional cross-lingual timbre matching. The
cache records that distinction as metadata; neither group establishes speaker
identity. Build all 1,377 PhoMT shards plus the FLEURS train/validation caches
on the H100 with:

```bash
export HIBIKI_RECIPE=grounded-v2
unset HIBIKI_PILOT
./finetune/h100.sh cache-grounded
```

The launcher defaults to four supervised workers and fails the whole cache job
if any worker exits nonzero. Its `h100` profile batches the forced-alignment
Viterbi pass, bounds Wav2Vec2 and Mimi batches by padded audio samples, and logs
per-worker rows/second. It keeps Mimi in fp32. Set `HIBIKI_CACHE_WORKERS` only
after measuring GPU and host headroom. A cold full build cannot finish within
two hours unless the pod sustains roughly 80 MB/s from Hugging Face. Treat an
aggregate rate above 100 attempted rows/s after downloads warm as the two-hour
launch gate.

Both builders reject CTC alignments below mean posterior 0.5 and store the
alignment score plus normalized spoken transcript. Inspect the rejected rows
and the lowest-scoring accepted tail before launch. The threshold was checked
on 48 decoded PhoMT English clips spanning both generation phases; after using
the Wav2Vec2 model's correct no-attention-mask batching, scores ranged from
0.814 to 0.987.

The full recipe uses one epoch, 95% PhoMT / 5% FLEURS sampling, effective batch
16, AdamW `(beta1=0.9, beta2=0.95, weight_decay=0.1)`, 1,000 warmup steps, and
cosine LR `1e-5 -> 1e-6`. The pilot keeps the optimizer recipe but uses 100
warmup steps and exactly 50,000 samples / 1,000 optimizer steps. It sets audio
loss to zero and masks only the target-audio input codebooks, leaving Vietnamese
source codes, English text history, and English text targets intact. Its ordered
`cache_index,id` membership (including repeats) is frozen in
`sample_manifest.jsonl`; resume requires identical bytes and SHA-256.

Build and run the pilot with:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh cache-grounded
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

Pilot cache construction always selects exactly 104 evenly spaced PhoMT shards
and builds explicit pilot FLEURS caches. Preflight requires exactly those cache
counts and at least 47,500 accepted PhoMT rows. The evaluator runs at steps
0/250/500/1,000 using fixed-seed temperature 0.4 correct and duration-matched
shuffled conditions. Pilot and full recovery namespaces can never warm-start
one another. Full grounded-v2 rejects the obsolete `HIBIKI_MAX_SAMPLES`,
`HIBIKI_MAX_STEPS`, and `HIBIKI_CACHE_SAMPLE_SHARDS` controls.

The current early target timing is a diagnostic policy, not established
causality: many rows emit English supervision before enough Vietnamese evidence
exists. Keep the first corrected pilot on the current cache. If source dependence
remains near zero, repeat the same contract with 75--100% delay by additionally
exporting `HIBIKI_HIGH_DELAY_PILOT=1` for cache, preflight, smoke, train, and
resume. This mode samples deterministic uniform delay ratios in `[0.75, 1.0]`
with seed 1234 and owns only `*_grounded_v2_pilot_high_delay` paths plus the
`grounded_v2_pilot_high_delay/` recovery prefix. Do not reuse ordinary pilot
caches. It requires the ordinary pilot manifest at SHA-256
`52ef91a79dc09fb6c00a6f800bf087f2228b7c0842ecb2705ac873d3ef3a458f` and
reconstructs its exact ordered 50,000 entries: 47,500 PhoMT plus 2,500 FLEURS,
48,892 unique. Repeats and order are preserved; cache-weight sampling,
max-sample selection, sorting, and production shuffling are forbidden. The
post-reconstruction frame gate is 480 because the rebuilt cohort's high-delay
maximum is 479. Use physical batch 8 with two-step accumulation; the smoke
feeds the longest rows first without changing the persisted manifest. The
independent teacher-forced validation set is deterministically length-sorted and retains
all rows at batch 1 under a 704-frame hard cap; its observed maximum is 701.
Run configuration records the delay bounds, cache seed, membership SHA, and
frame caps. This exact high-delay pilot failed all health and source-dependence
gates through step 1,000, so delay alone is rejected. The next isolated pilot
adds a duration-matched shuffled-source margin loss. It reuses the verified
high-delay cache; do not run `cache-grounded` again:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
export HIBIKI_HIGH_DELAY_PILOT=1
export HIBIKI_CONTRASTIVE_PILOT=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

The contrastive run owns only
`*_grounded_v2_pilot_high_delay_contrastive` smoke/run paths and the matching
HF prefix. A frozen duration-block permutation assigns a different-ID,
duration-matched Vietnamese donor to every target. Collation preserves target
timing and source EOS, so duration cannot reveal which source is shuffled. The
ordinary correct-source CE remains unchanged; a sequential shuffled forward
adds weight-1 `relu(0.5 + correct_nll - shuffled_nll)` over per-row English
content NLL. Persist and verify the mapping SHA, contrastive loss,
shuffled-minus-correct NLL gap, and active-margin fraction. Consider Vietnamese
acoustic preadaptation only if this pilot also fails. Sequential positive and
negative forwards use physical batch 4 / accumulation 4 on the 94 GB H100 while
preserving effective batch 16. A high-delay cache remains a curriculum, never
the final timing policy, because sentence-delay supervision can leave
persistent lag.

The exact contrastive pilot failed all promotion gates. Its teacher-forced NLL
gap reached 1.04 by step 1,000, proving that the ranking objective operated, but
paired free-running gaps were only 0.04 BLEU / 0.61 chrF and generation remained
overlong and repetitive. Reject text-only contrastive SFT as the full recipe.
The next diagnostic is Vietnamese acoustic preadaptation from the upstream base,
followed by a fresh isolated translation pilot; do not warm-start the full run
from this checkpoint.

Run the diagnostic on the verified high-delay cache and exact frozen membership;
do not run `cache-grounded` again:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
export HIBIKI_HIGH_DELAY_PILOT=1
unset HIBIKI_CONTRASTIVE_PILOT
export HIBIKI_ASR_PREADAPT=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

This mode derives full-sentence Vietnamese ASR examples in memory: Vietnamese
source codes remain through source EOS, the Vietnamese transcript follows, and
English text plus target audio are absent. It owns only
`*_grounded_v2_pilot_vi_asr_preadapt` artifacts and freezes the tokenizer plus
ordered source-text policy in `source_asr.json`. Use physical batch 4 /
accumulation 4 with train/validation caps 672/640 on the 94 GB H100. Paired
Vietnamese evaluation uses a 24-second tail and requires normal health, source
gaps at least 1.0 BLEU / 5.0 chrF, correct-source chrF at least 50, and WER at
most 0.60. This diagnoses Vietnamese acoustic learnability; it is not a complete
audio-pretraining stage. A pass authorizes only a fresh isolated translation
pilot initialized from the selected ASR checkpoint.

The 1,000-step run consumed only 16,000/50,000 ordered positions. It passed
health and source dependence at step 1,000 (1.18 BLEU / 7.64 chrF gaps) but
failed absolute ASR with chrF 18.31 and WER 0.678. This proves that the backbone
can learn Vietnamese acoustic routing, but the checkpoint is not qualified for
translation initialization. Run a separate base-start 3,125-step diagnostic to
cover one exact cohort epoch; do not extend the completed run under a changed LR
horizon or reuse its recovery namespace.

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
export HIBIKI_HIGH_DELAY_PILOT=1
unset HIBIKI_CONTRASTIVE_PILOT
export HIBIKI_ASR_PREADAPT=1
export HIBIKI_ASR_ONE_EPOCH=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

This run completed but failed absolute ASR: final chrF was 26.72 despite healthy
generation and 6.65 BLEU / 15.43 chrF source gaps. Corrected
diacritic-insensitive WER is 0.775; the recorded 0.639 used the obsolete
ASCII-only word normalizer. Raw Vietnamese requires 4.14 tokenizer pieces/word
and produced replacement characters in 110/128 hypotheses.

Repeat the same base-start one-epoch contract with tokenizer-compatible ASCII
Vietnamese (1.87 pieces/word) in a fresh namespace:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
export HIBIKI_HIGH_DELAY_PILOT=1
unset HIBIKI_CONTRASTIVE_PILOT
export HIBIKI_ASR_PREADAPT=1
export HIBIKI_ASR_ONE_EPOCH=1
export HIBIKI_ASR_ASCII=1
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

This mode qualified at step 3,125 with BLEU/chrF 27.85/53.26, WER 0.514, and
27.74/34.48 source gaps. Its promoted model SHA is
`d37d69103bff8f128b9b69fc9634a018d8ab5c5c58dbb0b5cc98ecf5a26f92ca`.

Run the next isolated translation pilot from that exact parent:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_PILOT=1
export HIBIKI_ASR_TRANSLATION_PILOT=1
unset HIBIKI_HIGH_DELAY_PILOT HIBIKI_CONTRASTIVE_PILOT
unset HIBIKI_ASR_PREADAPT HIBIKI_ASR_ONE_EPOCH HIBIKI_ASR_ASCII
unset HIBIKI_HF_PREFIX
export HIBIKI_MIN_SOURCE_BLEU_GAP=1.0
export HIBIKI_MIN_SOURCE_CHRF_GAP=5.0
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

This mode owns `*_grounded_v2_pilot_vi_asr_warmstart`, reconstructs the exact
ordinary-timing 50k membership, pins the qualified parent SHA, starts a fresh
optimizer, and uses physical batch 16 for 1,000 masked-audio translation steps.
Evaluation remains paired at 0/250/500/750/1,000 with unchanged promotion gates.
