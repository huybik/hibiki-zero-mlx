# Vietnamese SFT mechanics and pod handoff

The training stack uses the PyTorch `moshi` package and is separate from the MLX
inference runtime. It supports one path: full-model SFT from the upstream
Hibiki-Zero 3B base checkpoint on one H100.

## Durable handoff contract

A Vast pod is disposable. Keep the three durable resources distinct:

- GitHub `huybik/hibiki-zero-mlx`: exact training code. Commit and push every
  training change before relying on a remote recovery checkpoint.
- HF dataset `huybik/hibiki-zero-vi-full-sft`: immutable PhoMT/FLEURS caches.
- HF model `huybik/hibiki-zero-vi-full-sft`: published checkpoints. The current
  legacy, grounded full, grounded pilot, high-delay pilot, and contrastive pilot
  runs own only `full_run/`, `grounded_v2/`, `grounded_v2_pilot/`,
  `grounded_v2_pilot_high_delay/`, and
  `grounded_v2_pilot_high_delay_contrastive/` respectively. The Vietnamese-ASR
  diagnostic owns `grounded_v2_pilot_vi_asr_preadapt/`; phase-1 is preserved.

Never put a token, SSH endpoint, or pod-specific path in Git. A new agent should
first read `AGENTS.md`, `CONTEXT.md`, and the pod's `/etc/vast-agents-guide.md`.

## New H100 pod from zero

Rent one H100 with at least 79 GB VRAM, NVIDIA driver 580 or newer, 128 GB host
RAM, and 300 GB disk. Give the new agent the current SSH command, then run the
following from the pod.

### 1. Clone and install the pinned environment

```bash
cd /workspace
git clone https://github.com/huybik/hibiki-zero-mlx.git hibiki-zero
cd hibiki-zero
```

Before installing, inspect `full_run/` in the public checkpoint model repo. If
`run.json` exists, this is a recovery rather than a fresh run. Check out its
recorded commit first:

```bash
recovery_commit="$(curl -fsSL \
  https://huggingface.co/huybik/hibiki-zero-vi-full-sft/resolve/main/full_run/run.json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["commit"])')"
git checkout "$recovery_commit"
```

Skip that command only when `full_run/` is empty. The recorded commit must exist
on GitHub; this is why training changes must be pushed before launch. Now install:

```bash
export HF_HOME=/workspace/.hf_home
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./finetune/h100.sh setup
./.venv/bin/hf auth login
```

`setup` installs exactly Torch `2.13.0+cu132`, CUDA 13.2, Moshi 0.2.13, and the
pinned training dependencies. Do not reuse a `.venv` copied from another pod.

### 2. Restore the upstream base files

```bash
./.venv/bin/hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
```

Preflight verifies the exact SHA-256 of all four files.

### 3. Restore the published training caches

Only download the eight cache chunks and the FLEURS cache archive. The large
`pairs_w*.jsonl` source files are not needed for cached training.

```bash
cache_stage="$(mktemp -d /workspace/hibiki-cache.XXXXXX)"
./.venv/bin/hf download huybik/hibiki-zero-vi-full-sft \
  cache_chunk_0.tar.zst cache_chunk_1.tar.zst \
  cache_chunk_2.tar.zst cache_chunk_3.tar.zst \
  cache_chunk_4.tar.zst cache_chunk_5.tar.zst \
  cache_chunk_6.tar.zst cache_chunk_7.tar.zst \
  fleurs_cache.tar.zst \
  --repo-type dataset --local-dir "$cache_stage"

mkdir -p finetune/cache/phomt_stream
for archive in "$cache_stage"/cache_chunk_*.tar.zst; do
  tar --zstd --exclude='._*' --exclude='*/._*' \
    -xf "$archive" -C finetune/cache/phomt_stream
done
tar --zstd --exclude='._*' --exclude='*/._*' \
  -xf "$cache_stage/fleurs_cache.tar.zst" -C finetune
find "$cache_stage" -depth -delete
```

The exclusions remove AppleDouble metadata from the published archives.
Expected cache counts are 1,377 PhoMT shards, 46 FLEURS train shards, and five
FLEURS validation shards.

### 4. Restore evaluation audio

The cache archive includes the frozen pair manifests, but validation and test
audio must be materialized for free-running evaluation:

```bash
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split validation
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split test
```

Expected manifest rows are 128 for `val128`, 149 for validation, and 347 for
test. Do not rebuild the supplied manifests on a launch pod.

### 5. Qualify the pod and stop at the launch boundary

```bash
git status --short
./finetune/h100.sh preflight
./finetune/h100.sh smoke
```

`git status --short` must show no tracked changes. Preflight verifies the GPU,
driver, package pins, hashes, cache counts, manifests, audio, RAM, and free disk.
The smoke must train finite steps, save, run standalone paired evaluation,
resume exactly, and leave at least 2 GiB VRAM headroom. It writes a `SMOKE_OK`
marker tied to the current Git commit.

The default new-agent handoff stops here. Report the preflight profile, smoke
peak VRAM, Git commit, HF destination, and idle GPU. Do not launch full training
until explicitly requested.

## Fresh run versus recovery

Before launch, inspect the public model repo's `full_run/` directory:

- If `full_run/` is empty, start once with:

  ```bash
  export HF_HOME=/workspace/.hf_home
  export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
  ./finetune/h100.sh train
  ```

- If `full_run/` contains `run.json`, do not use `train`. Restore the newest
  complete model/trainer pair, `run.json`, and current best files by following
  [training_plan.md](training_plan.md). The recorded commit must have been
  checked out before `setup`; run preflight/smoke on the new pod, then use:

  ```bash
  ./finetune/h100.sh resume \
    finetune/runs/vi_base_full/trainer_stepNNNNNN.pt
  ```

`h100.sh` enforces the empty-prefix rule for a fresh run and the shared run
identity for recovery. It supervises checkpoint sync and stops training after
three consecutive sync failures. Recovery sync also preserves `run_config.json`,
the pilot's frozen `sample_manifest.jsonl`, and the optional
`source_derangement.json`, `source_asr.json`, or `source_asr_replay.json` under
`metadata/`.

## Cache format and rebuild path

Training consumes cached `shard_*.pt` files. Each sample is
`codes[1+n_q, T]`: English text, English target Mimi codes, then Vietnamese
source Mimi codes with a source-EOS frame. The published archives are the
launch path. Rebuild only when intentionally replacing the dataset:

```bash
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split train
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split validation
./.venv/bin/python finetune/build_pairs.py --splits train validation
./.venv/bin/python finetune/cache_codes.py \
  --pairs finetune/pairs/train.jsonl
./.venv/bin/python finetune/cache_codes.py \
  --pairs finetune/pairs/validation.jsonl \
  --out-dir finetune/cache/validation
```

`cache_phomt_stream.py` rebuilds the large PhoMT cache directly from parquet,
one source shard at a time.

Set `HIBIKI_RECIPE=grounded-v2` only for the experimental word-timed recipe.
Its cache builders use English Wav2Vec2 CTC word timing, pinned PhoMT input, and
a 0.5 alignment threshold. On an H100, `./finetune/h100.sh cache-grounded` is
the cache-build entrypoint and also builds explicit FLEURS train/validation
caches.

The spend-control run additionally requires `HIBIKI_PILOT=1`. That mode forces
isolated `*_grounded_v2_pilot` caches, smoke/run directories, and the
`grounded_v2_pilot/` Hugging Face prefix. It builds exactly 104 evenly sampled
PhoMT shards, trains exactly 50,000 selected rows for 1,000 steps with 100-step
warmup, evaluates at step 0 and every 250 steps, disables audio loss, and masks
target-audio teacher inputs. The ordered sampled membership and SHA-256 are
persisted and checked on resume. Full grounded-v2 keeps 1,000-step warmup and
rejects the old pilot limit environment variables.

After the ordinary pilot fails source dependence, add
`HIBIKI_HIGH_DELAY_PILOT=1` to run the otherwise identical pilot with a
deterministic uniform target delay in 75--100% of source duration. The launcher
forces separate `*_grounded_v2_pilot_high_delay` cache/run/smoke paths and the
`grounded_v2_pilot_high_delay/` recovery prefix. Cache metadata, preflight, and
the trainer run config verify delay ratios 0.75/1.0 and cache seed 1234. It also
requires the ordinary pilot's
`finetune/runs/vi_grounded_v2_pilot/sample_manifest.jsonl` at SHA-256
`52ef91a79dc09fb6c00a6f800bf087f2228b7c0842ecb2705ac873d3ef3a458f`.
Training reconstructs that exact ordered 50,000-entry cohort, including repeats,
then hard-gates it at 480 frames. It does not resample, sort, or shuffle it.
On a fresh pod, restore the pinned input before high-delay cache or preflight;
the ignored `.env` supplies `HF_TOKEN`:

```bash
set -a
source .env
set +a
manifest_restore="$(mktemp -d /workspace/hibiki-manifest.XXXXXX)"
./.venv/bin/hf download huybik/hibiki-zero-vi-full-sft \
  grounded_v2_pilot/metadata/sample_manifest.jsonl \
  --local-dir "$manifest_restore"
mkdir -p finetune/runs/vi_grounded_v2_pilot
cp "$manifest_restore/grounded_v2_pilot/metadata/sample_manifest.jsonl" \
  finetune/runs/vi_grounded_v2_pilot/sample_manifest.jsonl
find "$manifest_restore" -depth -delete
```

High-delay teacher-forced validation is independently length-sorted, never
shuffled, and retains every row with batch 1 under a separate 704-frame hard
cap; the observed maximum is 701.

After the high-delay pilot fails, add `HIBIKI_CONTRASTIVE_PILOT=1` for
preflight, smoke, train, and resume. Do not rebuild the cache: this mode reuses
the verified high-delay cache and exact membership while isolating smoke, run,
and recovery artifacts under `*_grounded_v2_pilot_high_delay_contrastive`.
It freezes a deterministic 256-row duration-block donor permutation with no
duplicate-ID donors, replaces only Vietnamese source codes, and trims or
silence-pads each donor to the target source duration while preserving source
EOS. The added English-content loss is
`relu(0.5 + correct_nll - shuffled_nll)` at weight 1. The smoke verifies the
mapping hash/permutation and finite contrastive loss, shuffled-minus-correct
NLL gap, and active-margin fraction.
Because each microbatch performs sequential correct and shuffled forwards, the
94 GB H100 recipe uses physical batch 4 / accumulation 4 while preserving
effective batch 16.

The completed 1,000-step contrastive pilot learned its teacher-forced objective
but failed free-running qualification. At step 1,000, shuffled-minus-correct
English-content NLL was 1.04, margin loss was 0.039, and 18.8% of rows remained
active. Paired generation still produced only 0.04 BLEU / 0.61 chrF gaps, 69
repeated-4gram failures, and mean length ratio 2.95. Do not extend or use this
checkpoint for full training; proceed to a Vietnamese acoustic-preadaptation
diagnostic.

For that diagnostic, unset the contrastive flag and add
`HIBIKI_ASR_PREADAPT=1` to preflight, smoke, train, and resume. Do not rebuild
the cache. The mode reuses the verified high-delay cache and exact frozen 50k
membership but owns isolated `*_grounded_v2_pilot_vi_asr_preadapt` smoke/run/HF
artifacts. In memory it removes English text and target audio, retains only the
Vietnamese source codebooks through source EOS, then emits the cached Vietnamese
transcript. Text before that point is supervised PAD, target-audio input is
masked, and audio loss remains zero. The ordered source-text policy and tokenizer
hash are frozen in `source_asr.json` and checked on resume.

This is a bounded full-sentence Vietnamese ASR preadaptation test, not a
reproduction of Kyutai's multilingual audio pretraining. Exact simulated cohort
lengths require 672 training and 640 validation frame caps; observed maxima are
668 and 627. The 94 GB H100 uses physical batch 4 / accumulation 4 and validation
batch 1. Paired free-running evaluation uses Vietnamese references, temperature
0.4, and a 24-second tail. Qualification requires normal health, source gaps of
at least 1.0 BLEU and 5.0 chrF, correct-source chrF at least 50, and WER at most
0.60. Even a passing ASR checkpoint only authorizes a separate warm-start
translation pilot; it does not authorize the full run.

The completed 1,000-step ASR pilot covered only 16,000 of 50,000 ordered sample
positions. At step 1,000 it passed output health and both source gaps (1.18 BLEU
and 7.64 chrF), demonstrating learnable Vietnamese acoustic routing, but correct
chrF was only 18.31 and WER was 0.678. Do not initialize translation from it.
The separate base-start one-epoch run consumed every frozen cohort position. At
step 3,125 it passed health and source dependence (6.65 BLEU / 15.43 chrF gaps)
but failed absolute ASR at chrF 26.72. Its recorded WER 0.639 is invalid because
the old word normalizer deleted accented Vietnamese letters; corrected
diacritic-insensitive WER is 0.775. The fixed tokenizer encodes raw Vietnamese
at 4.14 pieces/word, and 110/128 final hypotheses contain invalid-byte
replacement characters.

The corrected `HIBIKI_ASR_ASCII=1` run qualified at step 3,125: 127/128
nonempty, 128 EOS, no repetition failures, BLEU 27.85, chrF 53.26, WER 0.514,
and 27.74 BLEU / 34.48 chrF source gaps. Its promoted model SHA is
`d37d69103bff8f128b9b69fc9634a018d8ab5c5c58dbb0b5cc98ecf5a26f92ca`.

The isolated `HIBIKI_ASR_TRANSLATION_PILOT=1` test reconstructed the exact
ordinary-timing 50k cohort and started a fresh optimizer from that qualified
parent. It failed: at step 1,000, correct-source BLEU/chrF was 0.06/8.44,
source gaps were 0.01/-0.36, and 24 rows failed the repetition gate. No best
checkpoint was promoted. A plain ASR-to-translation switch is rejected.

The next bounded test is `HIBIKI_ASR_REPLAY_TRANSLATION_PILOT=1`. It keeps the
same parent, translation cohort, timing, masked target audio, zero audio loss,
batch 16, and 1,000-step schedule, but adds one deterministic four-row ASCII-ASR
batch per optimizer step at weight 1. Replay PAD has zero loss weight, so the
auxiliary objective preserves Vietnamese content routing after source EOS
without teaching translation frames to remain silent. Replay uses the same
ordered manifest, hard-gates its measured maximum at 434 frames, freezes its
policy in `source_asr_replay.json`, resumes both iterators exactly, and owns
`*_grounded_v2_pilot_vi_asr_replay` artifacts. A pass selects a text-translation
recipe candidate; it still does not authorize full training. This mode disables
Torch compilation because its batch-16 translation and batch-4 replay shapes
filled 94.5/95.8 GiB with compiled graph caches by production step 30 despite
the short smoke passing. The compile choice is persisted and checked on resume.

All selection evaluation is paired at fixed seed and text temperature 0.4. A
SHA-verified duration-matched derangement is reused for correct and shuffled
conditions, and evaluation restores training RNG afterward. Calibrate explicit
minimum BLEU/chrF source gaps from Vietnamese base, phase-1, and healthy French
controls before grounded preflight; see [validation_plan.md](validation_plan.md).
After all caches validate, `publish_grounded_cache.py` stores eight PhoMT chunks,
one grounded FLEURS archive, manifests, and checksums under the isolated dataset
`grounded-v2/` prefix. The legacy dataset-root archives remain unchanged.

## Trainer invariants

- Every model parameter is trainable; there is no adapter or freeze map.
- CUDA uses fp32 master weights and bf16 autocast.
- `--max-frames 280` bounds ordinary-run memory; the exact-membership high-delay
  pilot uses 480 for training and a separate 704 validation cap at batch 1.
  The source-ASR diagnostic uses 672/640 train/validation caps.
  Sixteen-frame buckets bound compiled CUDA shapes.
- H100 80 GB uses batch 8 with two accumulation steps; 94 GB uses batch 16.
  The high-delay pilot uses batch 8 / accumulation 2, while its contrastive
  variant and source-ASR diagnostic use batch 4 / accumulation 4 on the 94 GB
  H100. The ASR-replay pilot performs sequential batch-16 translation and
  batch-4 replay forwards with one optimizer update.
- Text content, PAD, and first-content losses are reduced independently before
  weighting, so PAD prevalence cannot change its aggregate gradient budget.
- Learning-rate, text-loss, and audio-loss schedules use `value@fraction` syntax.
- Checkpoints contain the exact full model. Loading rejects missing or extra keys.
- `--resume-checkpoint` is only for interruption recovery in the same run.

Teacher-forced validation is diagnostic. `--eval-every` performs paired
free-running evaluation and writes condition plus consolidated artifacts.
The exact recipe is in [training_plan.md](training_plan.md); final selection
follows [validation_plan.md](validation_plan.md).
