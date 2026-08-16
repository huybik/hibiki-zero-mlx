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
  run owns only `full_run/`; existing phase-1 artifacts are preserved.

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
The smoke must train finite steps, save, run standalone greedy evaluation,
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
three consecutive sync failures.

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
Its cache builders run an English Wav2Vec2 CTC forced alignment and place each
SentencePiece group at the corresponding target-speech frames, with text EOS at
the target-audio end. PhoMT input is pinned; its publisher-recorded source ranges
label the later timbre-matched subset without excluding the earlier
gender-consistent pairs, and low-score CTC rows are rejected. The v2 launcher
reads only `*_grounded_v2` caches and uses the independent `grounded_v2/`
Hugging Face recovery prefix; see
[training_plan.md](training_plan.md) for exact commands and hyperparameters.

## Trainer invariants

- Every model parameter is trainable; there is no adapter or freeze map.
- CUDA uses fp32 master weights and bf16 autocast.
- `--max-frames 280` bounds memory; length-sorted 16-frame buckets minimize padding
  while keeping compiled CUDA shapes stable.
- H100 80 GB uses batch 8 with two accumulation steps; 94 GB uses batch 16.
- Text prefix PAD has weight 0.5; content and EOS remain weight 1.
- Learning-rate, text-loss, and audio-loss schedules use `value@fraction` syntax.
- Checkpoints contain the exact full model. Loading rejects missing or extra keys.
- `--resume-checkpoint` is only for interruption recovery in the same run.

Teacher-forced validation is diagnostic. `--eval-every` performs deterministic
free-running evaluation and writes predictions plus generation-health metrics.
The exact recipe is in [training_plan.md](training_plan.md); final selection
follows [validation_plan.md](validation_plan.md).
