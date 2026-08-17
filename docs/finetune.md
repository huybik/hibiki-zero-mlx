# Direct Vietnamese SFT pod handoff

The only supported training path is one full-data epoch of direct
voice-preserving Vietnamese-to-English simultaneous translation. It starts from
the upstream Hibiki-Zero model, not an ASR or translation checkpoint.

## Frozen contract

- Keep the grounded-v2 cache streams unchanged: English target Mimi audio,
  CTC-timed English text ending in tokenizer EOS, and Vietnamese source Mimi
  audio ending in explicit codec-card EOS.
- Apply no post-source transform. English target audio remains visible to the
  model as teacher forcing and its audio CE weight is `1.0`.
- Train all parameters with physical batch 16, accumulation 1, raw train and
  validation `max_frames=280`, fixed LR `1e-6`, and fused AdamW betas
  `(0.9, 0.95)` with weight decay `0.1`.
- Train exactly 719,120 frozen rows for one epoch / 44,945 steps. Validation is
  138 eligible rows, batch 4, observed maximum 277 frames, and `shuffle=False`.
- Recovery artifacts live only under `grounded_v2_full_direct_voice`.

There is no pilot, ASR warm-start or replay, post-source curriculum,
contrastive or anti-repetition loss, shuffled validation, or multi-epoch
schedule.

## Pod requirements

Use one H100 with at least 90 GiB VRAM, NVIDIA driver 570 or newer, at least
110 GiB host RAM, and at least 190 GiB free disk at preflight. The launcher also
requires Bash 5.1+ for training supervision. Never commit an HF token, SSH
endpoint, or pod-specific path.

## New pod from zero

### 1. Clone and select the run commit

```bash
cd /workspace
git clone https://github.com/huybik/hibiki-zero-mlx.git hibiki-zero
cd hibiki-zero
```

Inspect the public recovery prefix before doing anything else. If `run.json`
exists, this is a recovery: check out its recorded commit before setup.

```bash
recovery_commit="$(curl -fsSL \
  https://huggingface.co/huybik/hibiki-zero-vi-full-sft/resolve/main/grounded_v2_full_direct_voice/run.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["commit"])')"
git checkout "$recovery_commit"
```

Skip that command only when the prefix is empty and a genuinely fresh run is
intended. Training refuses dirty tracked files, so all intended code and docs
must already be committed.

### 2. Install the pinned environment

```bash
export HF_HOME=/workspace/.hf_home
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./finetune/h100.sh setup
./.venv/bin/hf auth login
```

`setup` creates a new `.venv` and installs Torch `2.8.0+cu128`, CUDA 12.8,
Moshi `0.2.13`, and the pinned dependencies. It refuses to reuse an existing
environment.

### 3. Restore upstream Hibiki-Zero

```bash
./.venv/bin/hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
```

These four upstream artifacts are the only initialization inputs. Preflight
checks their exact SHA-256 values.

### 4. Restore grounded-v2 caches

The cache files are under the dataset repository's `grounded-v2/` prefix.
Preserve that prefix while downloading, then extract into the exact directories
used by the launcher:

```bash
cache_stage="$(mktemp -d /workspace/hibiki-cache.XXXXXX)"
cache_files=(
  grounded-v2/cache_chunk_{0..7}.tar.zst
  grounded-v2/fleurs_cache.tar.zst
)
./.venv/bin/hf download huybik/hibiki-zero-vi-full-sft \
  "${cache_files[@]}" \
  --repo-type dataset --local-dir "$cache_stage"

mkdir -p finetune/cache/phomt_grounded_v2
for archive in "$cache_stage"/grounded-v2/cache_chunk_*.tar.zst; do
  tar --zstd --exclude='._*' --exclude='*/._*' \
    -xf "$archive" -C finetune/cache/phomt_grounded_v2
done
tar --zstd --exclude='._*' --exclude='*/._*' \
  -xf "$cache_stage/grounded-v2/fleurs_cache.tar.zst" -C finetune
find "$cache_stage" -depth -delete
```

Expected directories and shard counts are:

- `finetune/cache/phomt_grounded_v2`: 1,377 shards;
- `finetune/cache/train_grounded_v2`: 46 shards;
- `finetune/cache/validation_grounded_v2`: 5 shards.

The large source pair manifests are not needed for cached training.

### 5. Preflight and smoke

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
git status --short
./finetune/h100.sh preflight
./finetune/h100.sh smoke
```

`git status --short` must show no tracked changes. Preflight checks hardware,
package pins, artifact hashes, cache receipts, and freezes the exact 719,120-row
manifest. Smoke trains the longest raw B16 rows, reaches the longest raw
non-shuffled validation row, verifies active target-audio CE, saves and resumes,
and requires at least 2 GiB VRAM headroom. Its `SMOKE_OK` marker is tied to the
current Git commit.

## Start a fresh full run

Use this only when the remote `grounded_v2_full_direct_voice/` prefix is empty:

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./finetune/h100.sh train
```

The launcher creates
`finetune/runs/vi_grounded_v2_full_direct_voice`, uploads `run.json`, trains one
epoch, validates every 1,000 steps, saves every 3,000 steps plus step 44,945,
and keeps two complete recovery pairs locally and remotely.

## Recover the same run on a replacement pod

After checking out the commit from remote `run.json`, completing setup, weights,
caches, preflight, and smoke, restore the newest complete model/trainer pair and
its run metadata. Replace `NNNNNN` with the newest step present for both files.

```bash
restore_dir="$(mktemp -d /workspace/hibiki-recovery.XXXXXX)"
prefix=grounded_v2_full_direct_voice
./.venv/bin/hf download "$HIBIKI_HF_REPO" \
  "$prefix/run.json" \
  "$prefix/metadata/run_config.json" \
  "$prefix/metadata/sample_manifest.jsonl" \
  "$prefix/metadata/full_data_receipt.json" \
  "$prefix/checkpoints/model_stepNNNNNN.safetensors" \
  "$prefix/checkpoints/trainer_stepNNNNNN.pt" \
  --local-dir "$restore_dir"

run_dir=finetune/runs/vi_grounded_v2_full_direct_voice
mkdir -p "$run_dir"
cp "$restore_dir/$prefix/run.json" "$run_dir/run_id.json"
cp "$restore_dir/$prefix/metadata/run_config.json" "$run_dir/run_config.json"
cp "$restore_dir/$prefix/metadata/sample_manifest.jsonl" "$run_dir/sample_manifest.jsonl"
cp "$restore_dir/$prefix/metadata/full_data_receipt.json" "$run_dir/full_data_receipt.json"
cp "$restore_dir/$prefix/checkpoints/model_stepNNNNNN.safetensors" "$run_dir/"
cp "$restore_dir/$prefix/checkpoints/trainer_stepNNNNNN.pt" "$run_dir/"
find "$restore_dir" -depth -delete

./finetune/h100.sh resume \
  "$run_dir/trainer_stepNNNNNN.pt"
```

Resume is only interruption recovery for this exact run. It requires the newest
complete pair, identical optimizer/receipt/manifest/config, the original run
identity, and the commit recorded by `run.json`. It cannot extend the one-epoch
stop or initialize a different run.

## Optional correct-source evaluation

Free-running evaluation is not part of the training loop and never shuffles the
source. Materialize validation audio only when this check is wanted:

```bash
./.venv/bin/python remote_dataset/download_fleurs_vi_en.py --split validation
./.venv/bin/python finetune/eval.py \
  --device cuda --dtype bfloat16 \
  --checkpoint finetune/runs/vi_grounded_v2_full_direct_voice/model_stepNNNNNN.safetensors \
  --pairs finetune/pairs/val128.jsonl \
  --limit 128 --batch-size 8 --text-temp 0.4 \
  --stop-on-eos --text-only --seed 42 \
  --out-dir finetune/runs/vi_grounded_v2_full_direct_voice/eval_stepNNNNNN
```

See [training_plan.md](training_plan.md) for the receipt and
[validation_plan.md](validation_plan.md) for monitoring.
