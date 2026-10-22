# Full-model SFT on an H100 pod

`training/h100.sh` sets up a pinned environment, checks the pod, and runs
`training.finetune.train` with Hugging Face disaster recovery. Training starts from
the upstream Hibiki-Zero weight and updates every parameter on grounded cache
directories built by `training.prep.cache_codes`.

## Pod requirements

Use one H100 with at least 90 GiB VRAM, NVIDIA driver 570 or newer, at least
110 GiB host RAM, and at least 190 GiB free disk at preflight. The launcher also
requires Bash 5.1+ for training supervision. Never commit an HF token, SSH
endpoint, or pod-specific path.

## Set up a pod

```bash
cd /workspace
git clone https://github.com/huybik/hibiki-zero-mlx.git hibiki-zero
cd hibiki-zero
export HF_HOME=/workspace/.hf_home
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./training/h100.sh setup
./.venv/bin/hf auth login
./.venv/bin/hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
./training/h100.sh preflight
```

`setup` creates a new `.venv` with Torch `2.8.0+cu128`, Moshi `0.2.13` and the
pinned dependencies, and refuses to reuse an existing environment. `preflight`
checks the GPU, driver, package pins, host RAM, free disk, and the SHA-256 of the
four upstream artifacts, and requires a clean tracked worktree.

Put the train and validation caches under `data/cache/` (built with
[training/prep](../training/prep/README.md)).

## Train

```bash
./training/h100.sh train RUN_NAME \
  --cache-dir data/cache/TRAIN_A data/cache/TRAIN_B \
  --val-cache-dir data/cache/VALIDATION
```

Arguments after `RUN_NAME` go to `training.finetune.train` (`--help` lists them).
The defaults are batch 16, 280-frame rows, 5 epochs, constant LR `1e-6`, fused AdamW
`(0.9, 0.95)` with weight decay `0.1`, and validation and checkpoints every 9,000
steps.

`train` requires an empty `runs/RUN_NAME/`. It needs a public `HIBIKI_HF_REPO` whose
`RUN_NAME/` prefix is empty, and uploads `run.json` with the run identity and commit.
It then trains while `training.finetune.hf_sync` keeps the newest two complete recovery
pairs, the promoted best model, the run config, and the final logs under that prefix.
Training stops if sync fails. A start that fails before training (for example a
missing cache directory) has already claimed `runs/RUN_NAME/` and the remote
`RUN_NAME/run.json`: delete both, or choose a new name.

## Recover a run on a replacement pod

Check out the commit recorded in the remote `RUN_NAME/run.json` before setup, then set
up the pod as above and restore the newest complete model/trainer pair:

```bash
run=RUN_NAME
git checkout "$(curl -fsSL \
  "https://huggingface.co/huybik/hibiki-zero-vi-full-sft/resolve/main/$run/run.json" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["commit"])')"
```

```bash
run=RUN_NAME
restore_dir="$(mktemp -d /workspace/hibiki-recovery.XXXXXX)"
./.venv/bin/hf download "$HIBIKI_HF_REPO" \
  "$run/run.json" \
  "$run/metadata/run_config.json" \
  "$run/checkpoints/model_stepNNNNNN.safetensors" \
  "$run/checkpoints/trainer_stepNNNNNN.pt" \
  "$run/best/best_stepBBBBBB.safetensors" \
  "$run/best/best_stepBBBBBB.json" \
  --local-dir "$restore_dir"

run_dir="runs/$run"
mkdir -p "$run_dir"
cp "$restore_dir/$run/run.json" "$run_dir/run_id.json"
cp "$restore_dir/$run/metadata/run_config.json" "$run_dir/run_config.json"
cp "$restore_dir/$run/checkpoints/model_stepNNNNNN.safetensors" "$run_dir/"
cp "$restore_dir/$run/checkpoints/trainer_stepNNNNNN.pt" "$run_dir/"
cp "$restore_dir/$run/best/best_stepBBBBBB.safetensors" "$run_dir/"
cp "$restore_dir/$run/best/best_stepBBBBBB.json" "$run_dir/best.json"
find "$restore_dir" -depth -delete

./training/h100.sh resume "$run_dir/trainer_stepNNNNNN.pt" \
  --cache-dir data/cache/TRAIN_A data/cache/TRAIN_B \
  --val-cache-dir data/cache/VALIDATION
```

Pass the run's original training arguments again. Resume is interruption recovery for
the same run only:

- it needs the newest complete pair, an unchanged `run_config.json`, and the same run
  identity and commit;
- from the first validation step on, it also needs the promoted best model and marker;
- it cannot extend the stop step or start a different run.

Runs recorded before the `inference/`/`training/` split, such as
`grounded_v2_full_direct_voice_5epoch`, follow their own commit's `docs/finetune.md`.

## Evaluate a checkpoint

```bash
./.venv/bin/python -m training.finetune.validate \
  --cache-dir data/cache/VALIDATION \
  --checkpoint runs/RUN_NAME/best_stepNNNNNN.safetensors
./.venv/bin/python -m training.finetune.eval \
  --pairs PAIRS.jsonl \
  --checkpoint runs/RUN_NAME/best_stepNNNNNN.safetensors \
  --batch-size 8 --stop-on-eos --text-only --seed 42 \
  --out-dir runs/RUN_NAME/eval_stepNNNNNN
```

`validate` reports teacher-forced losses on a cache. `eval` runs correct-source
free-running generation on rows with `id`, `vi_audio` and `text_en` columns and
writes predictions, BLEU/chrF/WER, and a generation-health verdict.
