# Full-model SFT on an H100 pod

`training/h100.sh` sets up a pinned environment, checks the pod, and runs
`training.finetune.train` with Hugging Face disaster recovery. Training starts from
the upstream Hibiki-Zero weight and updates every parameter on grounded cache
directories built by `training.prep.cache_codes`. A run survives preemption: running
`h100.sh run RUN_NAME` again continues it from its newest local checkpoint, so an
interruptible pod reruns it on boot.

## Pod requirements

- One Hopper GPU with at least 80 GB: an H100 SXM (80 GB), H100 NVL (94 GB) or H200 / H200 NVL
  (141 GB). Training peaks at 75.1 GiB reserved or less, and validation should stay
  below that ([sft-speed](agent-memory/sft-speed.md#measurements)).
- The GPU at its stock power limit (`nvidia-smi -q -d POWER`): an H100 SXM its host capped at
  500 W instead of 700 W trained 1.8x slower than one at 700 W, and some providers cap every
  SXM they rent out. One epoch takes ~10 h on an H200 NVL or a 700 W H100 SXM and ~14 h on an
  H100 NVL ([sft-speed](agent-memory/sft-speed.md#by-gpu)).
- NVIDIA driver 580 or newer (CUDA 13), at least 80 GiB of RAM in the container, and Bash 5.1+.
- 350 GB of disk: the environment, weights, caches, their 70 GB of packs and the evaluation sets
  took 157 GiB on a pod, and the run needs 130 GiB: free, plus what a continued run already
  holds (`run` checks it).
- Interruptible pods work. Their disk survives a preemption, and the boot hook below continues
  the run; checkpoints every 1,000 steps (5–6.5 min) bound the work a preemption loses.

Never commit an HF token, SSH endpoint, or pod-specific path.

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

`setup` creates a new `.venv` with Torch `2.15.0+cu134`, Moshi `0.2.13` and the
pinned dependencies, and refuses to reuse an existing environment. `preflight`
checks the GPU, driver, package pins, RAM (the container's limit, not the host's), free disk,
and the SHA-256 of the four upstream artifacts, and requires a clean tracked worktree.

Restore the training caches from the cache repo (built with
[training/prep](../training/prep/README.md)): the `en_speech` cache with its validation split.
The PhoMT cache (`--include "phomt-grounded-v2-5s/shard_*.pt"`) and `cv-anquach`
(`--include "en-vi-grounded-v2-cv-anquach/train/shard_*.pt"`, 16 kHz English targets) train as
more `--cache-dir`s in later ablations. `--include` takes one pattern each; given two, `hf`
reads the second as a file name and ignores the include. Anonymous downloads hit HF's rate
limit (429) partway; logged in they do not, and `hf download` resumes where it stopped.

```bash
./.venv/bin/hf download huybik/hibiki-zero-vi-full-sft --repo-type dataset --local-dir data/cache \
  --include "en-vi-grounded-v2/*/shard_*.pt"
./.venv/bin/python -m training.finetune.streams \
  data/cache/en-vi-grounded-v2/train data/cache/en-vi-grounded-v2/validation
./.venv/bin/python -m training.prep.eval_sets
```

The second command packs each shard into a flat memory-mapped array beside it
(`<cache-dir>/.packed/`, 70 GB, ~30 s): every later start loads the 3.44M rows in ~9 s instead
of unpickling them for 7.5 minutes, and keeps them out of RAM. Training builds missing packs
itself, so this only moves that work before `run` checks the free disk. The third builds the
real-Vietnamese evaluation sets; training translates FLEURS dev at every validation to choose
its checkpoint, and refuses to start without it
([sft-training](agent-memory/sft-training.md#evaluation-sets)).

## Run

```bash
./training/h100.sh run RUN_NAME \
  --cache-dir data/cache/en-vi-grounded-v2/train \
  --val-cache-dir data/cache/en-vi-grounded-v2/validation
```

The first stream run trains on the `en_speech` cache alone; the PhoMT cache and `cv-anquach` come
back later as more `--cache-dir` ablations ([plan](vi_en_parity_plan.md#1-data)). Arguments after
`RUN_NAME` go to `training.finetune.train` (`--help` lists them). The defaults are 30–120 s
streams in batches of at most 4,608 frames, one epoch (131,553 steps), peak LR `1e-6` with 5%
warmup and cosine decay, audio codebook 0 weighted 100 and text PADs 0.5, fused AdamW
`(0.9, 0.95)` with weight decay `0.1`, checkpoints every 1,000 steps and validation every 9,000
([sft-training](agent-memory/sft-training.md#trainer)). Each validation logs teacher-forced
losses, translates FLEURS dev free-running, and promotes the checkpoint that passes the
generation health gates with the highest chrF
([sft-training](agent-memory/sft-training.md#checkpoints-and-recovery)).

A new run needs an empty `runs/RUN_NAME/` and a public `HIBIKI_HF_REPO` whose `RUN_NAME/`
prefix is empty. It records the run identity and commit in `runs/RUN_NAME/run_id.json` (uploaded
as `RUN_NAME/run.json`) and its arguments in `train_args.json`. It then trains while
`training.finetune.hf_sync` uploads, under that prefix, the pairs of the validation steps (the
newest two), the promoted best model, the run config and arguments, and the logs once the
training process exits. Training stops if sync fails.

`run RUN_NAME` with no arguments continues the run with its saved ones (other arguments are
refused): the trainer resumes the newest complete local pair, or starts over at step 0 if a
preemption came before the first save. A start that failed before training has already
claimed the run; running it again continues the same claim, so arguments `train.py` refuses
need a new RUN_NAME. Ctrl-C or SIGTERM to the launcher stops training and waits for the sync's
final upload.

### Interruptible pods

A preempted container restarts with its disk intact but nothing running. Its boot script,
`/root/onstart.sh` on Vast, can continue the run, retrying every 5 minutes until `run`
succeeds; a finished run exits 0 at once. `run` only continues a run that exists and exits
while another launcher holds it, so the line can stay while the run lasts. To stop a run for
good, remove the line from `/root/onstart.sh`, kill the loop
(`pkill -f 'until ./training/h100.sh run RUN_NAME'`), then send the launcher SIGTERM:

```bash
cat >> /root/onstart.sh <<'EOF'
cd /workspace/hibiki-zero && HF_HOME=/workspace/.hf_home HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft \
  nohup bash -c 'until ./training/h100.sh run RUN_NAME; do sleep 300; done' \
  >> /workspace/RUN_NAME.log 2>&1 &
EOF
```

A preemption loses the steps since the last save (≤1,000) and the restart: preflight's hash of
the 6.3 GB of weights, ~10 s of data loading, the model load, ~30 s of compilation, and a
validation if the run stopped between a validation step's save and its log line.

## Recover a run on a replacement pod

Check out the commit recorded in the remote `RUN_NAME/run.json` before setup, then set
up the pod as above and restore the newest complete model/trainer pair, with the promoted best
model and marker if the run has promoted one (`RUN_NAME/best/`):

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
  "$run/metadata/train_args.json" \
  "$run/checkpoints/model_stepNNNNNN.safetensors" \
  "$run/checkpoints/trainer_stepNNNNNN.pt" \
  "$run/best/best_stepBBBBBB.safetensors" \
  "$run/best/best_stepBBBBBB.json" \
  --local-dir "$restore_dir"

run_dir="runs/$run"
mkdir -p "$run_dir"
cp "$restore_dir/$run/run.json" "$run_dir/run_id.json"
cp "$restore_dir/$run/metadata/"{run_config,train_args}.json "$run_dir/"
cp "$restore_dir/$run/checkpoints/model_stepNNNNNN.safetensors" "$run_dir/"
cp "$restore_dir/$run/checkpoints/trainer_stepNNNNNN.pt" "$run_dir/"
cp "$restore_dir/$run/best/best_stepBBBBBB.safetensors" "$run_dir/"
cp "$restore_dir/$run/best/best_stepBBBBBB.json" "$run_dir/best.json"
# Logs exist in $run/artifacts/ once a sync ran after the training process exited; keep them.
./.venv/bin/hf download "$HIBIKI_HF_REPO" --include "$run/artifacts/*.jsonl" --local-dir "$restore_dir"
cp "$restore_dir/$run/artifacts/"*.jsonl "$run_dir/"
find "$restore_dir" -depth -delete

./training/h100.sh run "$run"
```

Continuing is interruption recovery for the same run only:

- it needs the run's identity and commit, its `train_args.json`, and an unchanged
  `run_config.json`;
- the remote pair may not be newer than the newest local one, and if the run has promoted a
  checkpoint (`RUN_NAME/best/`), its local `best.json` may not be older;
- a validation the interruption cut short runs again before training continues, and so does
  the resumed step's validation if the run's logs were not restored (the final sync then
  replaces the remote logs with the resumed run's only);
- it cannot extend the stop step or start a different run.

Runs recorded before the `inference/`/`training/` split, such as
`grounded_v2_full_direct_voice_5epoch`, follow their own commit's `docs/finetune.md`.

## Evaluate a checkpoint

Training has already translated FLEURS dev at every validation (`runs/RUN_NAME/val_log.jsonl`,
`runs/RUN_NAME/eval/stepNNNNNN/predictions.csv`). Check the promoted checkpoint on the other
sets, such as MultiMed `select`, the non-regression gate
([plan](vi_en_parity_plan.md#2-evaluation-sets)), or teacher-forced on another cache:

```bash
./.venv/bin/python -m training.finetune.validate \
  --cache-dir data/cache/en-vi-grounded-v2/validation \
  --checkpoint runs/RUN_NAME/best_stepNNNNNN.safetensors
./.venv/bin/python -m training.finetune.eval \
  --pairs data/eval/multimed-vi/select.jsonl --limit 1718 \
  --checkpoint runs/RUN_NAME/best_stepNNNNNN.safetensors \
  --batch-size 64 --stop-on-eos --text-only --seed 42 \
  --out-dir runs/RUN_NAME/multimed_select_stepNNNNNN
```

`validate` reports teacher-forced losses on a cache's validation streams. `eval` runs correct-source
free-running generation on rows with `id`, `vi_audio` and `text_en` columns and
writes predictions, BLEU/chrF/WER, and a generation-health verdict.
