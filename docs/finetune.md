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
# Needs uv (https://docs.astral.sh/uv/), a C compiler and cmake: free-threaded Python 3.14 with source builds.
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

`setup` creates a new `.venv` on free-threaded Python 3.14 with Torch `2.15.0+cu134`, Moshi
`0.2.13` and the pinned dependencies, and refuses to reuse an existing environment. `preflight`
checks the Python build, the GPU, driver, package pins, RAM (the container's limit, not the host's), free disk,
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
./.venv/bin/hf download huybik/hibiki-zero-vi-full-sft --repo-type dataset --local-dir data \
  --include "eval/*"
./.venv/bin/python -m training.prep.eval_sets --sets giga-heldout  # gated: HF_TOKEN with access
```

The second command packs each shard into a flat memory-mapped array beside it
(`<cache-dir>/.packed/`, 70 GB, ~30 s): every later start loads the 3.44M rows in ~9 s instead
of unpickling them for 7.5 minutes, and keeps them out of RAM. Training builds missing packs
itself, so this only moves that work before `run` checks the free disk. The third downloads the
evaluation sets into `data/eval/` (`training.prep.eval_sets` builds them anew); training translates FLEURS dev and the held-out `en-vi-val` rows at every
validation, scores each set apart, chooses its checkpoint on FLEURS, and refuses to start
without them
([sft-evaluation](agent-memory/sft-evaluation.md#evaluation-sets)). The fourth builds the held-out
real-Vietnamese set, which GigaSpeech 2's terms keep off that public repo; pass it to `eval` below.

## Run

```bash
./training/h100.sh run RUN_NAME \
  --cache-dir data/cache/en-vi-grounded-v2/train \
  --val-cache-dir data/cache/en-vi-grounded-v2/validation
```

The first stream run trains on the `en_speech` cache alone; the PhoMT cache and `cv-anquach` come
back later as more `--cache-dir` ablations ([plan](vi_en_parity_plan.md#1-data)). Arguments after
`RUN_NAME` go to `training.finetune.train` (`--help` lists them). The defaults are 30–120 s
streams in batches of at most 4,608 frames, one epoch (about 131,700 steps), peak LR `1e-6` with 5%
warmup and cosine decay, audio codebook 0 weighted 100 and text PADs 0.5, fused AdamW
`(0.9, 0.95)` with weight decay `0.1`, checkpoints every 1,000 steps and validation every 9,000
([sft-training](agent-memory/sft-training.md#trainer)). Each validation logs teacher-forced
losses, translates FLEURS dev and the `en-vi-val` sets free-running, scoring each set apart, and
promotes the checkpoint whose every set passes the generation health gates with the highest
FLEURS dev chrF ([sft-evaluation](agent-memory/sft-evaluation.md#checkpoint-selection)).

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

### A second epoch on the noisy cache

The noisy cache holds the clean cache's rows with the Vietnamese source of 60% of them re-encoded
under simulated rooms, noise and bands ([noisy-source](agent-memory/noisy-source.md)). Start its
`fetch` while the first run still trains: it downloads the 442 GB of parquet shards (~3.2 h at
the pod's ~39 MB/s) and keeps ~80 GB of the selected rows' Vietnamese audio, pausing below 50 GB
of free disk. Once the run has finished, `encode` writes the shards on the GPU, and a new run
trains from the first run's final model. Built once on 2026-11-04, the split is also in the
dataset repo under `en-vi-grounded-v2-noisy/train/` (1,832 shards, 73 GB). Another pod downloads
it with `--include "en-vi-grounded-v2-noisy/train/*"` and packs it rather than running `fetch` and
`encode`.

```bash

```bash
C=data/cache/en-vi-grounded-v2/train O=data/cache/en-vi-grounded-v2-noisy/train
V=data/cache/en-vi-grounded-v2/validation
# From a second worktree of the PR's commit: a preempted first run resumes only on its own commit.
nohup ./.venv/bin/python -m training.prep.mimi_cache_cuda fetch --cache-dir $C --out-dir $O >> fetch.log 2>&1 &
# After the first run's final save and sync (its final pair is in PREVIOUS_RUN/checkpoints/ on the Hub):
ln runs/PREVIOUS_RUN/model_stepFINAL.safetensors weights/PREVIOUS_RUN_stepFINAL.safetensors
rm runs/PREVIOUS_RUN/trainer_step*.pt runs/PREVIOUS_RUN/model_stepOTHER.safetensors
rm -r $C/.packed                                            # 70 GB
./.venv/bin/python -m training.prep.eval_sets --sets en-vi-val-noisy
./.venv/bin/python -m training.prep.mimi_cache_cuda encode --cache-dir $C --out-dir $O   # ends with "every shard written"
rm -r $O/.sources $C/shard_*.pt                             # 73 GB; the clean shards stay on the Hub
./.venv/bin/python -m training.finetune.streams $O $V       # pack before run checks the free disk
./training/h100.sh run RUN_NAME --cache-dir $O --val-cache-dir $V \
  --model-weight weights/PREVIOUS_RUN_stepFINAL.safetensors --seed 43 --val-every 18000 \
  --eval-pairs data/eval/fleurs-vi/dev.jsonl data/eval/en-vi-val/{cv27,libriheavy}.jsonl \
  data/eval/en-vi-val-noisy/{cv27,libriheavy}.jsonl
```

- Disk: the noisy split's shards (73 GB) and packs (70 GB) replace the clean ones. `run` checks
  130 GiB free before the trainer packs a split itself, so pack first; with the clean shards still
  there the run fills the 350 GB disk at its first checkpoints.
- `encode` skips a shard whose source a failed download left missing, and exits non-zero naming
  it: run `fetch` again (it retries each download 4 times), then `encode`.

- `--model-weight` loads the full fp32 model (moshi's loader is strict; for Hibiki models it
  also copies the text embedding's PAD row over row 2, which got no gradient in the first run).
  The hard link keeps it while the previous run's folder is pruned. Keep it at the same path
  until the run ends: a resume needs it there, and `en_speech_noisy_epoch2`'s commit also
  reloads it at every validation.
  A replacement pod downloads it from `PREVIOUS_RUN/checkpoints/` in the recovery repo.
- A new `--seed` draws new streams: the noisy split has the same rows, so seed 42 would replay
  the first run's streams and batch order.
- Train only after `encode` prints `every shard written`: the trainer takes whatever shards the
  folder holds.
- The run starts a new warmup and cosine decay. Validation every 18,000 steps (7 per epoch plus
  the final step) halves its share of the run; it also scores `en-vi-val-noisy`, so the run shows
  what the channel costs as it trains.
- `en_speech_noisy_epoch2` runs a commit that still gated EOS, which fails at every
  validation, so it promotes nothing, and the sync keeps only the newest two validation pairs. A
  script beside the run keeps its best validation model by FLEURS dev chrF, whatever the gates. After each
  validation it copies that model server side (`CommitOperationCopy`) to
  `RUN_NAME/offline_best/`, with its scores in `best.json`. `hf_sync` never prunes that folder.

### Interruptible pods

A preempted container restarts with its disk intact but nothing running. Its boot script,
`/root/onstart.sh` on Vast, can continue the run, retrying every 5 minutes until `run`
succeeds; a finished run exits 0 at once. A GRPO run's line uses `grpo` in place of `run`
([grpo](agent-memory/grpo.md#running)). `run` only continues a run that exists and exits
while another launcher holds it, so the line can stay while the run lasts. The same loop also
reruns a run that stops on its own, after a crash or after three failed syncs in a row. So start
a run through it by hand, not with a bare `run`, and add it to `/root/onstart.sh`. For a launcher
already running bare, start a loop that waits for its lock first:
`bash -c 'flock runs/.RUN_NAME.lock true; until …; done'`. To stop a run for
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

### Move a running run to another pod

While training runs, `hf_sync` uploads only the pairs of validation steps. To move a run to
another pod:

1. Wait for a validation's `val step=` lines.
2. Send the launcher SIGTERM. Its final sync uploads the newest pair and the logs.
3. Restore the run on the new pod as above, and start it through the boot hook's loop
   ([interruptible pods](#interruptible-pods)).

`en_speech_noisy_epoch2` moved this way at step 18,000 on 2026-11-04, with the GPUs idle ~25 min
between the two pods.

A run continues only on the commit in its `run.json`. To move it to a commit that changes no part
of training, only validation or docs (check with `git diff`):

1. Set `commit` in `runs/RUN_NAME/run_id.json` to the new HEAD.
2. Upload that file unchanged as `RUN_NAME/run.json` before `run`; `hf_sync` requires the two to
   be byte-equal.

That run moved from 991b3a4 to e852735 this way, to take the batch-sized KV cache.

The move to free-threaded Python 3.14 (PR #104) counts as such a commit: its `train.py` change
only names `fork` for the DataLoader, 3.12's default, and the loss measured the same on both. To
move a run onto it:

1. Stop the run at a checkpoint, by its PID file (`restart.sh` stops its whole process group).
2. Only then swap the environment: `mv .venv .venv-py312` and move the new one (built by
   `h100.sh setup` in another checkout, relocatable) to `.venv`; never under a running job, whose
   later imports and compile workers would load the other Python.
3. Pull the commit, set `commit` in `run_id.json` and upload it as `run.json` (the two steps
   above), then rerun.

Runs recorded before the `inference/`/`training/` split, such as
`grounded_v2_full_direct_voice_5epoch`, follow their own commit's `docs/finetune.md`.

## Evaluate a checkpoint

Training has already translated FLEURS dev and the `en-vi-val` sets at every validation
(`runs/RUN_NAME/val_log.jsonl`,
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
free-running generation on rows with `id`, `vi_audio`, `text_en` and `duration_s` columns (or
`en_audio` and `text_vi` for English to Vietnamese) and
writes predictions, BLEU/chrF/WER, and a generation-health verdict. `--pairs` takes several
files, all their rows unless `--limit` caps each, and each is scored as its own set.
