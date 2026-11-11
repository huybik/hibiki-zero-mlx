# SFT launcher and recovery

Read when touching `training/h100.sh`, `training/finetune/hf_sync.py`, or the checkpoint writer
and resume in `training/finetune/checkpoints.py` (shared by `train.py` and `grpo.py`).
Back to [CONTEXT.md](../../CONTEXT.md) · [Launcher](#launcher) ·
[Checkpoints and recovery](#checkpoints-and-recovery)

The trainer itself is in [sft-training](sft-training.md); how a checkpoint is chosen and promoted
in [sft-evaluation](sft-evaluation.md#checkpoint-selection); the pod procedure in
[docs/finetune.md](../finetune.md).

## Launcher

`training/h100.sh` runs from the repo root and invokes modules as
`python -m training.<package>.<module>`.

- `setup` needs Linux, driver 580+ (CUDA 13), exactly one H100 or H200 (Hopper, compute
  capability 9.0: the kernels and measurements are sm_90's), uv, a C compiler, cmake, network for
  rustup, and no existing `.venv`. It builds a relocatable `.venv` on free-threaded Python 3.14
  ([environment](environment.md#python); Rust for the safetensors and sphn source builds; built
  aside and moved into place, so a failure leaves none) with Torch `2.15.0+cu134` (CUDA 13.4, the newest build on 2026-11-03), the
  newest of every pinned dependency (`huggingface-hub` 2.4), and `moshi==0.2.13 --no-deps`
  (moshi declares `torch<2.10`; 0.2.13 is its newest release and runs on 2.15).
- `preflight` checks:
  - a clean tracked tree;
  - free-threaded Python 3.14, exact Torch and Moshi versions; one H100 or H200 with compute capability 9.0;
  - ≥78 GiB VRAM (an 80 GB card shows 79.2), ≥80 GiB of RAM (the container's cgroup limit
    when lower than `/proc/meminfo`, which shows the host on Vast) and 130 GiB of disk: free
    space plus, for `run`, what the run dir already holds (each hard-linked inode once). `run`
    and `grpo` size it by the run's `--keep-checkpoints`: the kept pairs, the promoted model and
    a pair the sync still uploads, 130 GiB at the default 2 and 93 at 1 (0 keeps every pair, so
    it gets the default's);
  - the SHA-256 of the four upstream files in `weights/`.
- `run RUN_NAME [ARGS…]` is the only way to train, and the same command starts and continues.
  It holds `flock` on `runs/.RUN_NAME.lock` (inherited by the trainer and the sync), so a
  second launcher of a running run exits; then preflight:
  - without `runs/RUN_NAME/run_id.json`: an empty run dir (a start cut short before
    `run_id.json` leaves only `train_args.json`, which is removed) and an empty `RUN_NAME/`
    prefix in a public `HIBIKI_HF_REPO`; it writes `train_args.json` (ARGS), then `run_id.json`
    (commit plus random id), then uploads it as `RUN_NAME/run.json`;
  - with it: the saved ARGS (other ARGS are refused), a remote pair no newer than the newest local
    one, a local `best.json` no older than the remote's newest `best/` marker, and a `run_id.json`
    commit equal to HEAD; a missing remote `run.json` (its upload failed) is uploaded if the prefix
    is otherwise empty.
  Either way it runs `train.py --out-dir runs/RUN_NAME ARGS…` (`grpo RUN_NAME` runs `grpo.py`
  the same way, [grpo](grpo.md#running)) beside the `hf_sync` watcher, and
  kills training if sync dies. INT or TERM to the launcher stops training and waits for the sync's
  final upload (`wait` is retried: a trapped signal interrupts it). On an interruptible pod the
  boot script runs it ([docs/finetune.md](../finetune.md#interruptible-pods)).

## Checkpoints and recovery

- A complete pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt` (12.5 + 25 GB);
  two are kept locally. `checkpoints.CheckpointWriter` copies the model and optimizer state into
  pinned host buffers on the training stream and returns; a thread waits for the copy, prunes
  to `--keep-checkpoints` − 1 pairs, writes the model then the trainer file (each fsynced before
  its rename, then the directory), and prunes again. The next save, a promotion (which
  hard-links the model file) and the end of training wait for it; a failed write raises at the
  next log step.
- A start in an out-dir with complete pairs resumes the newest one (`checkpoints.open_run`), and
  only if `run_config.json` matches on `RESUME_KEYS` (`checkpoints.claim_run_config`, before the
  model loads, so a resume with other arguments stops in seconds): data dirs, weights,
  hyperparameters, cadences,
  seed, train row count, evaluation pairs and rows, and total steps. Lines of `train_log.jsonl`
  and `val_log.jsonl` past the resumed step are dropped. A start without a pair begins at step 0
  and removes the logs, `best.json` and `eval/` of a run cut short before its first save.
- A pair is saved before its validation, and a validation's log line is written after its promotion.
  A resume at a validation step (a multiple of `--val-every`) that `val_log.jsonl` lacks validates
  it first; a finished run exits 0 after loading the data and before loading the model (the boot
  hook's retry loop stops on it). Promotion relinks `best_stepS.safetensors` (a link to a temp name,
  then a rename) unless it is already this run's `model_stepS` (a replacement pod can restore best S
  beside an older pair, then retrain S). A torn last log line, or one missing its newline, is
  rewritten on resume. The writer's thread is not a daemon, so a Python exception still finishes the
  pair being written; the launcher's TERM kills it, and the run resumes from the pair before. A
  replacement pod has `val_log.jsonl` only if it restored the run's `artifacts/` (synced whenever
  the training process exits); without it, the resumed step's validation runs again and may promote
  it on sampling noise.
- `h100.sh run` refuses to continue unless the local `best.json` is at least as new as the
  recovery repo's newest `best/` marker: a resume without it would promote a worse checkpoint,
  and `hf_sync`, which keeps only the newest remote best, would replace the better one.
- Checked on the pods (2026-11-03), killing `train.py` with `kill -9` and rerunning the same
  command: on the H200 NVL, killed at step 70 with the step-60 pair half-written, it skipped
  that pair, resumed from step 40 and logged the same losses for steps 50–70; on the H100 SXM,
  killed once the step-80 pair was saved but before its validation was logged, it validated
  step 80 first, then trained to 160 at the s/step it had before the kill.
- Checked on the Mac CPU with a tiny random LM (2026-11-02): training, saving, both
  validations, promotion (a lower chrF keeps the promoted checkpoint, a higher one replaces it,
  a run failing the gates promotes none), resumes that rerun a validation cut short mid-run and
  at the final step with the original numbers. The launcher's best-checkpoint check was run on
  fake states.
- `hf_sync.py` uploads under `HIBIKI_HF_PREFIX` (the run name):
  - `checkpoints/`: on each poll, the newest complete pair at a validation step
    (`run_config.json`'s `val_every`) if newer than the remote, keeping the newest two; the
    final sync, after the training process exits, uploads the newest pair whatever its step;
  - `best/`: the newest promoted model only;
  - `metadata/run_config.json` and `metadata/train_args.json`;
  - logs, on the final sync only.
  It requires a public repo and a remote `run.json` byte-equal to the local `run_id.json`. It
  polls every 60 s and aborts after three failures.
- The sync and the launcher list the run's remote files with `list_repo_tree` under the prefix,
  which pages through any repo. `model_info(files_metadata=True).siblings` stops at 10,000
  files: once the recovery repo held 16,210 (2026-11-06), the first tranche run (`tranche_epoch1`)
  never saw its own uploads, failed verification three times and stopped at step 210. Checked:
  the tree gives the siblings' files and sizes exactly where those were complete (369 and 5,390
  files).
- The Hub caps a repo at 20,000 files. The recovery repo held 16,223 on 2026-11-06, most of them
  per-row evaluation `.txt` files of `en_speech_streams_epoch1b` (9,736) and
  `en_speech_noisy_epoch2` (5,390); a folder upload over the cap commits in parts until one is
  refused. Upload an evaluation as its `predictions.csv` and `metrics.json` only.
- `hf_sync` never restores anything; restoring is the manual procedure in
  [docs/finetune.md](../finetune.md).
