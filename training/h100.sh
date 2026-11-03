#!/usr/bin/env bash
# Hopper (H100/H200) setup and launcher for full-model Hibiki-Zero SFT with HF disaster recovery.
#
#   setup             a pinned .venv (Torch 2.15 on CUDA 13.4)
#   preflight         GPU, driver, pins, memory, disk, upstream weights
#   run RUN_NAME ...  start a run, or continue it: the same command resumes the newest local
#                     pair with the run's saved arguments, so a preempted pod reruns it on boot

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
RUNS_DIR="$REPO_ROOT/runs"
unset NO_TORCH_COMPILE

die() {
  echo "error: $*" >&2
  exit 1
}

require_python() {
  [[ -x "$PYTHON" ]] || die "run '$0 setup' first"
}

require_cuda_driver() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  [[ "$driver_version" =~ ^[0-9]+\. ]] || die "invalid NVIDIA driver: $driver_version"
  # CUDA 13 runs on R580 and newer.
  [[ "$(printf '%s\n' 580 "$driver_version" | sort -V | head -n 1)" == 580 ]] \
    || die "NVIDIA driver 580 or newer required, got $driver_version"
}

require_hopper_gpu() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ -n "$gpu_names" && "$gpu_names" != *$'\n'* ]] || die "exactly one GPU is required"
  [[ "${gpu_names^^}" == *H100* || "${gpu_names^^}" == *H200* ]] \
    || die "an H100 or H200 is required, got $gpu_names"
}

require_empty_dir() {
  local path="$1"
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "output directory is not empty: $path"
  fi
  mkdir -p "$path"
}

setup() {
  [[ "$(uname -s)" == Linux ]] || die "setup must run on the Linux GPU pod"
  command -v python3 >/dev/null || die "python3 is required"
  require_cuda_driver
  require_hopper_gpu
  python3 -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))' \
    || die "Python 3.10-3.13 is required"
  [[ ! -e "$VENV" ]] || die "refusing to reuse existing environment: $VENV"
  export PIP_NO_CACHE_DIR=1
  python3 -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install \
    torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
  "$PYTHON" -m pip install \
    aiohttp==3.14.5 \
    einops==0.8.2 \
    huggingface-hub==2.4.0 \
    numpy==2.5.4 \
    pyarrow==26.0.0 \
    sacrebleu==2.6.0 \
    safetensors==0.8.0 \
    sentencepiece==0.2.2 \
    sphn==0.2.1 \
    tqdm==4.70.1
  "$PYTHON" -m pip install --no-deps moshi==0.2.13
  "$PYTHON" -m pip freeze > "$VENV/h100-freeze.txt"
  echo "Training environment installed in $VENV"
}

preflight() {
  # The run needs MIN GiB of disk: free space plus what RUN_DIR (when given) already holds.
  local minimum_gib="$1" run_dir="${2:-}"
  require_python
  require_cuda_driver
  require_hopper_gpu
  git diff --quiet || die "tracked worktree changes must be committed before training"
  git diff --cached --quiet || die "staged changes must be committed before training"

  local profile
  profile="$("$PYTHON" - "$REPO_ROOT" "$minimum_gib" "$run_dir" <<'PY'
from __future__ import annotations

import hashlib
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import torch

root = Path(sys.argv[1])
minimum_gib = float(sys.argv[2])
run_dir = Path(sys.argv[3]) if sys.argv[3] else None

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise RuntimeError(f"Python 3.10-3.13 required, got {sys.version.split()[0]}")
if torch.__version__ != "2.15.0+cu134" or torch.version.cuda != "13.4":
    raise RuntimeError(f"torch 2.15.0+cu134 required, got {torch.__version__}")
if version("moshi") != "0.2.13":
    raise RuntimeError(f"moshi 0.2.13 required, got {version('moshi')}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA GPU is required")
name = torch.cuda.get_device_name(0)
# Hopper only: the kernels and the speed measurements are sm_90's.
hopper = any(gpu in name.upper() for gpu in ("H100", "H200"))
if not hopper or torch.cuda.get_device_capability(0) != (9, 0):
    raise RuntimeError(f"an H100 or H200 (compute capability 9.0) is required, got {name}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("CUDA bf16 support is required")
memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
# Training's 4,608-frame batches peak at 75.1 GiB reserved or less, and validation's translation
# below that: an 80 GB card fits (CUDA sees 79.2 GiB of it).
if memory_gib < 78:
    raise RuntimeError(f"an 80 GB GPU or larger is required, got {memory_gib:.1f} GiB")

host_bytes = 1024 * int(
    next(
        line.split()[1]
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
)
# In a container /proc/meminfo shows the host; the cgroup limit is what the run gets.
for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    text = Path(limit).read_text().strip() if Path(limit).is_file() else "max"
    if text != "max":
        host_bytes = min(host_bytes, int(text))
host_gib = host_bytes / 2**30
# The packed cache stays on disk; the checkpoint snapshot pins 37.5 GB.
if host_gib < 80:
    raise RuntimeError(f"at least 80 GiB of RAM required, got {host_gib:.1f}")
free_gib = shutil.disk_usage(root).free / 2**30
# A continued run's pairs, promoted model and staged uploads are disk it already has; the
# promoted model is a hard link, so each inode counts once.
held = {}
if run_dir is not None and run_dir.is_dir():
    for path in run_dir.rglob("*"):
        stat = path.lstat()
        if path.is_file() and not path.is_symlink():
            held[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
held_gib = sum(held.values()) / 2**30
if free_gib + held_gib < minimum_gib:
    raise RuntimeError(
        f"at least {minimum_gib:.0f} GiB of disk required for the run, got {free_gib:.0f} free"
        f" and {held_gib:.0f} held by the run"
    )

expected_hashes = {
    "weights/config.json": "a99f354a6131034b688fc9f91c889dc10e7eeff96ce65e94447be33d1be325a5",
    "weights/hibiki-pytorch-77f82164@110.safetensors": "cd78e453b3b80299255bea02be439bcc2552b57c03cd82dbf0e9792e20100db8",
    "weights/mimi-pytorch-e351c8d8@125.safetensors": "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    "weights/tokenizer_spm_48k_multi6_2.model": "c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for relative, expected in expected_hashes.items():
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {relative}")
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"hash mismatch for {relative}")

print(f"{memory_gib:.1f} {host_gib:.1f} {free_gib:.1f}")
PY
)"
  read -r GPU_GIB HOST_GIB FREE_GIB <<< "$profile"

  echo "Preflight passed: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "GPU=${GPU_GIB}GiB host=${HOST_GIB}GiB free_disk=${FREE_GIB}GiB"
}

wait_child() {  # PID: its exit status, waiting again while a trapped signal interrupts wait
  local status
  while true; do
    wait "$1"
    status=$?
    (( status > 128 )) && kill -0 "$1" 2>/dev/null || return "$status"
  done
}

run_with_sync() {
  local out_dir="$1"
  local mode="$2"
  shift 2
  (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) \
    || die "Bash 5.1 or newer is required"
  local repo="${HIBIKI_HF_REPO:-}"
  [[ -n "$repo" ]] || die "set HIBIKI_HF_REPO=owner/public-model-repo"
  local commit
  commit="$(git rev-parse HEAD)"
  "$PYTHON" - "$repo" "$mode" "$out_dir" "$commit" "$HIBIKI_HF_PREFIX" "$@" <<'PY'
import json
import re
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi


def steps(paths, pattern):
    return {int(match.group(1)) for path in paths if (match := re.fullmatch(pattern, path))}


api = HfApi()
info = api.model_info(sys.argv[1], files_metadata=True)
if info.private:
    raise RuntimeError("recovery model repo must be public")
prefix = sys.argv[5].strip("/") + "/"
remote_files = [
    item.rfilename.removeprefix(prefix)
    for item in info.siblings
    if item.rfilename.startswith(prefix)
]
run_dir = Path(sys.argv[3])
if sys.argv[2] == "fresh":
    if remote_files:
        raise RuntimeError(f"fresh training requires an empty {prefix} recovery prefix")
    # The arguments a continuation of the run reuses (the training command's, after
    # `-m training.finetune.train --out-dir DIR`), written first: run_id.json marks a run that
    # continues.
    (run_dir / "train_args.json").write_text(json.dumps(sys.argv[11:]) + "\n")
    identity = run_dir / "run_id.json"
    temporary = run_dir / ".run_id.json.tmp"
    temporary.write_text(
        json.dumps(
            {"commit": sys.argv[4], "run_id": secrets.token_hex(16), "version": 1},
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(identity)
    api.upload_file(
        path_or_fileobj=identity,
        path_in_repo=f"{prefix}run.json",
        repo_id=sys.argv[1],
        commit_message="Start full training run",
    )
else:
    local_models = steps(
        (path.name for path in run_dir.glob("model_step*.safetensors")),
        r"model_step(\d+)\.safetensors",
    )
    local_trainers = steps(
        (path.name for path in run_dir.glob("trainer_step*.pt")),
        r"trainer_step(\d+)\.pt",
    )
    local_latest = max(local_models & local_trainers, default=-1)
    remote_models = steps(remote_files, r"checkpoints/model_step(\d+)\.safetensors")
    remote_trainers = steps(remote_files, r"checkpoints/trainer_step(\d+)\.pt")
    remote_latest = max(remote_models & remote_trainers, default=-1)
    if remote_latest > local_latest:
        raise RuntimeError(f"remote checkpoint {remote_latest} is newer than local {local_latest}")
    # Without the promoted checkpoint, a worse one would be promoted and replace it remotely.
    remote_best = max(steps(remote_files, r"best/best_step(\d+)\.json"), default=-1)
    best = run_dir / "best.json"
    local_best = json.loads(best.read_text())["step"] if best.is_file() else -1
    if remote_best > local_best:
        raise RuntimeError(f"restore best/best_step{remote_best:06d} before resume")
    identity = run_dir / "run_id.json"
    if not identity.is_file():
        raise RuntimeError("restore run_id.json before resume")
    if json.loads(identity.read_text())["commit"] != sys.argv[4]:
        raise RuntimeError("training code changed since this run was created")
    if "run.json" not in remote_files:
        # The start wrote run_id.json but its upload failed: finish the claim.
        if remote_files:
            raise RuntimeError(f"{prefix} holds files but no run.json")
        api.upload_file(
            path_or_fileobj=identity,
            path_in_repo=f"{prefix}run.json",
            repo_id=sys.argv[1],
            commit_message="Start full training run",
        )
print(f"HF disaster recovery: {info.id}/tree/main/{prefix.rstrip('/')}")
PY

  "$@" &
  local train_pid=$!
  "$PYTHON" -m training.finetune.hf_sync "$out_dir" "$repo" --watch-pid "$train_pid" &
  local sync_pid=$!
  trap "kill -TERM $train_pid $sync_pid 2>/dev/null || true" EXIT
  trap "kill -TERM $train_pid 2>/dev/null || true" INT TERM

  # A trapped INT or TERM interrupts wait, which then returns no pid: wait again, so the sync
  # still makes its final upload after the trainer stops.
  local finished_pid="" first_status=0
  set +e
  until [[ -n "${finished_pid:-}" ]]; do
    wait -n -p finished_pid "$train_pid" "$sync_pid"
    first_status=$?
    [[ -n "${finished_pid:-}" ]] || (( first_status > 128 )) \
      || die "lost track of the trainer and the sync (wait returned $first_status)"
  done
  local train_status=0
  local sync_status=0
  if [[ "$finished_pid" == "$sync_pid" ]]; then
    sync_status=$first_status
    kill -TERM "$train_pid" 2>/dev/null || true
    wait_child "$train_pid"
    train_status=$?
    (( sync_status != 0 )) || sync_status=1
  else
    train_status=$first_status
    wait_child "$sync_pid"
    sync_status=$?
  fi
  set -e
  trap - INT TERM EXIT
  (( train_status == 0 )) || return "$train_status"
  (( sync_status == 0 )) || return "$sync_status"
}

run() {
  [[ $# -ge 1 ]] || die "usage: $0 run RUN_NAME [train args, needed when the run starts]"
  local run_name="$1"
  shift
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "RUN_NAME must start with a letter or digit and use only [A-Za-z0-9._-]"
  local arg
  for arg in "$@"; do
    [[ "$arg" != --out-dir && "$arg" != --out-dir=* ]] || die "the launcher sets --out-dir itself"
  done
  local run_dir="$RUNS_DIR/$run_name"
  export HIBIKI_HF_PREFIX="$run_name"
  # One launcher per run: the boot hook's and a manual one would clean each other's files. The
  # trainer and the sync inherit the lock, so it holds until all of them exit.
  mkdir -p "$RUNS_DIR"
  exec {lock_fd}>"$RUNS_DIR/.$run_name.lock"
  flock -n "$lock_fd" || die "$run_name is already running"
  # Two pairs (the default --keep-checkpoints), the promoted model, and a pair the sync still
  # uploads after it is pruned.
  preflight 130 "$run_dir"
  if [[ -f "$run_dir/run_id.json" ]]; then
    [[ -f "$run_dir/train_args.json" ]] || die "restore $run_dir/train_args.json to continue"
    local -a saved
    mapfile -t saved < <("$PYTHON" -c \
      'import json, sys; print("\n".join(json.load(open(sys.argv[1]))))' "$run_dir/train_args.json")
    if [[ $# -gt 0 && "$*" != "${saved[*]}" ]]; then
      die "$run_name was started with other arguments; run it with none to continue"
    fi
    run_with_sync "$run_dir" resume \
      "$PYTHON" -m training.finetune.train --out-dir "$run_dir" "${saved[@]}"
  else
    [[ $# -gt 0 ]] || die "a new run needs its train args (--cache-dir, --val-cache-dir, ...)"
    # A start cut short before run_id.json leaves only these behind.
    if [[ -d "$run_dir" && -z "$(find "$run_dir" -mindepth 1 -maxdepth 1 ! -name train_args.json \
        ! -name .run_id.json.tmp -print -quit)" ]]; then
      rm -f "$run_dir/train_args.json" "$run_dir/.run_id.json.tmp"
    fi
    require_empty_dir "$run_dir"
    run_with_sync "$run_dir" fresh "$PYTHON" -m training.finetune.train --out-dir "$run_dir" "$@"
  fi
}

command="${1:-}"
[[ $# -eq 0 ]] || shift
case "$command" in
  setup)
    setup
    ;;
  preflight)
    preflight 130
    ;;
  run)
    run "$@"
    ;;
  *)
    echo "usage: $0 setup|preflight|run RUN_NAME [ARGS...]" >&2
    exit 1
    ;;
esac
