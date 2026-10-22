#!/usr/bin/env bash
# H100 setup and launcher for full-model Hibiki-Zero SFT with HF disaster recovery.

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
  [[ "$(printf '%s\n' 570 "$driver_version" | sort -V | head -n 1)" == 570 ]] \
    || die "NVIDIA driver 570 or newer required, got $driver_version"
}

require_h100_gpu() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ -n "$gpu_names" && "$gpu_names" != *$'\n'* ]] || die "exactly one GPU is required"
  [[ "${gpu_names^^}" == *H100* ]] || die "H100 required, got $gpu_names"
}

require_empty_dir() {
  local path="$1"
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "output directory is not empty: $path"
  fi
  mkdir -p "$path"
}

setup() {
  [[ "$(uname -s)" == Linux ]] || die "setup must run on the Linux H100 pod"
  command -v python3 >/dev/null || die "python3 is required"
  require_cuda_driver
  require_h100_gpu
  python3 -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))' \
    || die "Python 3.10-3.13 is required"
  [[ ! -e "$VENV" ]] || die "refusing to reuse existing environment: $VENV"
  export PIP_NO_CACHE_DIR=1
  python3 -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  "$PYTHON" -m pip install \
    aiohttp==3.11.18 \
    einops==0.8.1 \
    huggingface-hub==1.21.0 \
    numpy==2.2.6 \
    sacrebleu==2.6.0 \
    safetensors==0.8.0 \
    sentencepiece==0.2.1 \
    sphn==0.2.1 \
    tqdm==4.67.1 \
    num2words==0.5.14 \
    transformers==5.14.1
  "$PYTHON" -m pip install --no-deps moshi==0.2.13
  "$PYTHON" -m pip freeze > "$VENV/h100-freeze.txt"
  echo "H100 environment installed in $VENV"
}

preflight() {
  local minimum_free_gib="${1:-190}"
  require_python
  require_cuda_driver
  require_h100_gpu
  git diff --quiet || die "tracked worktree changes must be committed before training"
  git diff --cached --quiet || die "staged changes must be committed before training"

  local profile
  profile="$("$PYTHON" - "$REPO_ROOT" "$minimum_free_gib" <<'PY'
from __future__ import annotations

import hashlib
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import torch

root = Path(sys.argv[1])
minimum_free_gib = float(sys.argv[2])

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise RuntimeError(f"Python 3.10-3.13 required, got {sys.version.split()[0]}")
if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
    raise RuntimeError(f"torch 2.8.0+cu128 required, got {torch.__version__}")
if version("moshi") != "0.2.13":
    raise RuntimeError(f"moshi 0.2.13 required, got {version('moshi')}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("exactly one CUDA GPU is required")
name = torch.cuda.get_device_name(0)
if "H100" not in name.upper() or torch.cuda.get_device_capability(0) != (9, 0):
    raise RuntimeError(f"H100 compute capability 9.0 required, got {name}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("CUDA bf16 support is required")
memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
if memory_gib < 90:
    raise RuntimeError(f"physical batch 16 requires at least 90 GiB, got {memory_gib:.1f}")

host_kib = int(
    next(
        line.split()[1]
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
)
host_gib = host_kib / 1024**2
if host_gib < 110:
    raise RuntimeError(f"at least 110 GiB host RAM required, got {host_gib:.1f}")
free_gib = shutil.disk_usage(root).free / 2**30
if free_gib < minimum_free_gib:
    raise RuntimeError(f"at least {minimum_free_gib:.0f} GiB free disk required")

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
  "$PYTHON" - "$repo" "$mode" "$out_dir" "$commit" "$HIBIKI_HF_PREFIX" <<'PY'
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
    local_latest = max(local_models & local_trainers)
    remote_models = steps(remote_files, r"checkpoints/model_step(\d+)\.safetensors")
    remote_trainers = steps(remote_files, r"checkpoints/trainer_step(\d+)\.pt")
    remote_latest = max(remote_models & remote_trainers, default=-1)
    if remote_latest > local_latest:
        raise RuntimeError(f"remote checkpoint {remote_latest} is newer than local {local_latest}")
    identity = run_dir / "run_id.json"
    if not identity.is_file():
        raise RuntimeError("restore run_id.json before resume")
    if json.loads(identity.read_text())["commit"] != sys.argv[4]:
        raise RuntimeError("training code changed since this run was created")
print(f"HF disaster recovery: {info.id}/tree/main/{prefix.rstrip('/')}")
PY

  if [[ "$mode" == resume ]]; then
    "$PYTHON" -m training.finetune.hf_sync "$out_dir" "$repo"
  fi
  "$@" &
  local train_pid=$!
  "$PYTHON" -m training.finetune.hf_sync "$out_dir" "$repo" --watch-pid "$train_pid" &
  local sync_pid=$!
  trap "kill -TERM $train_pid $sync_pid 2>/dev/null || true" EXIT
  trap "kill -TERM $train_pid 2>/dev/null || true" INT TERM

  local finished_pid=""
  set +e
  wait -n -p finished_pid "$train_pid" "$sync_pid"
  local first_status=$?
  local train_status=0
  local sync_status=0
  if [[ "$finished_pid" == "$sync_pid" ]]; then
    sync_status=$first_status
    kill -TERM "$train_pid" 2>/dev/null || true
    wait "$train_pid"
    train_status=$?
    (( sync_status != 0 )) || sync_status=1
  else
    train_status=$first_status
    wait "$sync_pid"
    sync_status=$?
  fi
  set -e
  trap - INT TERM EXIT
  (( train_status == 0 )) || return "$train_status"
  (( sync_status == 0 )) || return "$sync_status"
}

require_train_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --out-dir | --out-dir=* | --resume-checkpoint | --resume-checkpoint=*)
        die "the launcher sets $arg itself"
        ;;
    esac
  done
}

train() {
  [[ $# -ge 1 ]] \
    || die "usage: $0 train RUN_NAME --cache-dir DIR... --val-cache-dir DIR [train args]"
  local run_name="$1"
  shift
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "RUN_NAME must start with a letter or digit and use only [A-Za-z0-9._-]"
  require_train_args "$@"
  preflight 190
  local run_dir="$RUNS_DIR/$run_name"
  require_empty_dir "$run_dir"
  export HIBIKI_HF_PREFIX="$run_name"
  run_with_sync "$run_dir" fresh "$PYTHON" -m training.finetune.train \
    --out-dir "$run_dir" "$@"
}

resume() {
  [[ $# -ge 1 ]] \
    || die "usage: $0 resume runs/RUN_NAME/trainer_stepNNNNNN.pt [the run's train args]"
  preflight 50
  [[ -f "$1" ]] || die "missing resume checkpoint: $1"
  local checkpoint
  checkpoint="$(realpath "$1")"
  shift
  require_train_args "$@"
  local out_dir
  out_dir="$(dirname "$checkpoint")"
  [[ "$(dirname "$out_dir")" == "$(realpath "$RUNS_DIR")" ]] \
    || die "checkpoint is not in a runs/ directory"
  local run_name
  run_name="$(basename "$out_dir")"
  export HIBIKI_HF_PREFIX="$run_name"
  run_with_sync "$out_dir" resume "$PYTHON" -m training.finetune.train \
    --resume-checkpoint "$checkpoint" --out-dir "$out_dir" "$@"
}

command="${1:-}"
[[ $# -eq 0 ]] || shift
case "$command" in
  setup)
    setup
    ;;
  preflight)
    preflight 190
    ;;
  train)
    train "$@"
    ;;
  resume)
    resume "$@"
    ;;
  *)
    echo "usage: $0 setup|preflight|train RUN_NAME ARGS...|resume CHECKPOINT ARGS..." >&2
    exit 1
    ;;
esac
