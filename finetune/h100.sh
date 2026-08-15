#!/usr/bin/env bash
# H100 pod setup and launcher for the base-start Vietnamese full-model SFT.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
SMOKE_DIR="$REPO_ROOT/finetune/runs/h100_smoke"
RUN_DIR="$REPO_ROOT/finetune/runs/vi_base_full"

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
  [[ "$driver_version" =~ ^[0-9]+\. ]] || die "invalid NVIDIA driver version: $driver_version"
  [[ "$(printf '%s\n' 580 "$driver_version" | sort -V | head -n 1)" == "580" ]] \
    || die "CUDA 13.x minor-version compatibility requires NVIDIA driver 580 or newer, got $driver_version"
}

require_empty_dir() {
  local path="$1"
  if [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "output directory is not empty: $path"
  fi
  mkdir -p "$path"
}

setup() {
  [[ "$(uname -s)" == "Linux" ]] || die "setup must run on the Linux H100 pod"
  command -v python3 >/dev/null || die "python3 is required"
  require_cuda_driver
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ -n "$gpu_names" && "$gpu_names" != *$'\n'* ]] || die "exactly one visible GPU is required"
  [[ "${gpu_names^^}" == *H100* ]] || die "H100 required, got $gpu_names"
  python3 -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 14)))' \
    || die "Python 3.10-3.13 is required"
  [[ ! -e "$VENV" ]] || die "refusing to reuse existing environment: $VENV"
  export PIP_NO_CACHE_DIR=1
  python3 -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
  "$PYTHON" -m pip install \
    aiohttp==3.11.18 \
    datasets==4.8.5 \
    einops==0.8.1 \
    huggingface-hub==1.21.0 \
    numpy==2.2.6 \
    pyarrow==21.0.0 \
    sacrebleu==2.6.0 \
    safetensors==0.8.0 \
    sentencepiece==0.2.1 \
    soundfile==0.14.0 \
    sphn==0.2.1 \
    tqdm==4.67.1
  "$PYTHON" -m pip install --no-deps moshi==0.2.13
  "$PYTHON" -m pip freeze > "$VENV/h100-freeze.txt"
  echo "H100 environment installed in $VENV"
}

preflight() {
  local minimum_free_gib="${1:-190}"
  require_python
  require_cuda_driver
  git diff --quiet || die "tracked worktree changes must be committed before training"
  git diff --cached --quiet || die "staged changes must be committed before training"

  local profile
  profile="$("$PYTHON" - "$REPO_ROOT" "$minimum_free_gib" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import torch

root = Path(sys.argv[1])
minimum_free_gib = float(sys.argv[2])

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise RuntimeError(f"Python 3.10-3.13 required, got {sys.version.split()[0]}")
if torch.__version__ != "2.13.0+cu132":
    raise RuntimeError(f"torch 2.13.0+cu132 required, got {torch.__version__}")
if torch.version.cuda != "13.2":
    raise RuntimeError(f"CUDA 13.2 Torch build required, got {torch.version.cuda}")
if version("moshi") != "0.2.13":
    raise RuntimeError(f"moshi 0.2.13 required, got {version('moshi')}")
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"exactly one visible GPU required, got {torch.cuda.device_count()}")

name = torch.cuda.get_device_name(0)
if "H100" not in name.upper():
    raise RuntimeError(f"H100 required, got {name}")
if torch.cuda.get_device_capability(0) != (9, 0):
    raise RuntimeError(f"H100 compute capability 9.0 required, got {torch.cuda.get_device_capability(0)}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("CUDA bf16 support is required")

memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
if memory_gib >= 90:
    batch_size, grad_accum = 16, 1
elif memory_gib >= 75:
    batch_size, grad_accum = 8, 2
else:
    raise RuntimeError(f"at least 75 GiB GPU memory required, got {memory_gib:.1f} GiB")

meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
host_kib = int(next(line.split()[1] for line in meminfo if line.startswith("MemTotal:")))
host_gib = host_kib / 1024**2
if host_gib < 110:
    raise RuntimeError(f"at least 110 GiB host RAM required, got {host_gib:.1f} GiB")

free_gib = shutil.disk_usage(root).free / 2**30
if free_gib < minimum_free_gib:
    raise RuntimeError(
        f"at least {minimum_free_gib:.0f} GiB free disk required, got {free_gib:.1f} GiB"
    )

expected_hashes = {
    "weights/config.json": "a99f354a6131034b688fc9f91c889dc10e7eeff96ce65e94447be33d1be325a5",
    "weights/hibiki-pytorch-77f82164@110.safetensors": "cd78e453b3b80299255bea02be439bcc2552b57c03cd82dbf0e9792e20100db8",
    "weights/mimi-pytorch-e351c8d8@125.safetensors": "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    "weights/tokenizer_spm_48k_multi6_2.model": "c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697",
    "finetune/pairs/val128.jsonl": "ae978d6aee90e6774e1fbf4fe9a0b65bd98f4b43c93f4a59c2843409e4e7b627",
}


def sha256_file(path: Path) -> str:
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
        raise RuntimeError(f"hash mismatch for {relative}: {digest}")

expected_shards = {
    "finetune/cache/phomt_stream": 1377,
    "finetune/cache/train": 46,
    "finetune/cache/validation": 5,
}
for relative, expected in expected_shards.items():
    paths = sorted((root / relative).glob("shard_*.pt"))
    if len(paths) != expected:
        raise RuntimeError(f"{relative}: expected {expected} shards, got {len(paths)}")
    if any(path.stat().st_size == 0 for path in paths):
        raise RuntimeError(f"{relative}: found an empty shard")

expected_rows = {
    "finetune/pairs/val128.jsonl": 128,
    "finetune/pairs/validation.jsonl": 149,
    "finetune/pairs/test.jsonl": 347,
}
for relative, expected in expected_rows.items():
    manifest = root / relative
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {relative}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise RuntimeError(f"{relative}: expected {expected} rows, got {len(rows)}")
    for row in rows:
        audio = Path(row["vi_audio"])
        if not audio.is_absolute():
            audio = root / audio
        if not audio.is_file() or audio.stat().st_size == 0:
            raise FileNotFoundError(f"missing source audio for {row.get('id')}: {audio}")

print(batch_size, grad_accum, f"{memory_gib:.1f}", f"{host_gib:.1f}", f"{free_gib:.1f}")
PY
)"
  read -r BATCH_SIZE GRAD_ACCUM_STEPS GPU_GIB HOST_GIB FREE_GIB <<< "$profile"
  export BATCH_SIZE GRAD_ACCUM_STEPS
  export NO_TORCH_COMPILE=
  export HIBIKI_FRAME_BUCKET=32
  echo "Preflight passed: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "GPU=${GPU_GIB}GiB host=${HOST_GIB}GiB free_disk=${FREE_GIB}GiB batch=$BATCH_SIZE accum=$GRAD_ACCUM_STEPS"
}

common_args() {
  TRAIN_ARGS=(
    --device cuda
    --model-weight weights/hibiki-pytorch-77f82164@110.safetensors
    --cache-dir finetune/cache/phomt_stream finetune/cache/train
    --val-cache-dir finetune/cache/validation
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --max-frames 280
    --sort-by-length
    --epochs 2
    --text-prefix-pad-weight 0.5
    --seed 42
    --val-batch-size 8
    --eval-pairs finetune/pairs/val128.jsonl
    --eval-batch-size 8
    --eval-text-temp 0
  )
}

stop_monitor() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

check_smoke_outputs() {
  "$PYTHON" - "$SMOKE_DIR" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = [json.loads(line) for line in (root / "train_log.jsonl").read_text().splitlines() if line]
if not logs or logs[-1]["step"] != 11:
    raise RuntimeError("resume smoke did not reach step 11")
for item in logs:
    for key in ("loss", "audio_loss", "text_loss", "lr"):
        if not math.isfinite(float(item[key])):
            raise RuntimeError(f"non-finite {key} at step {item['step']}")

eval_dir = root / "standalone_eval_step10"
metrics = json.loads((eval_dir / "metrics.json").read_text())
with (eval_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
    predictions = list(csv.DictReader(handle))
if metrics["num_predictions"] != 8 or len(predictions) != 8:
    raise RuntimeError("standalone smoke eval did not produce eight predictions")

rows = []
with (root / "vram.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.reader(handle):
        if len(row) >= 3:
            rows.append((float(row[1]), float(row[2])))
if not rows:
    raise RuntimeError("VRAM monitor produced no samples")
peak_mib = max(row[0] for row in rows)
total_mib = max(row[1] for row in rows)
headroom_mib = total_mib - peak_mib
if headroom_mib < 2048:
    raise RuntimeError(f"smoke left only {headroom_mib / 1024:.1f} GiB VRAM headroom")
print(f"Smoke passed: peak VRAM {peak_mib / 1024:.1f}/{total_mib / 1024:.1f} GiB")
PY
}

smoke() {
  preflight 190
  require_empty_dir "$SMOKE_DIR"
  common_args

  nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.total \
    --format=csv,noheader,nounits -l 1 > "$SMOKE_DIR/vram.csv" &
  local monitor_pid=$!
  trap 'stop_monitor "$monitor_pid"' INT TERM EXIT

  set +e
  (
    set -Eeuo pipefail
    "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-4@0" --warmup-steps 500 \
      --text-weight-schedule "5@0" \
      --max-steps 10 \
      --val-every 10 --val-batches 1 \
      --eval-every 10 --eval-limit 8 \
      --save-every 10 --keep-checkpoints 1 --log-every 1 \
      --out-dir "$SMOKE_DIR"

    cp "$SMOKE_DIR/run_config.json" "$SMOKE_DIR/run_config_step10.json"
    "$PYTHON" finetune/eval.py \
      --device cuda --dtype float32 \
      --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
      --checkpoint "$SMOKE_DIR/model_step000010.safetensors" \
      --pairs finetune/pairs/val128.jsonl \
      --limit 8 --batch-size 8 --text-temp 0 \
      --stop-on-eos --text-only --seed 42 \
      --out-dir "$SMOKE_DIR/standalone_eval_step10"

    "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-4@0" --warmup-steps 500 \
      --text-weight-schedule "5@0" \
      --max-steps 11 \
      --val-every 0 --val-batches 1 --eval-every 0 \
      --save-every 11 --keep-checkpoints 1 --log-every 1 \
      --resume-checkpoint "$SMOKE_DIR/trainer_step000010.pt" \
      --out-dir "$SMOKE_DIR"
  )
  local status=$?
  set -e
  stop_monitor "$monitor_pid"
  trap - INT TERM EXIT
  (( status == 0 )) || return "$status"

  check_smoke_outputs
  rm -f \
    "$SMOKE_DIR/model_step000011.safetensors" \
    "$SMOKE_DIR/trainer_step000011.pt" \
    "$SMOKE_DIR"/best_step*.safetensors \
    "$SMOKE_DIR/best.json"
  {
    echo "commit=$(git rev-parse HEAD)"
    echo "batch_size=$BATCH_SIZE"
    echo "grad_accum_steps=$GRAD_ACCUM_STEPS"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$SMOKE_DIR/SMOKE_OK"
  echo "SMOKE_OK: $SMOKE_DIR"
}

require_current_smoke() {
  local marker="$SMOKE_DIR/SMOKE_OK"
  [[ -f "$marker" ]] || die "run '$0 smoke' successfully first"
  local smoke_commit
  smoke_commit="$(sed -n 's/^commit=//p' "$marker")"
  [[ "$smoke_commit" == "$(git rev-parse HEAD)" ]] || die "code changed after smoke; rerun it"
}

run_with_sync() {
  local out_dir="$1"
  local mode="$2"
  shift 2
  (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) \
    || die "Bash 5.1 or newer is required to supervise training and sync"
  local repo="${HIBIKI_HF_REPO:-}"
  [[ -n "$repo" ]] || die "set HIBIKI_HF_REPO=owner/public-model-repo before training"
  local commit
  commit="$(git rev-parse HEAD)"
  "$PYTHON" - "$repo" "$mode" "$out_dir" "$commit" <<'PY'
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
    raise RuntimeError("disaster-recovery model repo must be public")
prefix = "full_run/"
remote_files = [
    item.rfilename.removeprefix(prefix)
    for item in info.siblings
    if item.rfilename.startswith(prefix)
]
if sys.argv[2] == "fresh" and remote_files:
    raise RuntimeError("fresh training requires an empty full_run/ recovery prefix")
if sys.argv[2] == "resume":
    run_dir = Path(sys.argv[3])
    local_models = steps(
        (path.name for path in run_dir.glob("model_step*.safetensors")),
        r"model_step(\d+)\.safetensors",
    )
    local_trainers = steps(
        (path.name for path in run_dir.glob("trainer_step*.pt")), r"trainer_step(\d+)\.pt"
    )
    remote_models = steps(remote_files, r"checkpoints/model_step(\d+)\.safetensors")
    remote_trainers = steps(remote_files, r"checkpoints/trainer_step(\d+)\.pt")
    local_latest = max(local_models & local_trainers)
    remote_latest = max(remote_models & remote_trainers, default=-1)
    if remote_latest > local_latest:
        raise RuntimeError(f"remote checkpoint {remote_latest} is newer than local {local_latest}")
    remote_best_models = steps(remote_files, r"best/best_step(\d+)\.safetensors")
    remote_best_markers = steps(remote_files, r"best/best_step(\d+)\.json")
    remote_best = max(remote_best_models & remote_best_markers, default=-1)
    if remote_best >= 0:
        marker = run_dir / "best.json"
        if not marker.is_file():
            raise RuntimeError("restore the repo's best model and best.json before resume")
        state = json.loads(marker.read_text())
        if int(state["step"]) < remote_best or not (run_dir / str(state["model"])).is_file():
            raise RuntimeError("local best model is older than the repo best; restore it before resume")
    local_identity = run_dir / "run_id.json"
    if not local_identity.is_file():
        raise RuntimeError("missing local run_id.json; restore it from the repo before resume")
    if json.loads(local_identity.read_text())["commit"] != sys.argv[4]:
        raise RuntimeError("training code changed since this run was created")
if sys.argv[2] == "fresh":
    run_dir = Path(sys.argv[3])
    identity = run_dir / "run_id.json"
    temp = run_dir / ".run_id.json.tmp"
    temp.write_text(
        json.dumps(
            {"commit": sys.argv[4], "run_id": secrets.token_hex(16), "version": 1},
            sort_keys=True,
        )
        + "\n"
    )
    temp.replace(identity)
    upload_error = None
    try:
        api.upload_file(
            path_or_fileobj=identity,
            path_in_repo=f"{prefix}run.json",
            repo_id=sys.argv[1],
            commit_message="Start full training run",
        )
    except Exception as exc:
        upload_error = exc
    sizes = {
        item.rfilename.removeprefix(prefix): item.size
        for item in api.model_info(sys.argv[1], files_metadata=True).siblings
        if item.rfilename.startswith(prefix) and item.size is not None
    }
    if sizes.get("run.json") != identity.stat().st_size:
        if upload_error is not None:
            raise upload_error
        raise RuntimeError("failed to verify remote run identity")
print(f"HF disaster recovery: {info.id}/tree/main/{prefix.rstrip('/')}")
PY

  if [[ "$mode" == "resume" ]]; then
    "$PYTHON" finetune/hf_sync.py "$out_dir" "$repo"
  fi

  "$@" &
  local train_pid=$!
  "$PYTHON" finetune/hf_sync.py "$out_dir" "$repo" --watch-pid "$train_pid" &
  local sync_pid=$!
  trap "kill -TERM $train_pid $sync_pid 2>/dev/null || true" EXIT
  trap "kill -TERM $train_pid 2>/dev/null || true" INT TERM

  local finished_pid=""
  set +e
  wait -n -p finished_pid "$train_pid" "$sync_pid"
  local first_status=$?
  local train_status=0
  local sync_status=0
  if [[ -z "${finished_pid:-}" ]]; then
    wait "$train_pid"
    train_status=$?
    wait "$sync_pid"
    sync_status=$?
  elif [[ "$finished_pid" == "$sync_pid" ]]; then
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

train() {
  preflight 190
  require_current_smoke
  require_empty_dir "$RUN_DIR"
  common_args
  run_with_sync "$RUN_DIR" fresh "$PYTHON" finetune/train.py \
    "${TRAIN_ARGS[@]}" \
    --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
    --text-weight-schedule "5@0,2@0.6" \
    --val-every 2000 \
    --eval-every 9000 --eval-limit 128 \
    --save-every 3000 --keep-checkpoints 2 --log-every 10 \
    --out-dir "$RUN_DIR"
}

resume() {
  [[ $# -eq 1 ]] || die "usage: $0 resume <trainer_stepNNNNNN.pt>"
  preflight 50
  require_current_smoke
  [[ -f "$1" ]] || die "missing resume checkpoint: $1"
  local checkpoint
  checkpoint="$(realpath "$1")"
  local out_dir
  out_dir="$(dirname "$checkpoint")"
  common_args
  run_with_sync "$out_dir" resume "$PYTHON" finetune/train.py \
    "${TRAIN_ARGS[@]}" \
    --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
    --text-weight-schedule "5@0,2@0.6" \
    --val-every 2000 \
    --eval-every 9000 --eval-limit 128 \
    --save-every 3000 --keep-checkpoints 2 --log-every 10 \
    --resume-checkpoint "$checkpoint" \
    --out-dir "$out_dir"
}

case "${1:-}" in
  setup)
    setup
    ;;
  preflight)
    preflight 190
    ;;
  smoke)
    smoke
    ;;
  train)
    train
    ;;
  resume)
    shift
    resume "$@"
    ;;
  *)
    echo "usage: $0 setup|preflight|smoke|train|resume <trainer checkpoint>" >&2
    exit 1
    ;;
esac
