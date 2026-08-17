#!/usr/bin/env bash
# H100 setup and launcher for direct Vietnamese-to-English full-model SFT.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
RECIPE="${HIBIKI_RECIPE:-grounded-v2}"
SMOKE_DIR="$REPO_ROOT/finetune/runs/h100_smoke_grounded_v2_full_direct_voice_5epoch"
RUN_DIR="$REPO_ROOT/finetune/runs/vi_grounded_v2_full_direct_voice_5epoch"
PHOMT_CACHE="finetune/cache/phomt_grounded_v2"
TRAIN_CACHE="finetune/cache/train_grounded_v2"
VAL_CACHE="finetune/cache/validation_grounded_v2"
FULL_DATA_DIR="$REPO_ROOT/finetune/runs/grounded_v2_full_direct_5epoch_receipt"
FULL_DATA_RECEIPT="$FULL_DATA_DIR/full_data_receipt.json"
FULL_MANIFEST="$FULL_DATA_DIR/sample_manifest.jsonl"
HIBIKI_HF_PREFIX="grounded_v2_full_direct_voice_5epoch"
export HIBIKI_HF_PREFIX HIBIKI_HF_SYNC_INTERVAL=9000 HIBIKI_FRAME_BUCKET=16
unset NO_TORCH_COMPILE

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "$RECIPE" == grounded-v2 ]] || die "only HIBIKI_RECIPE=grounded-v2 is supported"
for obsolete in \
  HIBIKI_PILOT HIBIKI_HIGH_DELAY_PILOT HIBIKI_CONTRASTIVE_PILOT \
  HIBIKI_ASR_PREADAPT HIBIKI_ASR_ONE_EPOCH HIBIKI_ASR_ASCII \
  HIBIKI_ASR_TRANSLATION_PILOT HIBIKI_ASR_REPLAY_TRANSLATION_PILOT \
  HIBIKI_POST_SOURCE_EOS_TRANSLATION_PILOT HIBIKI_POST_SOURCE_EOS_EXTENSION; do
  obsolete_value="${!obsolete:-0}"
  [[ "$obsolete_value" == 0 ]] \
    || die "$obsolete is obsolete; the launcher now has one direct full-data recipe"
done
[[ -z "${HIBIKI_MAX_SAMPLES:-}" ]] || die "full training rejects HIBIKI_MAX_SAMPLES"
[[ -z "${HIBIKI_MAX_STEPS:-}" ]] || die "full training rejects HIBIKI_MAX_STEPS"
[[ -z "${HIBIKI_CACHE_SAMPLE_SHARDS:-}" ]] \
  || die "full cache building rejects HIBIKI_CACHE_SAMPLE_SHARDS"
[[ -z "${HIBIKI_HF_PREFIX_OVERRIDE:-}" ]] \
  || die "the direct full-run HF prefix is fixed"

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
    tqdm==4.67.1 \
    num2words==0.5.14 \
    transformers==5.14.1
  "$PYTHON" -m pip install --no-deps moshi==0.2.13
  "$PYTHON" -m pip freeze > "$VENV/h100-freeze.txt"
  echo "H100 environment installed in $VENV"
}

cache_grounded() {
  require_python
  require_cuda_driver
  require_h100_gpu
  [[ -f finetune/pairs/train.jsonl ]] || die "missing finetune/pairs/train.jsonl"
  [[ -f finetune/pairs/validation.jsonl ]] || die "missing finetune/pairs/validation.jsonl"
  local workers="${HIBIKI_CACHE_WORKERS:-4}"
  [[ "$workers" =~ ^[1-9][0-9]*$ ]] || die "HIBIKI_CACHE_WORKERS must be positive"
  local pids=()
  local worker
  for ((worker = 0; worker < workers; worker++)); do
    "$PYTHON" finetune/cache_phomt_stream.py \
      --recipe grounded-v2 --device cuda --profile h100 --out-dir "$PHOMT_CACHE" \
      --worker "$worker" --num-workers "$workers" &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT
  local remaining="$workers"
  local status
  while (( remaining > 0 )); do
    set +e
    wait -n
    status=$?
    set -e
    if (( status != 0 )); then
      kill "${pids[@]}" 2>/dev/null || true
      wait "${pids[@]}" 2>/dev/null || true
      trap - INT TERM EXIT
      return "$status"
    fi
    remaining=$((remaining - 1))
  done
  trap - INT TERM EXIT

  "$PYTHON" - "$PHOMT_CACHE" <<'PY'
import sys
from pathlib import Path
from finetune.publish_grounded_cache import validate_cache

stats = validate_cache(Path(sys.argv[1]))
print(
    f"Validated PhoMT cache: {stats['shards']} shards / "
    f"{stats['accepted_samples']} accepted / {stats['rejected_samples']} rejected"
)
PY
  "$PYTHON" finetune/cache_codes.py \
    --recipe grounded-v2 --device cuda --pairs finetune/pairs/train.jsonl \
    --out-dir "$TRAIN_CACHE"
  "$PYTHON" finetune/cache_codes.py \
    --recipe grounded-v2 --device cuda --pairs finetune/pairs/validation.jsonl \
    --out-dir "$VAL_CACHE"
  echo "Grounded-v2 caches complete"
}

preflight() {
  local minimum_free_gib="${1:-190}"
  require_python
  require_cuda_driver
  require_h100_gpu
  git diff --quiet || die "tracked worktree changes must be committed before training"
  git diff --cached --quiet || die "staged changes must be committed before training"

  local profile
  profile="$("$PYTHON" - "$REPO_ROOT" "$minimum_free_gib" \
    "$PHOMT_CACHE" "$TRAIN_CACHE" "$VAL_CACHE" <<'PY'
from __future__ import annotations

import hashlib
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import torch

root = Path(sys.argv[1])
minimum_free_gib = float(sys.argv[2])
cache_paths = sys.argv[3:6]

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

for relative, expected in zip(cache_paths, (1377, 46, 5), strict=True):
    shards = sorted((root / relative).glob("shard_*.pt"))
    if len(shards) != expected or any(path.stat().st_size == 0 for path in shards):
        raise RuntimeError(f"{relative}: expected {expected} complete shards")
    payload = torch.load(shards[0], map_location="cpu")
    if payload.get("format") != "hibiki_vn_grounded_cache_v2":
        raise RuntimeError(f"{relative}: not a grounded-v2 cache")
    if float(payload.get("alignment_min_score") or 0) != 0.5:
        raise RuntimeError(f"{relative}: CTC threshold is not 0.5")
    if any(
        sample.get("text_timing") != "wav2vec2_ctc_word_v1"
        for sample in payload.get("samples", [])
    ):
        raise RuntimeError(f"{relative}: missing CTC-timed English text")

print(f"{memory_gib:.1f} {host_gib:.1f} {free_gib:.1f}")
PY
)"
  read -r GPU_GIB HOST_GIB FREE_GIB <<< "$profile"

  "$PYTHON" finetune/freeze_full_data_receipt.py \
    --phomt-cache "$PHOMT_CACHE" \
    --fleurs-train-cache "$TRAIN_CACHE" \
    --fleurs-validation-cache "$VAL_CACHE" \
    --config-path weights/config.json \
    --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
    --mimi-weight weights/mimi-pytorch-e351c8d8@125.safetensors \
    --tokenizer weights/tokenizer_spm_48k_multi6_2.model \
    --out-dir "$FULL_DATA_DIR"

  echo "Preflight passed: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "GPU=${GPU_GIB}GiB host=${HOST_GIB}GiB free_disk=${FREE_GIB}GiB batch=16 accum=1"
}

common_args() {
  TRAIN_ARGS=(
    --device cuda
    --model-weight weights/hibiki-pytorch-77f82164@110.safetensors
    --mimi-weight weights/mimi-pytorch-e351c8d8@125.safetensors
    --tokenizer weights/tokenizer_spm_48k_multi6_2.model
    --cache-dir "$PHOMT_CACHE" "$TRAIN_CACHE"
    --val-cache-dir "$VAL_CACHE"
    --full-data-receipt "$FULL_DATA_RECEIPT"
    --input-sample-manifest "$FULL_MANIFEST"
    --epochs 5
    --batch-size 16
    --grad-accum-steps 1
    --max-frames 280
    --val-max-frames 280
    --val-batch-size 4
    --lr 1e-6
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1
    --grad-clip 1.0
    --audio-loss-weight 1.0
    --text-loss-weight 1.0
    --text-pad-loss-weight 0.05
    --seed 42
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
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = [
    json.loads(line)
    for line in (root / "train_log.jsonl").read_text().splitlines()
    if line
]
if not logs or logs[-1]["step"] != 11:
    raise RuntimeError("resume smoke did not reach step 11")
for item in logs:
    for key in ("loss", "audio_loss", "text_loss", "lr"):
        if not math.isfinite(float(item[key])):
            raise RuntimeError(f"non-finite {key} at step {item['step']}")
    if item["lr"] != 1e-6 or item["audio_weight"] != 1.0:
        raise RuntimeError("smoke changed the fixed LR or audio CE weight")
    if not item["target_audio_teacher_forcing"] or int(item["audio_tokens"]) <= 0:
        raise RuntimeError("target-audio teacher forcing/audio CE is inactive")

config = json.loads((root / "run_config_step10.json").read_text())
expected = {
    "batch_size": 16,
    "grad_accum_steps": 1,
    "max_frames": 280,
    "val_max_frames": 280,
    "val_batch_size": 4,
    "epochs": 5,
    "lr": 1e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "weight_decay": 0.1,
    "audio_loss_weight": 1.0,
    "initialization": "upstream_hibiki_zero",
    "target_audio_teacher_forcing": True,
    "audio_ce": True,
    "best_metric": "teacher_forced_loss",
    "post_source_transform": None,
    "validation_shuffle": False,
    "sample_manifest_rows": 719_120,
    "sample_manifest_cache_counts": [683_164, 35_956],
    "steps_per_epoch": 44_945,
    "total_steps": 10,
    "frame_bucket": 16,
    "torch_compile_enabled": True,
    "smoke_longest_first": True,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise RuntimeError(f"direct smoke config mismatch for {key}: {config.get(key)}")

receipt_document = json.loads((root / "full_data_receipt.json").read_text())
receipt = receipt_document["full_data_receipt"]
digest = hashlib.sha256(
    json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if digest != receipt_document["sha256"]:
    raise RuntimeError("full-data receipt hash mismatch")
if receipt["strategy"] != "direct_voice_preserving_simultaneous_translation":
    raise RuntimeError("smoke used the wrong training receipt")
if (
    receipt["version"] != 3
    or receipt["epochs"] != 5
    or receipt["steps_per_epoch"] != 44_945
    or receipt["total_steps"] != 224_725
    or receipt["validation_every_steps"] != 9_000
    or receipt["checkpoint_every_steps"] != 9_000
    or receipt["best_metric"] != "teacher_forced_loss"
):
    raise RuntimeError("smoke changed the five-epoch horizon or cadence")
if receipt["streams"]["transform"] is not None:
    raise RuntimeError("smoke applied a post-source transform")
expected_streams = {
    "target_audio": {
        "content": "cached_english_mimi",
        "termination": "native_minus_one_mask",
    },
    "target_text": {
        "content": "cached_ctc_timed_english",
        "termination": "tokenizer_eos",
    },
    "source_audio": {
        "content": "cached_vietnamese_mimi",
        "termination": "explicit_card_eos",
    },
    "transform": None,
}
if receipt["streams"] != expected_streams:
    raise RuntimeError("smoke changed the cached stream/EOS contract")
if receipt["validation"] != {
    "rows": 138,
    "batch_size": 4,
    "max_frames": 280,
    "observed_max_frames": 277,
    "shuffle": False,
}:
    raise RuntimeError("smoke changed the raw validation receipt")
manifest_digest = hashlib.sha256((root / "sample_manifest.jsonl").read_bytes()).hexdigest()
if manifest_digest != receipt["sample_manifest_sha256"]:
    raise RuntimeError("smoke changed frozen training membership")
if config["observed_train_max_frames"] != receipt["observed_max_frames"]:
    raise RuntimeError("smoke training maximum differs from receipt")
if max(int(item["max_frames"]) for item in logs) != receipt["observed_max_frames"]:
    raise RuntimeError("smoke missed the longest raw training row")
if any(int(item["samples"]) != 16 for item in logs):
    raise RuntimeError("smoke did not use physical batch 16")

val_logs = [
    json.loads(line)
    for line in (root / "val_log.jsonl").read_text().splitlines()
    if line
]
if not val_logs or max(int(item["max_frames"]) for item in val_logs) != receipt["validation"]["observed_max_frames"]:
    raise RuntimeError("smoke missed the longest non-shuffled validation row")
if not (root / "model_step000011.safetensors").is_file() or not (
    root / "trainer_step000011.pt"
).is_file():
    raise RuntimeError("save/resume smoke is incomplete")
best = json.loads((root / "best.json").read_text())
best_model = root / best["model"]
if (
    best["metric"] != "teacher_forced_loss"
    or not best_model.is_file()
    or float(best["validation_loss"])
    != min(float(item["loss"]) for item in val_logs)
):
    raise RuntimeError("best-checkpoint promotion is invalid")

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
      --smoke-longest-first --max-steps 10 \
      --val-every 10 --val-batches 1 \
      --save-every 10 --keep-checkpoints 1 --log-every 1 \
      --out-dir "$SMOKE_DIR"
    cp "$SMOKE_DIR/run_config.json" "$SMOKE_DIR/run_config_step10.json"
    "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --smoke-longest-first --max-steps 11 \
      --val-every 10 --val-batches 1 \
      --save-every 10 --keep-checkpoints 1 --log-every 1 \
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
    echo "batch_size=16"
    echo "grad_accum_steps=1"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$SMOKE_DIR/SMOKE_OK"
  echo "SMOKE_OK: $SMOKE_DIR"
}

require_current_smoke() {
  local marker="$SMOKE_DIR/SMOKE_OK"
  [[ -f "$marker" ]] || die "run '$0 smoke' successfully first"
  local smoke_commit
  smoke_commit="$(sed -n 's/^commit=//p' "$marker")"
  [[ "$smoke_commit" == "$(git rev-parse HEAD)" ]] || die "code changed after smoke"
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
        commit_message="Start direct full training run",
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

train() {
  preflight 190
  require_current_smoke
  require_empty_dir "$RUN_DIR"
  common_args
  run_with_sync "$RUN_DIR" fresh "$PYTHON" finetune/train.py \
    "${TRAIN_ARGS[@]}" \
    --val-every 9000 --val-batches 0 \
    --save-every 9000 --keep-checkpoints 2 --log-every 10 \
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
  [[ "$out_dir" == "$RUN_DIR" ]] || die "checkpoint is not in the direct full-run directory"
  common_args
  run_with_sync "$out_dir" resume "$PYTHON" finetune/train.py \
    "${TRAIN_ARGS[@]}" \
    --val-every 9000 --val-batches 0 \
    --save-every 9000 --keep-checkpoints 2 --log-every 10 \
    --resume-checkpoint "$checkpoint" \
    --out-dir "$out_dir"
}

case "${1:-}" in
  setup)
    setup
    ;;
  cache-grounded)
    cache_grounded
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
    echo "usage: $0 setup|cache-grounded|preflight|smoke|train|resume <trainer checkpoint>" >&2
    exit 1
    ;;
esac
