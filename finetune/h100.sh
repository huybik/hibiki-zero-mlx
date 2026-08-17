#!/usr/bin/env bash
# H100 pod setup and launcher for the base-start Vietnamese full-model SFT.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
RECIPE="${HIBIKI_RECIPE:-legacy}"
PILOT="${HIBIKI_PILOT:-0}"
HIGH_DELAY_PILOT="${HIBIKI_HIGH_DELAY_PILOT:-0}"
CONTRASTIVE_PILOT="${HIBIKI_CONTRASTIVE_PILOT:-0}"
ASR_PREADAPT="${HIBIKI_ASR_PREADAPT:-0}"
ASR_ONE_EPOCH="${HIBIKI_ASR_ONE_EPOCH:-0}"
ASR_ASCII="${HIBIKI_ASR_ASCII:-0}"
ASR_TRANSLATION_PILOT="${HIBIKI_ASR_TRANSLATION_PILOT:-0}"
ASR_REPLAY_TRANSLATION_PILOT="${HIBIKI_ASR_REPLAY_TRANSLATION_PILOT:-0}"
BASELINE_PILOT_MANIFEST="$REPO_ROOT/finetune/runs/vi_grounded_v2_pilot/sample_manifest.jsonl"
BASELINE_PILOT_MANIFEST_SHA256=52ef91a79dc09fb6c00a6f800bf087f2228b7c0842ecb2705ac873d3ef3a458f
ASR_PARENT_CHECKPOINT="$REPO_ROOT/finetune/runs/vi_grounded_v2_pilot_vi_asr_ascii_epoch1/best_step003125.safetensors"
ASR_PARENT_CHECKPOINT_SHA256=d37d69103bff8f128b9b69fc9634a018d8ab5c5c58dbb0b5cc98ecf5a26f92ca

die() {
  echo "error: $*" >&2
  exit 1
}

[[ "$PILOT" == 0 || "$PILOT" == 1 ]] || die "HIBIKI_PILOT must be 0 or 1"
[[ "$HIGH_DELAY_PILOT" == 0 || "$HIGH_DELAY_PILOT" == 1 ]] \
  || die "HIBIKI_HIGH_DELAY_PILOT must be 0 or 1"
[[ "$CONTRASTIVE_PILOT" == 0 || "$CONTRASTIVE_PILOT" == 1 ]] \
  || die "HIBIKI_CONTRASTIVE_PILOT must be 0 or 1"
[[ "$ASR_PREADAPT" == 0 || "$ASR_PREADAPT" == 1 ]] \
  || die "HIBIKI_ASR_PREADAPT must be 0 or 1"
[[ "$ASR_ONE_EPOCH" == 0 || "$ASR_ONE_EPOCH" == 1 ]] \
  || die "HIBIKI_ASR_ONE_EPOCH must be 0 or 1"
[[ "$ASR_ASCII" == 0 || "$ASR_ASCII" == 1 ]] \
  || die "HIBIKI_ASR_ASCII must be 0 or 1"
[[ "$ASR_TRANSLATION_PILOT" == 0 || "$ASR_TRANSLATION_PILOT" == 1 ]] \
  || die "HIBIKI_ASR_TRANSLATION_PILOT must be 0 or 1"
[[ "$ASR_REPLAY_TRANSLATION_PILOT" == 0 || "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]] \
  || die "HIBIKI_ASR_REPLAY_TRANSLATION_PILOT must be 0 or 1"
[[ "$PILOT" == 0 || "$RECIPE" == "grounded-v2" ]] \
  || die "HIBIKI_PILOT=1 requires HIBIKI_RECIPE=grounded-v2"
[[ "$HIGH_DELAY_PILOT" == 0 || ( "$PILOT" == 1 && "$RECIPE" == "grounded-v2" ) ]] \
  || die "HIBIKI_HIGH_DELAY_PILOT=1 requires the grounded-v2 pilot"
[[ "$CONTRASTIVE_PILOT" == 0 || "$HIGH_DELAY_PILOT" == 1 ]] \
  || die "HIBIKI_CONTRASTIVE_PILOT=1 requires the high-delay pilot"
[[ "$ASR_PREADAPT" == 0 || "$HIGH_DELAY_PILOT" == 1 ]] \
  || die "HIBIKI_ASR_PREADAPT=1 requires the high-delay pilot cache"
[[ "$ASR_PREADAPT" == 0 || "$CONTRASTIVE_PILOT" == 0 ]] \
  || die "ASR preadaptation and contrastive translation are exclusive"
[[ "$ASR_ONE_EPOCH" == 0 || "$ASR_PREADAPT" == 1 ]] \
  || die "HIBIKI_ASR_ONE_EPOCH=1 requires HIBIKI_ASR_PREADAPT=1"
[[ "$ASR_ASCII" == 0 || ( "$ASR_PREADAPT" == 1 && "$ASR_ONE_EPOCH" == 1 ) ]] \
  || die "HIBIKI_ASR_ASCII=1 requires the one-epoch ASR diagnostic"
[[ "$ASR_TRANSLATION_PILOT" == 0 || (
  "$RECIPE" == grounded-v2 && "$PILOT" == 1 && "$HIGH_DELAY_PILOT" == 0 &&
  "$CONTRASTIVE_PILOT" == 0 && "$ASR_PREADAPT" == 0 && "$ASR_ONE_EPOCH" == 0 &&
  "$ASR_ASCII" == 0
) ]] || die "HIBIKI_ASR_TRANSLATION_PILOT=1 requires the ordinary grounded pilot alone"
[[ "$ASR_REPLAY_TRANSLATION_PILOT" == 0 || (
  "$RECIPE" == grounded-v2 && "$PILOT" == 1 && "$HIGH_DELAY_PILOT" == 0 &&
  "$CONTRASTIVE_PILOT" == 0 && "$ASR_PREADAPT" == 0 && "$ASR_ONE_EPOCH" == 0 &&
  "$ASR_ASCII" == 0 && "$ASR_TRANSLATION_PILOT" == 0
) ]] || die "HIBIKI_ASR_REPLAY_TRANSLATION_PILOT=1 requires the ordinary grounded pilot alone"
case "$RECIPE" in
  legacy)
    SMOKE_DIR="$REPO_ROOT/finetune/runs/h100_smoke"
    RUN_DIR="$REPO_ROOT/finetune/runs/vi_base_full"
    PHOMT_CACHE="finetune/cache/phomt_stream"
    TRAIN_CACHE="finetune/cache/train"
    VAL_CACHE="finetune/cache/validation"
    [[ -z "${HIBIKI_HF_PREFIX:-}" || "$HIBIKI_HF_PREFIX" == full_run ]] \
      || die "legacy recovery prefix must be full_run"
    HIBIKI_HF_PREFIX=full_run
    ;;
  grounded-v2)
    if [[ "$PILOT" == 1 ]]; then
      pilot_namespace=grounded_v2_pilot
      cache_namespace="$pilot_namespace"
      if [[ "$HIGH_DELAY_PILOT" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_high_delay
        cache_namespace="$pilot_namespace"
      fi
      if [[ "$CONTRASTIVE_PILOT" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_high_delay_contrastive
      fi
      if [[ "$ASR_PREADAPT" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_vi_asr_preadapt
      fi
      if [[ "$ASR_ONE_EPOCH" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_vi_asr_preadapt_epoch1
      fi
      if [[ "$ASR_ASCII" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_vi_asr_ascii_epoch1
      fi
      if [[ "$ASR_TRANSLATION_PILOT" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_vi_asr_warmstart
      fi
      if [[ "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]]; then
        pilot_namespace=grounded_v2_pilot_vi_asr_replay
      fi
      SMOKE_DIR="$REPO_ROOT/finetune/runs/h100_smoke_$pilot_namespace"
      RUN_DIR="$REPO_ROOT/finetune/runs/vi_$pilot_namespace"
      PHOMT_CACHE="finetune/cache/phomt_$cache_namespace"
      TRAIN_CACHE="finetune/cache/train_$cache_namespace"
      VAL_CACHE="finetune/cache/validation_$cache_namespace"
      [[ -z "${HIBIKI_HF_PREFIX:-}" || "$HIBIKI_HF_PREFIX" == "$pilot_namespace" ]] \
        || die "pilot recovery prefix must be $pilot_namespace"
      HIBIKI_HF_PREFIX="$pilot_namespace"
    else
      SMOKE_DIR="$REPO_ROOT/finetune/runs/h100_smoke_grounded_v2"
      RUN_DIR="$REPO_ROOT/finetune/runs/vi_grounded_v2"
      PHOMT_CACHE="finetune/cache/phomt_grounded_v2"
      TRAIN_CACHE="finetune/cache/train_grounded_v2"
      VAL_CACHE="finetune/cache/validation_grounded_v2"
      [[ -z "${HIBIKI_HF_PREFIX:-}" || "$HIBIKI_HF_PREFIX" == grounded_v2 ]] \
        || die "full grounded-v2 recovery prefix must be grounded_v2"
      HIBIKI_HF_PREFIX=grounded_v2
    fi
    ;;
  *)
    echo "error: HIBIKI_RECIPE must be legacy or grounded-v2" >&2
    exit 1
    ;;
esac
export HIBIKI_HF_PREFIX

TARGET_DELAY_MIN_RATIO=0
TARGET_DELAY_MAX_RATIO=0.5
TARGET_DELAY_SEED=1234
if [[ "$HIGH_DELAY_PILOT" == 1 ]]; then
  TARGET_DELAY_MIN_RATIO=0.75
  TARGET_DELAY_MAX_RATIO=1.0
fi

if [[ "$RECIPE" == "grounded-v2" && "$PILOT" == 0 ]]; then
  [[ -z "${HIBIKI_MAX_SAMPLES:-}" ]] || die "full grounded-v2 rejects HIBIKI_MAX_SAMPLES"
  [[ -z "${HIBIKI_MAX_STEPS:-}" ]] || die "full grounded-v2 rejects HIBIKI_MAX_STEPS"
  [[ -z "${HIBIKI_CACHE_SAMPLE_SHARDS:-}" ]] \
    || die "full grounded-v2 rejects HIBIKI_CACHE_SAMPLE_SHARDS"
fi

require_python() {
  [[ -x "$PYTHON" ]] || die "run '$0 setup' first"
}

require_baseline_pilot_manifest() {
  [[ "$HIGH_DELAY_PILOT" == 1 || "$ASR_TRANSLATION_PILOT" == 1 || \
    "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]] || return
  [[ -f "$BASELINE_PILOT_MANIFEST" ]] || die "missing baseline pilot sample manifest"
  command -v sha256sum >/dev/null || die "sha256sum is required"
  local actual
  actual="$(sha256sum "$BASELINE_PILOT_MANIFEST" | cut -d' ' -f1)"
  [[ "$actual" == "$BASELINE_PILOT_MANIFEST_SHA256" ]] \
    || die "baseline pilot sample manifest hash mismatch: $actual"
}

require_asr_parent_checkpoint() {
  [[ "$ASR_TRANSLATION_PILOT" == 1 || "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]] || return
  [[ -f "$ASR_PARENT_CHECKPOINT" ]] || die "missing qualified ASR parent checkpoint"
}

require_cuda_driver() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  [[ "$driver_version" =~ ^[0-9]+\. ]] || die "invalid NVIDIA driver version: $driver_version"
  [[ "$(printf '%s\n' 580 "$driver_version" | sort -V | head -n 1)" == "580" ]] \
    || die "CUDA 13.x minor-version compatibility requires NVIDIA driver 580 or newer, got $driver_version"
}

require_h100_gpu() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ -n "$gpu_names" && "$gpu_names" != *$'\n'* ]] || die "exactly one visible GPU is required"
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
  [[ "$(uname -s)" == "Linux" ]] || die "setup must run on the Linux H100 pod"
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
  if [[ "$RECIPE" == "grounded-v2" ]]; then
    "$PYTHON" -m pip install num2words==0.5.14 transformers==5.14.1
  fi
  "$PYTHON" -m pip install --no-deps moshi==0.2.13
  "$PYTHON" -m pip freeze > "$VENV/h100-freeze.txt"
  echo "H100 environment installed in $VENV"
}

cache_grounded() {
  [[ "$RECIPE" == "grounded-v2" ]] \
    || die "cache-grounded requires HIBIKI_RECIPE=grounded-v2"
  [[ "$CONTRASTIVE_PILOT" == 0 ]] \
    || die "the contrastive pilot reuses the verified high-delay caches"
  [[ "$ASR_PREADAPT" == 0 ]] \
    || die "ASR preadaptation reuses the verified high-delay caches"
  [[ "$ASR_TRANSLATION_PILOT" == 0 ]] \
    || die "the ASR warm-start pilot reuses the verified ordinary caches"
  [[ "$ASR_REPLAY_TRANSLATION_PILOT" == 0 ]] \
    || die "the ASR-replay pilot reuses the verified ordinary caches"
  require_python
  require_baseline_pilot_manifest
  require_cuda_driver
  require_h100_gpu
  [[ -f finetune/pairs/train.jsonl ]] || die "missing finetune/pairs/train.jsonl"
  [[ -f finetune/pairs/validation.jsonl ]] || die "missing finetune/pairs/validation.jsonl"

  local workers="${HIBIKI_CACHE_WORKERS:-4}"
  [[ "$workers" =~ ^[1-9][0-9]*$ ]] || die "HIBIKI_CACHE_WORKERS must be a positive integer"
  local sample_shards=0
  if [[ "$PILOT" == 1 ]]; then
    sample_shards=104
  fi
  local delay_args=()
  if [[ "$HIGH_DELAY_PILOT" == 1 ]]; then
    delay_args=(
      --target-delay-ratio "$TARGET_DELAY_MAX_RATIO"
      --target-delay-min-ratio "$TARGET_DELAY_MIN_RATIO"
      --seed "$TARGET_DELAY_SEED"
    )
  fi
  local cache_args=(
    --recipe grounded-v2 --device cuda --profile h100 --out-dir "$PHOMT_CACHE"
    "${delay_args[@]}"
  )
  if (( sample_shards > 0 )); then
    cache_args+=(--sample-shards "$sample_shards")
  fi
  local pids=()
  local worker
  for ((worker = 0; worker < workers; worker++)); do
    "$PYTHON" finetune/cache_phomt_stream.py \
      "${cache_args[@]}" \
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

  if (( sample_shards == 0 )); then
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
  fi

  "$PYTHON" finetune/cache_codes.py \
    --recipe grounded-v2 --device cuda --pairs finetune/pairs/train.jsonl \
    --out-dir "$TRAIN_CACHE" "${delay_args[@]}"
  "$PYTHON" finetune/cache_codes.py \
    --recipe grounded-v2 --device cuda --pairs finetune/pairs/validation.jsonl \
    --out-dir "$VAL_CACHE" "${delay_args[@]}"
  if (( sample_shards > 0 )); then
    echo "Grounded-v2 pilot caches complete: $sample_shards sampled PhoMT shards"
  else
    echo "Grounded-v2 caches complete"
  fi
}

preflight() {
  local minimum_free_gib="${1:-190}"
  require_python
  require_baseline_pilot_manifest
  require_asr_parent_checkpoint
  if [[ "$RECIPE" == "grounded-v2" ]]; then
    source_gap_args
  fi
  require_cuda_driver
  git diff --quiet || die "tracked worktree changes must be committed before training"
  git diff --cached --quiet || die "staged changes must be committed before training"

  local profile
  profile="$("$PYTHON" - \
    "$REPO_ROOT" "$minimum_free_gib" "$RECIPE" "$PILOT" "$HIGH_DELAY_PILOT" \
    "$CONTRASTIVE_PILOT" "$ASR_PREADAPT" "$ASR_TRANSLATION_PILOT" \
    "$ASR_REPLAY_TRANSLATION_PILOT" \
    "$TARGET_DELAY_MIN_RATIO" "$TARGET_DELAY_MAX_RATIO" "$TARGET_DELAY_SEED" \
    "$PHOMT_CACHE" "$TRAIN_CACHE" "$VAL_CACHE" <<'PY'
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
recipe = sys.argv[3]
pilot = bool(int(sys.argv[4]))
high_delay_pilot = bool(int(sys.argv[5]))
contrastive_pilot = bool(int(sys.argv[6]))
source_asr = bool(int(sys.argv[7]))
asr_translation_pilot = bool(int(sys.argv[8]))
asr_replay_translation_pilot = bool(int(sys.argv[9]))
target_delay = {
    "min_ratio": float(sys.argv[10]),
    "max_ratio": float(sys.argv[11]),
    "seed": int(sys.argv[12]),
}
phomt_cache, train_cache, val_cache = sys.argv[13:16]

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
if asr_replay_translation_pilot:
    if memory_gib < 75:
        raise RuntimeError(
            f"ASR replay requires at least 75 GiB GPU memory, got {memory_gib:.1f} GiB"
        )
    batch_size, grad_accum = 8, 2
elif asr_translation_pilot:
    if memory_gib < 90:
        raise RuntimeError(
            f"ASR translation pilots require at least 90 GiB GPU memory, got {memory_gib:.1f} GiB"
        )
    batch_size, grad_accum = 16, 1
elif (contrastive_pilot or source_asr) and memory_gib >= 75:
    batch_size, grad_accum = 4, 4
elif high_delay_pilot and memory_gib >= 75:
    batch_size, grad_accum = 8, 2
elif memory_gib >= 90:
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

expected_shards = (
    {
        phomt_cache: 1377,
        train_cache: 46,
        val_cache: 5,
    }
    if recipe == "legacy"
    else {
        phomt_cache: 104 if pilot else 1377,
        train_cache: 46,
        val_cache: 5,
    }
)
for relative, expected in expected_shards.items():
    paths = sorted((root / relative).glob("shard_*.pt"))
    if len(paths) != expected:
        raise RuntimeError(f"{relative}: expected {expected} shards, got {len(paths)}")
    if any(path.stat().st_size == 0 for path in paths):
        raise RuntimeError(f"{relative}: found an empty shard")
    if recipe == "grounded-v2":
        payload = torch.load(paths[0], map_location="cpu")
        if payload.get("format") != "hibiki_vn_grounded_cache_v2":
            raise RuntimeError(f"{relative}: not a grounded-v2 cache")
        if not payload.get("samples"):
            raise RuntimeError(f"{relative}: first grounded-v2 shard is empty")
        if float(payload.get("alignment_min_score") or 0) != 0.5:
            raise RuntimeError(f"{relative}: CTC threshold is not 0.5")
        if high_delay_pilot and payload.get("target_delay") != target_delay:
            raise RuntimeError(f"{relative}: not a deterministic 75--100% delay cache")
        if any(sample.get("text_timing") != "wav2vec2_ctc_word_v1" for sample in payload["samples"]):
            raise RuntimeError(f"{relative}: missing word-aligned text timing")
        if any(float(sample.get("alignment_score") or 0) < 0.5 for sample in payload["samples"]):
            raise RuntimeError(f"{relative}: contains a below-threshold CTC alignment")
        if high_delay_pilot:
            for path in paths[1:]:
                if torch.load(path, map_location="cpu").get("target_delay") != target_delay:
                    raise RuntimeError(f"{relative}: target-delay policy mismatch in {path.name}")

if recipe == "grounded-v2" and pilot:
    phomt_rows = sum(
        len(torch.load(path, map_location="cpu")["samples"])
        for path in sorted((root / phomt_cache).glob("shard_*.pt"))
    )
    fleurs_rows = sum(
        len(torch.load(path, map_location="cpu")["samples"])
        for path in sorted((root / train_cache).glob("shard_*.pt"))
    )
    if phomt_rows < 47_500:
        raise RuntimeError(f"pilot PhoMT cache needs at least 47,500 rows, got {phomt_rows}")
    if fleurs_rows < 1:
        raise RuntimeError("pilot FLEURS train cache is empty")

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
  if [[ "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]]; then
    export NO_TORCH_COMPILE=1
  else
    export NO_TORCH_COMPILE=
  fi
  export HIBIKI_FRAME_BUCKET=16
  echo "Preflight passed: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "GPU=${GPU_GIB}GiB host=${HOST_GIB}GiB free_disk=${FREE_GIB}GiB batch=$BATCH_SIZE accum=$GRAD_ACCUM_STEPS"
}

source_gap_args() {
  local bleu_gap="${HIBIKI_MIN_SOURCE_BLEU_GAP:-}"
  local chrf_gap="${HIBIKI_MIN_SOURCE_CHRF_GAP:-}"
  if [[ "$RECIPE" == "grounded-v2" ]]; then
    [[ -n "$bleu_gap" ]] || die "set calibrated HIBIKI_MIN_SOURCE_BLEU_GAP"
    [[ -n "$chrf_gap" ]] || die "set calibrated HIBIKI_MIN_SOURCE_CHRF_GAP"
  else
    bleu_gap="${bleu_gap:-0}"
    chrf_gap="${chrf_gap:-0}"
  fi
  "$PYTHON" - "$bleu_gap" "$chrf_gap" <<'PY'
import math
import sys

for name, value in zip(("BLEU", "chrF"), sys.argv[1:], strict=True):
    if not math.isfinite(float(value)):
        raise ValueError(f"source {name} gap must be finite")
PY
  SOURCE_GAP_ARGS=(
    --min-source-bleu-gap "$bleu_gap"
    --min-source-chrf-gap "$chrf_gap"
  )
}

common_args() {
  local max_steps="${HIBIKI_MAX_STEPS:-0}"
  local max_frames=280
  local val_max_frames=0
  local val_batch_size=8
  local sort_args=(--sort-by-length)
  if [[ "$PILOT" == 1 ]]; then
    max_steps=1000
  fi
  if [[ "$HIGH_DELAY_PILOT" == 1 ]]; then
    max_frames=480
    val_max_frames=704
    val_batch_size=1
    sort_args=(--no-sort-by-length)
  fi
  if [[ "$ASR_PREADAPT" == 1 ]]; then
    max_frames=672
    val_max_frames=640
  fi
  if [[ "$ASR_ONE_EPOCH" == 1 ]]; then
    max_steps=3125
  fi
  if [[ "$ASR_TRANSLATION_PILOT" == 1 || "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]]; then
    sort_args=(--no-sort-by-length)
  fi
  [[ "$max_steps" =~ ^[0-9]+$ ]] || die "HIBIKI_MAX_STEPS must be a non-negative integer"
  source_gap_args
  EVAL_EVERY=9000
  TRAIN_ARGS=(
    --device cuda
    --model-weight weights/hibiki-pytorch-77f82164@110.safetensors
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --max-steps "$max_steps"
    --max-frames "$max_frames"
    --val-max-frames "$val_max_frames"
    "${sort_args[@]}"
    --seed 42
    --val-batch-size "$val_batch_size"
    --eval-pairs finetune/pairs/val128.jsonl
    --eval-batch-size 8
    --eval-text-temp 0.4
    --eval-duration-column vi_duration_s
    "${SOURCE_GAP_ARGS[@]}"
  )
  if [[ "$RECIPE" == "legacy" ]]; then
    TRAIN_ARGS+=(
      --cache-dir "$PHOMT_CACHE" "$TRAIN_CACHE"
      --val-cache-dir "$VAL_CACHE"
      --epochs 2
      --text-pad-loss-weight 0.5
    )
  else
    EVAL_EVERY=3000
    if [[ "$PILOT" == 1 ]]; then
      EVAL_EVERY=250
    fi
    if [[ "$ASR_ONE_EPOCH" == 1 ]]; then
      EVAL_EVERY=2000
    fi
    TRAIN_ARGS+=(
      --cache-dir "$PHOMT_CACHE" "$TRAIN_CACHE"
      --val-cache-dir "$VAL_CACHE"
      --epochs 1
      --text-pad-loss-weight 0.05
      --first-content-loss-weight 1.0
      --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.1
      --eval-at-start
    )
    if [[ "$HIGH_DELAY_PILOT" == 0 && "$ASR_TRANSLATION_PILOT" == 0 && \
      "$ASR_REPLAY_TRANSLATION_PILOT" == 0 ]]; then
      TRAIN_ARGS+=(--cache-weights 0.95 0.05)
    fi
    if [[ "$PILOT" == 1 ]]; then
      TRAIN_ARGS+=(
        --text-pad-mode prefix
        --audio-loss-weight 0
        --mask-target-audio-input
        --persist-sample-manifest
      )
      if [[ "$HIGH_DELAY_PILOT" == 1 ]]; then
        TRAIN_ARGS+=(
          --input-sample-manifest "$BASELINE_PILOT_MANIFEST"
          --input-sample-manifest-sha256 "$BASELINE_PILOT_MANIFEST_SHA256"
          --expected-target-delay-min-ratio "$TARGET_DELAY_MIN_RATIO"
          --expected-target-delay-max-ratio "$TARGET_DELAY_MAX_RATIO"
          --expected-target-delay-seed "$TARGET_DELAY_SEED"
        )
        if [[ "$CONTRASTIVE_PILOT" == 1 ]]; then
          TRAIN_ARGS+=(
            --contrastive-source-weight 1.0
            --contrastive-source-margin 0.5
          )
        fi
        if [[ "$ASR_PREADAPT" == 1 ]]; then
          TRAIN_ARGS+=(
            --source-asr-pretrain
            --eval-reference-column text_vi
            --eval-tail-s 24
            --min-correct-chrf 50
            --max-correct-wer 0.6
          )
          if [[ "$ASR_ASCII" == 1 ]]; then
            TRAIN_ARGS+=(--source-asr-ascii)
          fi
        fi
      elif [[ "$ASR_TRANSLATION_PILOT" == 1 ]]; then
        TRAIN_ARGS+=(
          --input-sample-manifest "$BASELINE_PILOT_MANIFEST"
          --input-sample-manifest-sha256 "$BASELINE_PILOT_MANIFEST_SHA256"
          --init-checkpoint "$ASR_PARENT_CHECKPOINT"
          --init-checkpoint-sha256 "$ASR_PARENT_CHECKPOINT_SHA256"
        )
      elif [[ "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]]; then
        TRAIN_ARGS+=(
          --input-sample-manifest "$BASELINE_PILOT_MANIFEST"
          --input-sample-manifest-sha256 "$BASELINE_PILOT_MANIFEST_SHA256"
          --init-checkpoint "$ASR_PARENT_CHECKPOINT"
          --init-checkpoint-sha256 "$ASR_PARENT_CHECKPOINT_SHA256"
          --source-asr-ascii
          --source-asr-replay-weight 1.0
          --source-asr-replay-batch-size 4
          --source-asr-replay-max-frames 434
        )
      else
        TRAIN_ARGS+=(--max-samples 50000)
      fi
    else
      TRAIN_ARGS+=(
        --max-samples 0
        --text-pad-mode all
        --audio-loss-weight 1
      )
    fi
  fi
}

stop_monitor() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

check_smoke_outputs() {
  "$PYTHON" - \
    "$SMOKE_DIR" "$HIGH_DELAY_PILOT" "$BASELINE_PILOT_MANIFEST_SHA256" \
    "$CONTRASTIVE_PILOT" "$ASR_PREADAPT" "$ASR_ASCII" \
    "$ASR_TRANSLATION_PILOT" "$ASR_REPLAY_TRANSLATION_PILOT" \
    "$ASR_PARENT_CHECKPOINT" \
    "$ASR_PARENT_CHECKPOINT_SHA256" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
high_delay_pilot = bool(int(sys.argv[2]))
baseline_manifest_sha256 = sys.argv[3]
contrastive_pilot = bool(int(sys.argv[4]))
source_asr = bool(int(sys.argv[5]))
source_asr_ascii = bool(int(sys.argv[6]))
asr_translation_pilot = bool(int(sys.argv[7]))
asr_replay_translation_pilot = bool(int(sys.argv[8]))
asr_parent_checkpoint = sys.argv[9]
asr_parent_checkpoint_sha256 = sys.argv[10]
logs = [json.loads(line) for line in (root / "train_log.jsonl").read_text().splitlines() if line]
if not logs or logs[-1]["step"] != 11:
    raise RuntimeError("resume smoke did not reach step 11")
for item in logs:
    for key in ("loss", "audio_loss", "text_loss", "lr"):
        if not math.isfinite(float(item[key])):
            raise RuntimeError(f"non-finite {key} at step {item['step']}")

if high_delay_pilot:
    config = json.loads((root / "run_config_step10.json").read_text())
    expected_batch_size = 4 if contrastive_pilot or source_asr else 8
    expected_grad_accum = 4 if contrastive_pilot or source_asr else 2
    expected = {
        "batch_size": expected_batch_size,
        "grad_accum_steps": expected_grad_accum,
        "max_frames": 672 if source_asr else 480,
        "val_max_frames": 640 if source_asr else 704,
        "val_batch_size": 1,
        "max_samples": 0,
        "cache_weights": None,
        "sort_by_length": False,
        "smoke_longest_first": True,
        "input_sample_manifest_sha256": baseline_manifest_sha256,
        "sample_manifest_sha256": baseline_manifest_sha256,
        "expected_target_delay_min_ratio": 0.75,
        "expected_target_delay_max_ratio": 1.0,
        "expected_target_delay_seed": 1234,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"high-delay smoke config mismatch for {key}: {config.get(key)}")
    manifest_digest = hashlib.sha256((root / "sample_manifest.jsonl").read_bytes()).hexdigest()
    if manifest_digest != baseline_manifest_sha256:
        raise RuntimeError("high-delay smoke changed authoritative sample membership")
    if max(int(item["max_batch_size"]) for item in logs) != expected_batch_size:
        raise RuntimeError(
            f"high-delay smoke did not exercise physical batch {expected_batch_size}"
        )
    observed_train_max_frames = int(config.get("observed_train_max_frames", 0))
    if not 0 < observed_train_max_frames <= int(config["max_frames"]):
        raise RuntimeError("high-delay observed training maximum exceeds its frame cap")
    if max(int(item["max_frames"]) for item in logs) != observed_train_max_frames:
        raise RuntimeError("high-delay smoke did not exercise the longest manifest row")
    if contrastive_pilot:
        expected_contrastive = {
            "contrastive_source_weight": 1.0,
            "contrastive_source_margin": 0.5,
        }
        for key, value in expected_contrastive.items():
            if config.get(key) != value:
                raise RuntimeError(f"contrastive smoke config mismatch for {key}")
        mapping_path = root / "source_derangement.json"
        document = json.loads(mapping_path.read_text())
        mapping = document["mapping"]
        digest = hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != document["sha256"] or digest != config.get("source_derangement_sha256"):
            raise RuntimeError("contrastive source derangement hash mismatch")
        pairs = mapping["pairs"]
        if len(pairs) != 50_000:
            raise RuntimeError("contrastive source derangement has the wrong cohort size")
        if sorted(pair["source_index"] for pair in pairs) != list(range(len(pairs))):
            raise RuntimeError("contrastive source donors are not a permutation")
        if any(pair["source_id"] == pair["target_id"] for pair in pairs):
            raise RuntimeError("contrastive source derangement contains a duplicate-id donor")
        for item in logs:
            for key in (
                "contrastive_source_loss",
                "source_text_nll_gap",
                "contrastive_active_fraction",
            ):
                if not math.isfinite(float(item[key])):
                    raise RuntimeError(f"non-finite {key} at step {item['step']}")
        if max(float(item["contrastive_active_fraction"]) for item in logs) <= 0:
            raise RuntimeError("contrastive source margin was never active")
    if source_asr:
        expected_asr = {
            "source_asr_pretrain": True,
            "source_asr_ascii": source_asr_ascii,
            "eval_reference_column": "text_vi",
            "eval_tail_s": 24.0,
            "min_correct_chrf": 50.0,
            "max_correct_wer": 0.6,
        }
        for key, value in expected_asr.items():
            if config.get(key) != value:
                raise RuntimeError(f"source-ASR smoke config mismatch for {key}")
        document = json.loads((root / "source_asr.json").read_text())
        payload = document["source_asr"]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != document["sha256"] or digest != config.get("source_asr_sha256"):
            raise RuntimeError("source-ASR policy hash mismatch")
        expected_strategy = (
            "full_sentence_ascii_vi_asr_after_source_eos"
            if source_asr_ascii
            else "full_sentence_vi_asr_after_source_eos"
        )
        if payload.get("strategy") != expected_strategy:
            raise RuntimeError("source-ASR strategy mismatch")
        observed_max_frames = int(payload.get("observed_max_frames", 0))
        if payload.get("rows") != 50_000 or not 0 < observed_max_frames <= 672:
            raise RuntimeError("source-ASR cohort shape mismatch")
        if not source_asr_ascii and observed_max_frames != 668:
            raise RuntimeError("source-ASR raw-text shape mismatch")

if asr_translation_pilot:
    config = json.loads((root / "run_config_step10.json").read_text())
    expected = {
        "batch_size": 8,
        "grad_accum_steps": 2,
        "max_frames": 280,
        "val_max_frames": 0,
        "val_batch_size": 8,
        "max_samples": 0,
        "cache_weights": None,
        "sort_by_length": False,
        "smoke_longest_first": True,
        "input_sample_manifest_sha256": baseline_manifest_sha256,
        "sample_manifest_sha256": baseline_manifest_sha256,
        "init_checkpoint": asr_parent_checkpoint,
        "init_checkpoint_sha256": asr_parent_checkpoint_sha256,
        "source_asr_pretrain": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"ASR warm-start smoke config mismatch for {key}")
    manifest_digest = hashlib.sha256((root / "sample_manifest.jsonl").read_bytes()).hexdigest()
    if manifest_digest != baseline_manifest_sha256:
        raise RuntimeError("ASR warm-start changed authoritative sample membership")
    if max(int(item["max_batch_size"]) for item in logs) != 16:
        raise RuntimeError("ASR warm-start smoke did not exercise physical batch 16")
    observed_max_frames = int(config.get("observed_train_max_frames", 0))
    if not 0 < observed_max_frames <= 280:
        raise RuntimeError("ASR warm-start training maximum exceeds its frame cap")
    if max(int(item["max_frames"]) for item in logs) != observed_max_frames:
        raise RuntimeError("ASR warm-start smoke did not exercise the longest row")

if asr_replay_translation_pilot:
    config = json.loads((root / "run_config_step10.json").read_text())
    expected = {
        "batch_size": 16,
        "grad_accum_steps": 1,
        "max_frames": 280,
        "val_max_frames": 0,
        "val_batch_size": 8,
        "max_samples": 0,
        "cache_weights": None,
        "sort_by_length": False,
        "smoke_longest_first": True,
        "input_sample_manifest_sha256": baseline_manifest_sha256,
        "sample_manifest_sha256": baseline_manifest_sha256,
        "init_checkpoint": asr_parent_checkpoint,
        "init_checkpoint_sha256": asr_parent_checkpoint_sha256,
        "source_asr_pretrain": False,
        "source_asr_ascii": True,
        "source_asr_replay_weight": 1.0,
        "source_asr_replay_batch_size": 4,
        "source_asr_replay_max_frames": 434,
        "observed_source_asr_replay_max_frames": 434,
        "torch_compile_enabled": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"ASR-replay smoke config mismatch for {key}")
    manifest_digest = hashlib.sha256((root / "sample_manifest.jsonl").read_bytes()).hexdigest()
    if manifest_digest != baseline_manifest_sha256:
        raise RuntimeError("ASR replay changed authoritative translation membership")
    document = json.loads((root / "source_asr_replay.json").read_text())
    payload = document["source_asr"]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != document["sha256"] or digest != config.get("source_asr_replay_sha256"):
        raise RuntimeError("ASR-replay policy hash mismatch")
    if (
        payload.get("strategy") != "full_sentence_ascii_vi_asr_after_source_eos"
        or payload.get("rows") != 50_000
        or payload.get("observed_max_frames") != 434
    ):
        raise RuntimeError("ASR-replay cohort shape mismatch")
    if max(int(item["max_batch_size"]) for item in logs) != 8:
        raise RuntimeError("ASR-replay smoke did not exercise translation batch 8")
    observed_train_max_frames = int(config.get("observed_train_max_frames", 0))
    if not 0 < observed_train_max_frames <= 280:
        raise RuntimeError("ASR-replay translation maximum exceeds its frame cap")
    if max(int(item["max_frames"]) for item in logs) != observed_train_max_frames:
        raise RuntimeError("ASR-replay smoke did not exercise the longest translation row")
    for item in logs:
        if not math.isfinite(float(item["source_asr_replay_loss"])):
            raise RuntimeError(f"non-finite ASR-replay loss at step {item['step']}")
        if int(item["samples"]) != 16 or int(item["microbatches"]) != 2:
            raise RuntimeError("ASR-replay smoke did not preserve effective batch 16")
        if int(item["source_asr_replay_samples"]) != 4:
            raise RuntimeError("ASR-replay smoke did not use one replay batch per step")
    if max(int(item["source_asr_replay_max_frames"]) for item in logs) != 434:
        raise RuntimeError("ASR-replay smoke did not exercise the longest replay row")

eval_dir = root / "standalone_eval_step10"
metrics = json.loads((eval_dir / "metrics.json").read_text())
with (eval_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
    predictions = list(csv.DictReader(handle))
if metrics["correct"]["num_predictions"] != 8 or len(predictions) != 8:
    raise RuntimeError("standalone smoke eval did not produce eight predictions")
if not (eval_dir / "correct" / "metrics.json").is_file():
    raise RuntimeError("standalone smoke eval is missing correct-condition artifacts")
if not (eval_dir / "shuffled" / "metrics.json").is_file():
    raise RuntimeError("standalone smoke eval is missing shuffled-condition artifacts")

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
    local smoke_data_args=()
    if [[ "$HIGH_DELAY_PILOT" == 1 || "$ASR_TRANSLATION_PILOT" == 1 || \
      "$ASR_REPLAY_TRANSLATION_PILOT" == 1 ]]; then
      smoke_data_args=(--smoke-longest-first)
    fi
    local smoke_lr=(--lr-schedule "1e-4@0" --warmup-steps 500 --text-weight-schedule "5@0")
    if [[ "$RECIPE" == "grounded-v2" ]]; then
      smoke_lr=(--lr-schedule "1e-5@0" --cosine-lr-end 1e-6 --warmup-steps 1000 --text-weight-schedule "2@0")
      if [[ "$PILOT" == 1 ]]; then
        smoke_lr=(--lr-schedule "1e-5@0" --cosine-lr-end 1e-6 --warmup-steps 100 --text-weight-schedule "2@0")
      fi
    fi
    "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      "${smoke_lr[@]}" \
      "${smoke_data_args[@]}" \
      --max-steps 10 \
      --val-every 10 --val-batches 1 \
      --eval-every 10 --eval-limit 8 \
      --save-every 10 --keep-checkpoints 1 --log-every 1 \
      --out-dir "$SMOKE_DIR"

    cp "$SMOKE_DIR/run_config.json" "$SMOKE_DIR/run_config_step10.json"
    local smoke_eval_args=()
    if [[ "$ASR_PREADAPT" == 1 ]]; then
      smoke_eval_args=(--reference-column text_vi --tail-s 24)
      if [[ "$ASR_ASCII" == 1 ]]; then
        smoke_eval_args+=(--ascii-reference)
      fi
    fi
    "$PYTHON" finetune/eval.py \
      --device cuda --dtype float32 \
      --model-weight weights/hibiki-pytorch-77f82164@110.safetensors \
      --checkpoint "$SMOKE_DIR/model_step000010.safetensors" \
      --pairs finetune/pairs/val128.jsonl \
      --limit 8 --batch-size 8 --text-temp 0.4 \
      --stop-on-eos --text-only --seed 42 \
      "${smoke_eval_args[@]}" \
      --out-dir "$SMOKE_DIR/standalone_eval_step10"

    "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      "${smoke_lr[@]}" \
      "${smoke_data_args[@]}" \
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
    raise RuntimeError("disaster-recovery model repo must be public")
prefix = sys.argv[5].strip("/") + "/"
remote_files = [
    item.rfilename.removeprefix(prefix)
    for item in info.siblings
    if item.rfilename.startswith(prefix)
]
if sys.argv[2] == "fresh" and remote_files:
    raise RuntimeError(f"fresh training requires an empty {prefix} recovery prefix")
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
  if [[ "$RECIPE" == "legacy" ]]; then
    run_with_sync "$RUN_DIR" fresh "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
      --text-weight-schedule "5@0,2@0.6" \
      --val-every 2000 \
      --eval-every "$EVAL_EVERY" --eval-limit 128 \
      --save-every 3000 --keep-checkpoints 2 --log-every 10 \
      --out-dir "$RUN_DIR"
  else
    local grounded_warmup=1000
    local grounded_val_every=1000
    local grounded_save_every=3000
    if [[ "$PILOT" == 1 ]]; then
      grounded_warmup=100
      grounded_val_every=250
    fi
    if [[ "$ASR_ONE_EPOCH" == 1 ]]; then
      grounded_val_every=1000
      grounded_save_every=2000
    fi
    run_with_sync "$RUN_DIR" fresh "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-5@0" --cosine-lr-end 1e-6 --warmup-steps "$grounded_warmup" \
      --text-weight-schedule "2@0" \
      --val-every "$grounded_val_every" \
      --eval-every "$EVAL_EVERY" --eval-limit 128 \
      --save-every "$grounded_save_every" --keep-checkpoints 2 --log-every 10 \
      --out-dir "$RUN_DIR"
  fi
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
  if [[ "$RECIPE" == "legacy" ]]; then
    run_with_sync "$out_dir" resume "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-4@0,3e-5@0.5" --warmup-steps 500 \
      --text-weight-schedule "5@0,2@0.6" \
      --val-every 2000 \
      --eval-every "$EVAL_EVERY" --eval-limit 128 \
      --save-every 3000 --keep-checkpoints 2 --log-every 10 \
      --resume-checkpoint "$checkpoint" \
      --out-dir "$out_dir"
  else
    local grounded_warmup=1000
    local grounded_val_every=1000
    local grounded_save_every=3000
    if [[ "$PILOT" == 1 ]]; then
      grounded_warmup=100
      grounded_val_every=250
    fi
    if [[ "$ASR_ONE_EPOCH" == 1 ]]; then
      grounded_val_every=1000
      grounded_save_every=2000
    fi
    run_with_sync "$out_dir" resume "$PYTHON" finetune/train.py \
      "${TRAIN_ARGS[@]}" \
      --lr-schedule "1e-5@0" --cosine-lr-end 1e-6 --warmup-steps "$grounded_warmup" \
      --text-weight-schedule "2@0" \
      --val-every "$grounded_val_every" \
      --eval-every "$EVAL_EVERY" --eval-limit 128 \
      --save-every "$grounded_save_every" --keep-checkpoints 2 --log-every 10 \
      --resume-checkpoint "$checkpoint" \
      --out-dir "$out_dir"
  fi
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
