"""Repository paths, default artifacts and file guards shared by training code."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "weights" / "config.json"
DEFAULT_MODEL_WEIGHT = REPO_ROOT / "weights" / "hibiki-pytorch-77f82164@110.safetensors"
DEFAULT_MIMI_WEIGHT = REPO_ROOT / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
DEFAULT_TOKENIZER = REPO_ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"

# Written by training.prep.cache_codes (Mac) and required by streams.load_rows (H100); kept here
# so the trainer never imports the builder's Mac-only packages.
CACHE_FORMAT = "hibiki_vn_grounded_cache_v2"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def repo_display_path(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_file(path: str | Path, label: str) -> Path:
    path = resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_dir(path: str | Path, label: str) -> Path:
    path = resolve_repo_path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def durable(path: Path) -> None:
    """Flush a file or directory to the drive itself: F_FULLFSYNC on macOS, where fsync stops at
    the drive's cache; fsync on Linux."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        else:
            os.fsync(fd)
    finally:
        os.close(fd)


def exclusive(path: Path, busy: str):
    """Hold an exclusive lock on `path` for as long as the returned file stays open; exit with
    `busy` when another process holds it."""
    lock_file = path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise SystemExit(busy) from None
    return lock_file


def read_json(path: str | Path) -> dict[str, Any]:
    path = require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def cpu_budget() -> float:
    """The CPUs this process may use: the CPUs it may run on, capped by a container's CPU quota
    (cgroup v2 or v1), which neither `nproc` nor `os.cpu_count()` shows."""
    try:
        cpus = float(len(os.sched_getaffinity(0)))
    except AttributeError:
        cpus = float(os.cpu_count() or 1)
    v2, v1 = Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if v2.exists():
        quota, period = v2.read_text().split()
        if quota != "max":
            cpus = min(cpus, int(quota) / int(period))
    elif v1.exists() and (quota := int(v1.read_text())) > 0:
        cpus = min(cpus, quota / int(v1.with_name("cpu.cfs_period_us").read_text()))
    return cpus
