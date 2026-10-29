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

# Written by training.prep.cache_codes (Mac) and required by CachedCodeDataset (H100); kept here
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
    """Flush a file or directory to the drive itself; on macOS fsync stops at the drive's cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
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
