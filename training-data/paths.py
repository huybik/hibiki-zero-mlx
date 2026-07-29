"""Shared data locations. Override with PHOMT_DATA_DIR; audio/cache live outside the repo."""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATASETS_DIR = r"D:\Code\datasets" if os.name == "nt" else "~/datasets"
DATASETS_DIR = Path(os.environ.get("PHOMT_DATA_DIR", _DEFAULT_DATASETS_DIR)).expanduser()
HF_CACHE_DIR = DATASETS_DIR / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
