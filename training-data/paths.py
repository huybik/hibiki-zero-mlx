"""Shared data locations. Override with PHOMT_DATA_DIR; audio/cache live outside the repo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# macOS default lives on the external data disk: the system disk also hosts swap,
# and generation tranches are ~80 GB. HF_HOME already depends on this mount.
if os.name == "nt":
    _DEFAULT_DATASETS_DIR = r"D:\Code\datasets"
elif sys.platform == "darwin":
    _DEFAULT_DATASETS_DIR = "/Volumes/data/datasets"
else:
    _DEFAULT_DATASETS_DIR = "~/datasets"
DATASETS_DIR = Path(os.environ.get("PHOMT_DATA_DIR", _DEFAULT_DATASETS_DIR)).expanduser()
HF_CACHE_DIR = DATASETS_DIR / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
