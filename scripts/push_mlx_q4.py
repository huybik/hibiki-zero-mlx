#!/usr/bin/env python
"""Repush the q4 weights to the HF Hub (patch/verify shims already live there)."""
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "huybik/hibiki-zero-3b-mlx-q4"
HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)

# left side = path_in_repo on the Hub (kept stable); right side = local source.
FILES = {
    "hibiki.q4.safetensors": HERE / "weights" / "hibiki.q4.safetensors",
}

ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src))
       for dst, src in FILES.items()]

HfApi().create_commit(
    repo_id=REPO,
    repo_type="model",
    operations=ops,
    commit_message="Regenerate q4 weights",
)
print("pushed:", ", ".join(FILES))
