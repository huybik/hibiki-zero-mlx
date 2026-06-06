#!/usr/bin/env python
"""Repush the new q4 weights + coupled patch/verify to the HF Hub in one commit."""
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "huybik/hibiki-zero-3b-mlx-q4"
HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)

# left side = path_in_repo on the Hub (kept stable); right side = local source.
FILES = {
    "hibiki.q4.safetensors": HERE / "weights" / "hibiki.q4.safetensors",
    "mlx_hibiki_patch.py": HERE / "src" / "mlx_hibiki_patch.py",
    "verify_mlx_q4.py": HERE / "scripts" / "verify_mlx_q4.py",
}

ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src))
       for dst, src in FILES.items()]

HfApi().create_commit(
    repo_id=REPO,
    repo_type="model",
    operations=ops,
    commit_message="Refactor depformer-LayerNorm fix (slices.{i}.norm); regenerate q4",
)
print("pushed:", ", ".join(FILES))
