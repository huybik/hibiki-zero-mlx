#!/usr/bin/env python
"""Push the MLX q4 model artifacts to the HF Hub."""
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "anquachdev/hbk-zero-3b-mlx-q4"
HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)

# left side = path_in_repo on the Hub (kept stable); right side = local source.
FILES = {
    "config.json": HERE / "weights" / "config.json",
    "hibiki.q4.safetensors": HERE / "weights" / "hibiki.q4.safetensors",
    "mimi-pytorch-e351c8d8@125.safetensors": (
        HERE / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
    ),
    "tokenizer_spm_48k_multi6_2.model": HERE / "weights" / "tokenizer_spm_48k_multi6_2.model",
    "mlx_hibiki_patch.py": HERE / "src" / "mlx_hibiki_patch.py",
    "verify_mlx_q4.py": HERE / "scripts" / "verify_mlx_q4.py",
}

missing = [str(src) for src in FILES.values() if not src.exists()]
if missing:
    raise FileNotFoundError("missing upload artifact(s): " + ", ".join(missing))

ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src))
       for dst, src in FILES.items()]

HfApi().create_commit(
    repo_id=REPO,
    repo_type="model",
    operations=ops,
    commit_message="Upload MLX q4 weights",
)
print("pushed:", ", ".join(FILES))
