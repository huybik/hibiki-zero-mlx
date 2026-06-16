#!/usr/bin/env python
"""Publish the staged Hibiki-M q4 MLX artifact repo to Hugging Face."""
import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, HfFolder, create_repo

HERE = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "huybik/hibiki-1b-mlx-q4"
DEFAULT_STAGE_DIR = HERE / "weights" / "hibiki-m-mlx-q4"
FILES = [
    "README.md",
    ".gitattributes",
    "config.json",
    "hibiki-mlx-dc2cf5a5@80.q4.safetensors",
    "mimi-dbaa9758@125.safetensors",
    "tokenizer_spm_48k_multi6_2.model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="target Hugging Face model repo")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the model repo as private if it does not already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (HfFolder.get_token() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise SystemExit("Hugging Face auth missing. Run `hf auth login` or set HF_TOKEN.")

    missing = [filename for filename in FILES if not (args.stage_dir / filename).exists()]
    if missing:
        raise SystemExit(
            "missing staged files: "
            + ", ".join(missing)
            + "\nrun scripts/convert_hibiki_m_mlx_q4.py first"
        )

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(args.stage_dir / filename))
        for filename in FILES
    ]
    HfApi().create_commit(
        repo_id=args.repo,
        repo_type="model",
        operations=operations,
        commit_message="Add Hibiki-M MLX q4 artifacts",
    )
    info = HfApi().model_info(args.repo)
    print(f"published {args.repo}@{info.sha}")


if __name__ == "__main__":
    main()
