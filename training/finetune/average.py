#!/usr/bin/env python
"""Average a run's model checkpoints into one weight (checkpoint averaging, an NMT staple):

    python -m training.finetune.average --out weights/NAME_avg.safetensors \\
        runs/RUN/model_step056000.safetensors runs/RUN/model_step064000.safetensors ...

Every input must hold the same tensors in the same shapes and dtypes, on the same tokenizer (its
`text_tokenizer` metadata). Floating tensors are averaged in float32 and written in the inputs'
dtype; any other tensor must be equal in all of them. The output keeps the first input's metadata
and names its inputs (`averaged_from`). Score it as any checkpoint: `eval.py` with the run's
--eval-pairs and --seed on the box, or `gate.py` after `convert_mlx.py` on the Mac. Tensors are
read one at a time, so memory holds the output and one input tensor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from training.finetune.model import TOKENIZER_KEY, save_state
from training.paths import repo_display_path, require_file, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", type=Path, nargs="+", help="Model checkpoints of one run.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def average(paths: list[Path]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The mean of `paths`' tensors and the first one's metadata, with `averaged_from`."""
    handles = [safe_open(str(path), framework="pt") for path in paths]
    first = handles[0]
    keys = sorted(first.keys())
    metadata = dict(first.metadata() or {})
    tokenizer = metadata.get(TOKENIZER_KEY)
    for path, handle in zip(paths[1:], handles[1:]):
        if sorted(handle.keys()) != keys:
            raise ValueError(f"{repo_display_path(path)} holds other tensors than {repo_display_path(paths[0])}")
        if (handle.metadata() or {}).get(TOKENIZER_KEY) != tokenizer:
            raise ValueError(f"{repo_display_path(path)} is on another tokenizer than {repo_display_path(paths[0])}")
    state = {}
    for key in keys:
        base = first.get_tensor(key)
        if not base.dtype.is_floating_point:
            for path, handle in zip(paths[1:], handles[1:]):
                if not torch.equal(handle.get_tensor(key), base):
                    raise ValueError(f"{key} differs in {repo_display_path(path)}, and it is not a float")
            state[key] = base
            continue
        total = base.float()
        for path, handle in zip(paths[1:], handles[1:]):
            tensor = handle.get_tensor(key)
            if tensor.shape != base.shape or tensor.dtype != base.dtype:
                raise ValueError(f"{key} is {tuple(tensor.shape)} {tensor.dtype} in {repo_display_path(path)}")
            total += tensor.float()
        state[key] = (total / len(paths)).to(base.dtype)
    metadata["averaged_from"] = ",".join(path.name for path in paths)
    return state, metadata


def main() -> None:
    args = parse_args()
    paths = [require_file(path, "checkpoint") for path in args.weights]
    if len(paths) < 2:
        raise ValueError("Averaging needs at least two checkpoints")
    files = {(stat.st_dev, stat.st_ino) for stat in (path.stat() for path in paths)}
    if len(files) != len(paths):  # a best_stepN is a hard link of its model_stepN
        raise ValueError("A checkpoint is named twice")
    state, metadata = average(paths)
    out = resolve_repo_path(args.out)
    save_state(state, out, metadata)
    print(f"Averaged {len(paths)} checkpoints into {repo_display_path(out)}")


if __name__ == "__main__":
    main()
