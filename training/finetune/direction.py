"""Give a two-way checkpoint the direction condition ([direction](../../docs/agent-memory/direction.md)):
a lookup table moshi sums into every frame's input (its `lut` conditioner, fused by `sum`), one entry
a direction, so the runtimes say which way to translate instead of the model guessing it from the
voice (it took the user's Vietnamese for English, and quiet English for Vietnamese).

The table's output projection starts at zero, so the model's outputs are the checkpoint's exactly
until training moves it; its 16-wide entries start random (--seed), so the projection's first
gradients are not zero. The out directory gets the parent's config.json with the conditioner, its
own weight and tokenizer named (`moshi_name`, `tokenizer_name`), and model.safetensors, the
parent's tensors and the table's (the parent's metadata kept, `direction_of` added). Train from it
with `training.finetune.train --config-path <out>/config.json --model-weight <out>/model.safetensors`.

  python -m training.finetune.direction --weight weights/grpo_twoway_best1400.safetensors --out-dir weights/twoway_direction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from training.finetune.common import save_state, weight_tokenizer
from training.finetune.hibiki_helpers import DIRECTION
from training.paths import DEFAULT_CONFIG_PATH, read_json, repo_display_path, require_file

DIRECTIONS = ["vi-en", "en-vi"]  # <source>-<target>, as the caches and the evaluation rows name them
TABLE_DIM = 16  # each entry's width before the projection, as Hibiki's own LUT conditioner
PREFIX = f"condition_provider.conditioners.{DIRECTION}."


def direction_config(parent: dict[str, Any]) -> dict[str, Any]:
    """The parent's config with the direction conditioner summed into every frame."""
    if parent.get("conditioners") or parent["fuser"]["sum"]:
        raise ValueError(f"the parent is already conditioned: {parent['conditioners']}, {parent['fuser']}")
    table = {"n_bins": len(DIRECTIONS), "dim": TABLE_DIM, "tokenizer": "noop", "possible_values": DIRECTIONS}
    return {**parent, "conditioners": {DIRECTION: {"type": "lut", "lut": table}},
            "fuser": {**parent["fuser"], "sum": [DIRECTION]}}


def direction_state(dim: int, seed: int) -> dict[str, torch.Tensor]:
    """The table's tensors, as moshi's LUTConditioner names them: the entries (and the padding
    row moshi adds, never used: every row names its direction), the projection at zero, and the
    padding vector, zero."""
    generator = torch.Generator().manual_seed(seed)
    return {
        PREFIX + "embed.weight": torch.randn(len(DIRECTIONS) + 1, TABLE_DIM, generator=generator),
        PREFIX + "output_proj.weight": torch.zeros(dim, TABLE_DIM),
        PREFIX + "learnt_padding": torch.zeros(1, 1, dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", type=Path, required=True, help="the parent checkpoint (safetensors)")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH, help="the parent's config")
    parser.add_argument("--out-dir", type=Path, required=True, help="gets config.json and model.safetensors")
    parser.add_argument("--seed", type=int, default=0, help="the table's entries")
    args = parser.parse_args()

    weight = require_file(args.weight, "parent weight")
    parent = read_json(require_file(args.config_path, "parent config"))
    config = direction_config(parent)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path, out_weight = args.out_dir / "config.json", args.out_dir / "model.safetensors"
    config.update(moshi_name=out_weight.name, tokenizer_name=weight_tokenizer(weight))  # not the parent's

    with safe_open(str(weight), "pt") as handle:
        metadata = dict(handle.metadata() or {})
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    if any(key.startswith("condition_provider.") for key in state):
        raise ValueError(f"{weight} already holds conditioner tensors")
    state |= direction_state(int(config["dim"]), args.seed)
    metadata["direction_of"] = repo_display_path(weight)
    save_state(state, out_weight, metadata)
    config_path.write_text(json.dumps(config, indent=2) + "\n")  # once the weight is in
    print(f"{repo_display_path(weight)} + the direction table ({', '.join(DIRECTIONS)}) -> "
          f"{repo_display_path(out_weight)}, {repo_display_path(config_path)}")


if __name__ == "__main__":
    main()
