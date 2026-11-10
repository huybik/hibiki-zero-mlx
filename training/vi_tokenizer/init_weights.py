#!/usr/bin/env python
"""Move a checkpoint onto the Vietnamese tokenizer swap
([vi-tokenizer](../../docs/agent-memory/vi-tokenizer.md#starting-a-checkpoint)).

The swap gives slots English never uses to Vietnamese pieces, so the text tables keep their
shapes and only those rows change meaning. Each replaced row of `text_emb`, `depformer_text_emb`
and `text_linear` becomes the mean of the rows of the pieces the base tokenizer split the new
piece into (`<model>_init.json`), read from the old tables, since some of those pieces are
replaced slots themselves. Every other tensor is copied as it is. The input must be on the base
tokenizer: its `text_tokenizer` metadata (training records it) names the base, or it has none.
The result starts a run with `--model-weight <out> --tokenizer <the swapped model>`. From the
repo root:

  python -m training.vi_tokenizer.init_weights --model-weight runs/<run>/best_step<N>.safetensors \\
      --init-map weights/tokenizer_spm_48k_multi6_2_vi6k_init.json --out weights/<name>.safetensors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from training.finetune.common import TOKENIZER_KEY, save_state, weight_tokenizer
from training.paths import require_file, resolve_repo_path

TABLES = ("text_emb.weight", "depformer_text_emb.weight", "text_linear.weight")


def swap_rows(table: torch.Tensor, pieces: list[dict]) -> torch.Tensor:
    """`table` with each piece's row set to the mean of its `init_from_ids` rows of `table`."""
    old = table.float()
    new = old.clone()
    for piece in pieces:
        new[piece["id"]] = old[piece["init_from_ids"]].mean(0)
    return new.to(table.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-weight", type=Path, required=True, help="A full-model checkpoint on the base tokenizer.")
    parser.add_argument("--init-map", type=Path, required=True, help="The swapped model's <model>_init.json.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = require_file(args.model_weight, "model weight")
    init = json.loads(require_file(args.init_map, "init map").read_text())
    out = resolve_repo_path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    with safe_open(str(source), framework="pt") as header:
        metadata = dict(header.metadata() or {})
    on = weight_tokenizer(source)
    if on != init["base_model"]:
        raise SystemExit(f"{source} is on {on}, not {init['base_model']}: its replaced rows are not the base's")

    state = load_file(str(source), device="cpu")
    pieces = init["pieces"]
    ids = [piece["id"] for piece in pieces]
    # text_linear has one row per piece; the embeddings hold one more, the stream's start token.
    card = int(state["text_linear.weight"].shape[0])
    if len(set(ids)) != len(ids):
        raise SystemExit("the init map replaces a slot twice")
    if any(not piece["init_from_ids"] or not all(0 <= i < card for i in [piece["id"], *piece["init_from_ids"]])
           for piece in pieces):
        raise SystemExit(f"the init map holds an id outside the {card} text pieces, or a piece with no rows to start from")
    for name in TABLES:
        state[name] = swap_rows(state[name], pieces)
    save_state(state, out, {**metadata, TOKENIZER_KEY: init["model"]})
    print(f"{len(pieces)} rows of {', '.join(TABLES)} start on {init['model']}: {out}")


if __name__ == "__main__":
    main()
