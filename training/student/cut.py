"""Cut a phone student from a Hibiki-Zero checkpoint ([phone-student](../../docs/agent-memory/phone-student.md),
plan stage 5.3): fewer main transformer layers, fewer codebooks a stream and a shallower depformer, every
kept tensor copied from the parent, so the student starts from what the 3B learned and trains on the
streams directly (`training.finetune.train --config-path <out>/config.json --model-weight <out>/model.safetensors`).

  main layers   28 -> --layers (or --num-layers, spread evenly with the first and the last kept)
  codebooks     16 a stream -> --codebooks: the target's first k and the source's first k
                (the input embeddings `emb`, the depformer's steps, its norms and output `linears`)
  depformer     6 layers -> --depformer-layers, each keeping its first k step slices; or, with
                --heads, none: k parallel heads in its slot (`heads.py`), initialized afresh
                (--seed), each codebook j delayed j frames

The student's config.json is the parent's with the cut's sizes, its `delays` and step schedule
truncated alike, and its own weight and that weight's tokenizer named (`moshi_name`, `tokenizer_name`);
moshi hands every key to the model, so the cut's record goes into the weight's metadata, which keeps
the parent's (its tokenizer among it) and adds `student_of` and `student`.

  python -m training.student.cut --weight weights/twoway_sft_final.safetensors --out-dir weights/student_10l
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    read_json,
    repo_display_path,
    require_file,
)
from training.student import heads


def spread(count: int, total: int) -> list[int]:
    """`count` indices of `total`, evenly spread, the first and the last among them."""
    if not 1 <= count <= total:
        raise ValueError(f"cannot keep {count} of {total}")
    if count == 1:
        return [0]
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def indices(text: str | None, count: int, total: int, label: str) -> list[int]:
    """Explicit comma-separated indices, or `count` spread evenly; checked against `total`."""
    kept = [int(part) for part in text.split(",") if part.strip()] if text is not None else spread(count, total)
    if not kept or kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= total:
        raise ValueError(f"{label}: need increasing indices in [0, {total}), got {kept}")
    return kept


def student_config(parent: dict[str, Any], layers: list[int], codebooks: int,
                   depformer_layers: list[int], parallel_heads: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """The parent's config with the cut's sizes, and the cut's record. The source stream's codebooks
    follow the target's (`n_q` = 2 x `dep_q`, as Hibiki's), and `delays` keeps text, then each
    stream's first `codebooks`; with `parallel_heads`, the target's are `heads.delays` and the
    depformer moshi builds before the heads replace it is one layer (`depformer_layers` unused)."""
    dep_q, n_q = int(parent["dep_q"]), int(parent["n_q"])
    if n_q != 2 * dep_q:
        raise ValueError(f"expected a source stream as wide as the target's, got n_q {n_q}, dep_q {dep_q}")
    if not 1 <= codebooks <= dep_q:
        raise ValueError(f"--codebooks must be in [1, {dep_q}], got {codebooks}")
    delays = list(parent["delays"])
    if len(delays) != 1 + n_q:
        raise ValueError(f"expected {1 + n_q} delays, got {len(delays)}")
    target = heads.delays(codebooks) if parallel_heads else delays[1 : 1 + codebooks]
    child = dict(parent)
    child.update(
        num_layers=len(layers),
        n_q=2 * codebooks,
        dep_q=codebooks,
        delays=[delays[0], *target, *delays[1 + dep_q : 1 + dep_q + codebooks]],
        depformer_num_layers=1 if parallel_heads else len(depformer_layers),
    )
    if (schedule := parent.get("depformer_weights_per_step_schedule")) is not None:
        # The kept steps' slices: the parent's 0..m, which the student holds as its own 0..m (moshi
        # sizes the per-step weights max + 1). Hibiki-Zero's steps 8-15 share slice 8.
        kept = schedule[:codebooks]
        if sorted(set(kept)) != list(range(max(kept) + 1)):
            raise ValueError(f"the parent's first {codebooks} steps use weight slices with gaps: {kept}")
        child["depformer_weights_per_step_schedule"] = kept
    if parallel_heads:
        child[heads.CONFIG_KEY] = heads.KIND
    record = {"main_layers": layers, "codebooks": codebooks,
              "depformer_layers": [] if parallel_heads else depformer_layers,
              "audio_heads": heads.KIND if parallel_heads else None,
              "parent_num_layers": int(parent["num_layers"]), "parent_dep_q": dep_q,
              "parent_depformer_num_layers": int(parent["depformer_num_layers"])}
    return child, record


def source_key(key: str, layers: list[int], codebooks: int, depformer_layers: list[int],
               parent_dep_q: int) -> str | None:
    """The parent tensor a student tensor copies (per-step tensors keep their index)."""
    patterns: list[tuple[str, Any]] = [
        (r"^transformer\.layers\.(\d+)\.", lambda m: f"transformer.layers.{layers[int(m[1])]}."),
        # Student embeddings 0..k-1 are the target's first k; k..2k-1 the source's first k.
        (r"^emb\.(\d+)\.", lambda m: f"emb.{int(m[1]) if int(m[1]) < codebooks else parent_dep_q + int(m[1]) - codebooks}."),
        (r"^depformer\.layers\.(\d+)\.", lambda m: f"depformer.layers.{depformer_layers[int(m[1])]}."),
    ]
    for pattern, replace in patterns:
        match = re.match(pattern, key)
        if match:
            return replace(match) + key[match.end():]
    return key  # per-step tensors keep their index: the student's steps are the parent's first k


def cut_state(parent: dict[str, Any], child_keys: dict[str, tuple[int, ...]], layers: list[int], codebooks: int,
              depformer_layers: list[int], parent_dep_q: int, fresh: dict[str, Any] | None = None) -> dict[str, Any]:
    """The student's state dict (`child_keys`: its names and shapes) from the parent's tensors,
    each copied whole, but the `fresh` ones (a heads student's heads); any missing or mis-shaped
    tensor raises."""
    state = {}
    for key, shape in child_keys.items():
        if fresh is not None and key in fresh:
            state[key] = fresh[key]
            continue
        src = source_key(key, layers, codebooks, depformer_layers, parent_dep_q)
        if src not in parent:
            raise KeyError(f"student {key} has no parent tensor ({src})")
        tensor = parent[src]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"student {key} is {tuple(shape)}, its parent {src} {tuple(tensor.shape)}")
        state[key] = tensor.contiguous()
    return state


def student_shapes(config_path: Path, weight: Path, tokenizer: Path = DEFAULT_TOKENIZER) -> dict[str, tuple[int, ...]]:
    """The names and shapes of the model the trainer builds from `config_path` (`common.build_lm`),
    on the meta device."""
    import torch

    from training.finetune import common

    # Mimi and the tokenizer are never read here; named so nothing is fetched from the Hub.
    info = common.load_checkpoint_info(argparse.Namespace(
        hf_repo="kyutai/hibiki-zero-3b-pytorch-bf16", model_weight=weight, mimi_weight=DEFAULT_MIMI_WEIGHT,
        tokenizer=tokenizer, config_path=config_path))
    lm = common.build_lm(info, "meta", torch.bfloat16, load_weight=False)
    return {key: tuple(value.shape) for key, value in lm.state_dict().items()}


def fresh_heads(config: dict[str, Any], dtype: Any, seed: int) -> dict[str, Any]:
    """A heads student's heads, initialized afresh (torch.nn.Linear's own init, seeded), named as in
    the student's state dict."""
    import torch

    torch.manual_seed(seed)
    module = heads.ParallelHeads(int(config["dim"]), int(config["card"]), int(config["dep_q"])).to(dtype=dtype)
    return {f"depformer.{key}": value.detach().contiguous() for key, value in module.state_dict().items()}


def main() -> None:
    from safetensors import safe_open

    from training.finetune.common import save_state, weight_tokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", type=Path, required=True, help="the parent checkpoint (safetensors)")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH, help="the parent's config")
    parser.add_argument("--out-dir", type=Path, required=True, help="gets config.json and model.safetensors")
    layers_arg = parser.add_mutually_exclusive_group()
    layers_arg.add_argument("--layers", help="main layers to keep, e.g. 0,3,6,9,12,15,18,21,24,27")
    layers_arg.add_argument("--num-layers", type=int, default=10, help="main layers to keep, spread evenly")
    parser.add_argument("--codebooks", type=int, default=8, help="codebooks a stream (the target's and the source's)")
    parser.add_argument("--heads", action="store_true", help="parallel codebook heads instead of the depformer")
    parser.add_argument("--seed", type=int, default=0, help="the heads' initialization (with --heads)")
    dep_arg = parser.add_mutually_exclusive_group()
    dep_arg.add_argument("--depformer-layers", help="depformer layers to keep, e.g. 0,2,5")
    dep_arg.add_argument("--num-depformer-layers", type=int, help="depformer layers to keep, spread evenly (default 3)")
    args = parser.parse_args()
    if args.heads and (args.depformer_layers is not None or args.num_depformer_layers is not None):
        parser.error("--depformer-layers and --num-depformer-layers do not apply with --heads")
    if not args.heads and args.seed != 0:
        parser.error("--seed applies only with --heads")

    weight = require_file(args.weight, "parent weight")
    parent_config = read_json(require_file(args.config_path, "parent config"))
    layers = indices(args.layers, args.num_layers, int(parent_config["num_layers"]), "--layers")
    depformer_layers = indices(args.depformer_layers, args.num_depformer_layers or 3,
                               int(parent_config["depformer_num_layers"]), "--depformer-layers")
    config, record = student_config(parent_config, layers, args.codebooks, depformer_layers, args.heads)
    if args.heads:
        record["heads_seed"] = args.seed  # with the parent, the record recuts the student exactly

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path, out_weight = args.out_dir / "config.json", args.out_dir / "model.safetensors"
    config.update(moshi_name=out_weight.name, tokenizer_name=weight_tokenizer(weight))  # not the parent's
    staged = args.out_dir / ".config.json.tmp"  # config.json only once the weight is in
    staged.write_text(json.dumps(config, indent=2) + "\n")
    shapes = student_shapes(staged, weight)

    with safe_open(str(weight), "pt") as f:
        metadata = dict(f.metadata() or {})
        names = set(f.keys())
        needed = {source_key(key, layers, args.codebooks, depformer_layers, int(parent_config["dep_q"]))
                  for key in shapes}
        parent = {key: f.get_tensor(key) for key in needed & names}
    fresh = fresh_heads(config, parent["text_linear.weight"].dtype, args.seed) if args.heads else None
    state = cut_state(parent, shapes, layers, args.codebooks, depformer_layers, int(parent_config["dep_q"]), fresh)
    metadata["student_of"] = repo_display_path(weight)
    metadata["student"] = json.dumps(record)
    save_state(state, out_weight, metadata)
    staged.replace(config_path)
    params = sum(t.numel() for t in state.values())
    dtypes = sorted({str(t.dtype) for t in state.values()})
    audio = "parallel heads" if args.heads else f"depformer layers {depformer_layers}"
    print(f"student: {len(layers)} main layers {layers}, {args.codebooks} codebooks a stream, {audio}; "
          f"{params / 1e9:.3f}B parameters ({', '.join(dtypes)}) -> "
          f"{repo_display_path(out_weight)}, {repo_display_path(config_path)}")


if __name__ == "__main__":
    main()
