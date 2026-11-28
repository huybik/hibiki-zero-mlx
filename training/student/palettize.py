"""Training against the phone's palettization (plan stage 6.4,
[student-palettize](../../docs/agent-memory/student-palettize.md)).

The Core ML export (`inference/tools/export_lm_coreml.py --bits N`) palettizes every weight of the
heads student's trunk and heads with more than 2^16 elements: k-means, one table of 2^N values for
each 16 output channels (rows of the torch weight; convert_mlx.py keeps the rows in order).
Measured on a layer used as lm_step uses it (`x @ W.T`, 2026-11-21): the tables come out
[rows / 16, 1, 2^N, 1], and a weight whose every 16 rows already hold at most 2^N values comes back
exact. So a model trained with the same tables and `snap`ped onto them exports without loss.

`Palettizer` puts that quantizer in train.py's step (`--palettize-bits`). Before each forward the
palettized weights become their nearest table entry; the backward's gradients at those weights
update the full-precision ones (the straight-through estimator), which come back before the
optimizer steps. The tables are refit (`fit`: entries spread as the density's cube root, the
optimal spacing of a fine scalar quantizer, then Lloyd iterations) on the
full-precision weights at the start, after every `refit` steps and after the last, so a resumed
pair, saved on a refit step, gets the tables its run had, and `python -m training.student.palettize`
snaps a saved checkpoint with the tables its next step would use (a fit is deterministic on a
device; the CLI fits on CUDA where there is one, as training does).

    python -m training.student.palettize --weight runs/RUN/best_stepNNNNNN.safetensors \\
        --bits 6 --out weights/NAME_pal6.safetensors
"""

from __future__ import annotations

import argparse
import contextlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

GROUP_ROWS = 16  # export_lm_coreml's group_size: one table per 16 output channels
THRESHOLD = 1 << 16  # its weight_threshold: a weight with more elements than this is palettized
# The weights lm_step's Trunk, Heads and DepformerStep hold (embeddings stay on the host, in fp16).
PALETTIZED = re.compile(
    r"transformer\.layers\.\d+\..+\.weight|text_linear\.weight|depformer\.heads\.\d+\.weight"
    r"|depformer\.layers\.\d+\..+\.weight|depformer_in\.\d+\.weight|linears\.\d+\.weight"
)
LLOYD_ITERATIONS = 10
BINS = 1024  # the equal-count bins `fit` places its first entries by


def palettized(named: Iterator[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    """The (name, tensor) pairs of `named` that the export palettizes."""
    chosen = []
    for name, tensor in named:
        if PALETTIZED.fullmatch(name) and tensor.ndim == 2 and tensor.numel() > THRESHOLD:
            if tensor.shape[0] % GROUP_ROWS:
                raise ValueError(f"{name} has {tensor.shape[0]} rows, not a multiple of {GROUP_ROWS}")
            chosen.append((name, tensor))
    if not chosen:
        raise ValueError("no weight to palettize: not a parallel-heads student's names")
    return chosen


def _groups(weight: torch.Tensor) -> torch.Tensor:
    """[rows / GROUP_ROWS, GROUP_ROWS * columns]: one table's values a row, float32."""
    return weight.detach().float().reshape(weight.shape[0] // GROUP_ROWS, -1)


def _nearest(groups: torch.Tensor, tables: torch.Tensor) -> torch.Tensor:
    """Each value's nearest entry of its group's sorted table, as an index."""
    bounds = (tables[:, 1:] + tables[:, :-1]) / 2
    return torch.searchsorted(bounds.contiguous(), groups.contiguous())


def _start(ordered: torch.Tensor, entries: int) -> torch.Tensor:
    """First tables for groups of sorted values: each group's entries at the (i + 0.5) / entries
    quantiles of its density raised to 1/3 (Panter and Dite's spacing for a fine scalar
    quantizer), from BINS bins of equal count, so the bulk is resolved however far the outliers
    reach; interpolated within a bin. Equal-count quantiles themselves would crowd the bulk and
    leave the tails coarse."""
    edges = ordered[:, torch.linspace(0, ordered.shape[1] - 1, BINS + 1, device=ordered.device).long()]
    width = edges[:, 1:] - edges[:, :-1]
    # A bin of equal count has density ∝ 1 / width, so its share of ∫ density^(1/3) ∝ width^(2/3).
    cdf = width.pow(2 / 3).cumsum(dim=1)
    cdf = cdf / cdf[:, -1:].clamp(min=1e-30)
    targets = ((torch.arange(entries, device=ordered.device) + 0.5) / entries).expand(len(ordered), -1)
    at = torch.searchsorted(cdf, targets.contiguous()).clamp(max=BINS - 1)
    below = torch.where(at > 0, cdf.gather(1, (at - 1).clamp(min=0)), torch.zeros_like(targets))
    inside = ((targets - below) / (cdf.gather(1, at) - below).clamp(min=1e-30)).clamp(0, 1)
    return edges.gather(1, at) + inside * width.gather(1, at)


@torch.no_grad()
def fit(weight: torch.Tensor, bits: int, iterations: int = LLOYD_ITERATIONS) -> torch.Tensor:
    """Sorted tables [rows / GROUP_ROWS, 2^bits] for `weight` [rows, columns]: `_start`, then Lloyd
    iterations on each group's sorted values, where an entry's values are one contiguous range
    (found by searching its bounds, ties going to the lower entry as in `snap`) and their mean
    comes from float64 prefix sums: no atomics, so a fit is the same on every run on a device.
    An entry no value picks (a start entry between far outliers) moves to the worst-fit value of
    another entry's range, the n-th empty entry to the n-th worst range, so no two land on one
    value."""
    ordered = _groups(weight).sort(dim=1).values
    count = ordered.shape[1]
    prefix = torch.nn.functional.pad(ordered.double().cumsum(dim=1), (1, 0))
    tables = _start(ordered, 1 << bits)
    for _ in range(iterations):
        bounds = (tables[:, 1:] + tables[:, :-1]) / 2
        edges = torch.searchsorted(ordered, bounds.contiguous(), right=True)
        starts = torch.nn.functional.pad(edges, (1, 0))
        ends = torch.nn.functional.pad(edges, (0, 1), value=count)
        sizes = ends - starts
        means = ((prefix.gather(1, ends) - prefix.gather(1, starts)) / sizes.clamp(min=1)).float()
        empty = sizes == 0
        low = ordered.gather(1, starts.clamp(max=count - 1))
        high = ordered.gather(1, (ends - 1).clamp(min=0))
        far = torch.where(means - low > high - means, low, high)  # each range's worst-fit value
        error = torch.where(empty, -1.0, torch.maximum(means - low, high - means))
        candidates = far.gather(1, error.argsort(dim=1, descending=True))
        rank = (empty.long().cumsum(dim=1) - 1).clamp(min=0)
        tables = torch.where(empty, candidates.gather(1, rank), means).sort(dim=1).values
    return tables


@torch.no_grad()
def snap(weight: torch.Tensor, tables: torch.Tensor) -> torch.Tensor:
    """`weight` with every value replaced by its group's nearest table entry, in its dtype."""
    groups = _groups(weight)
    return tables.gather(1, _nearest(groups, tables)).reshape(weight.shape).to(weight.dtype)


class Palettizer:
    """The export's palettization in a training step: `applied()` holds the palettized weights in
    place of the full-precision ones for a forward and its backward (or a validation), and
    restores them after; `refit()` fits the tables to the full-precision weights."""

    def __init__(self, lm: Any, bits: int):
        self.bits = bits
        self.params = [param for _, param in palettized(lm.named_parameters())]
        self.full = [torch.empty_like(param) for param in self.params]
        self.tables: list[torch.Tensor] = []

    def refit(self) -> None:
        self.tables = [fit(param.data, self.bits) for param in self.params]

    @contextlib.contextmanager
    def applied(self) -> Iterator[None]:
        if not self.tables:
            raise RuntimeError("Palettizer.applied before refit")
        with torch.no_grad():
            for param, full, tables in zip(self.params, self.full, self.tables, strict=True):
                full.copy_(param.data)
                param.data.copy_(snap(param.data, tables))
        try:
            yield
        finally:
            with torch.no_grad():
                for param, full in zip(self.params, self.full, strict=True):
                    param.data.copy_(full)


def main() -> None:
    from training.finetune import common
    from training.paths import repo_display_path, require_file, resolve_repo_path

    parser = argparse.ArgumentParser(description="Snap a heads student's checkpoint onto its palettization tables.")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    weight = require_file(args.weight, "checkpoint")
    from safetensors import safe_open

    with safe_open(str(weight), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    for name, tensor in palettized(iter(state.items())):
        on = tensor.to(args.device)
        state[name] = snap(on, fit(on, args.bits)).cpu()
    metadata["palettized"] = f"{args.bits} bits, {GROUP_ROWS} rows a table"
    out = resolve_repo_path(args.out)
    common.save_state(state, out, metadata)
    print(f"Snapped {repo_display_path(weight)} onto {args.bits}-bit tables: {repo_display_path(out)}")


if __name__ == "__main__":
    main()
