"""The Core ML conversion the export tools share (`export_mimi_coreml`, `export_lm_coreml`): a
fixed-shape PyTorch module traced and converted to an fp16 ML program for the Neural Engine, its
weights palettized, and the model saved compiled."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch


def convert(module: torch.nn.Module, inputs: dict[str, tuple[tuple[int, ...], type]],
            outputs: list[str], states: dict[str, tuple[int, ...]] | None = None) -> ct.models.MLModel:
    """`module` traced on zero inputs of the given shapes and dtypes (np.float16 inputs trace in
    float32) and converted to an fp16 ML program for the CPU and Neural Engine (iOS 18); each of
    `states` is an fp16 buffer of the module's, kept by Core ML between calls."""
    examples = []
    for shape, dtype in inputs.values():
        examples.append(torch.zeros(shape, dtype=torch.int32 if dtype is np.int32 else torch.float32))
    traced = torch.jit.trace(module.eval(), tuple(examples))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=d) for n, (s, d) in inputs.items()],
        outputs=[ct.TensorType(name=n) for n in outputs],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=np.float16), name=n)
                for n, s in (states or {}).items()],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )


def save_compiled(model: ct.models.MLModel, path: Path) -> None:
    """Compiles `model` to `path` (an .mlmodelc), replacing what is there only once it compiled.
    Compiled here, so the app loads it without compiling a copy into its temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / path.with_suffix(".mlpackage").name
        model.save(str(package))
        compiled = ct.models.utils.compile_model(str(package), str(Path(tmp) / path.name))
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        shutil.move(compiled, path)
    print(path)


def palettize(model: ct.models.MLModel, bits: int) -> ct.models.MLModel:
    """`model`'s weights palettized to `bits` bits: k-means, one table per 16 output channels
    (a weight of 65,536 elements or fewer stays as it is)."""
    op = cto.OpPalettizerConfig(nbits=bits, mode="kmeans", granularity="per_grouped_channel",
                                group_size=16, weight_threshold=1 << 16)
    return cto.palettize_weights(model, cto.OptimizationConfig(global_config=op))
