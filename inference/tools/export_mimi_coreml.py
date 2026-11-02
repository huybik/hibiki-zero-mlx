"""Export Mimi's streaming encoder and decoder steps (`mimi_step.py`) as compiled Core ML models
for the Apple Neural Engine: fp16 throughout, fixed shapes, state in and out.

    python inference/tools/export_mimi_coreml.py weights/mimi-pytorch-e351c8d8@125.safetensors \
        MODEL_DIR --decoder-codebooks 8

writes MODEL_DIR/mimi-decoder.mlmodelc, one frame per call, which the Swift runtime loads
(`NeuralDecoder`). `--encoder-frames K` also writes MODEL_DIR/mimi-encoder.mlmodelc, K frames
per call, for NPU experiments; the runtime does not use it: on the Neural Engine the encoder's
codes match fp32 on only 94% of frames in codebook 0 and ~17% in codebooks 8-15, so Mimi encodes
in fp32 on the CPU. Inputs and outputs are named as `mimi_step` names them; the host feeds
`conv`, `keys` and `values` back (`new_*`), starting from zeros, and computes `mask`, `cos`,
`sin` and `first` per call as `mimi_step.Inputs.at` does.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

import mimi_step as ms


def _export(module: torch.nn.Module, inputs: dict[str, tuple[tuple[int, ...], type]],
            outputs: list[str], path: Path) -> None:
    examples = []
    for shape, dtype in inputs.values():
        examples.append(torch.zeros(shape, dtype=torch.int32 if dtype is np.int32 else torch.float32))
    traced = torch.jit.trace(module.eval(), tuple(examples))
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=d) for n, (s, d) in inputs.items()],
        outputs=[ct.TensorType(name=n) for n in outputs],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    # Compiled here, so the app loads it without compiling a copy into its temporary folder.
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


def _per_call(frames: int) -> dict[str, tuple[tuple[int, ...], type]]:
    steps = frames * ms.STEPS_PER_FRAME
    return {
        "keys": ((ms.LAYERS, ms.HEADS, ms.WINDOW, ms.HEAD_DIM), np.float16),
        "values": ((ms.LAYERS, ms.HEADS, ms.WINDOW, ms.HEAD_DIM), np.float16),
        "mask": ((steps, ms.WINDOW + steps), np.float16),
        "cos": ((steps, ms.HEAD_DIM // 2), np.float16),
        "sin": ((steps, ms.HEAD_DIM // 2), np.float16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path, help="the model directory")
    parser.add_argument("--decoder-codebooks", type=int, required=True,
                        help="the LM's dep_q: 16 for upstream Hibiki-Zero, 8 for the 8-codebook cuts")
    parser.add_argument("--encoder-frames", type=int,
                        help="also export the encoder, this many frames per call (unused by the runtime)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    w = ms.load(args.checkpoint)
    state_out = ["new_conv", "new_keys", "new_values"]

    if k := args.encoder_frames:
        encoder = ms.EncoderStep(w, k)
        _export(
            encoder,
            {
                "pcm": ((1, 1, k * ms.FRAME), np.float16),
                "conv": ((1, encoder.size), np.float16),
                **_per_call(k),
                "first": ((1,), np.float16),
            },
            ["codes", *state_out],
            args.out_dir / "mimi-encoder.mlmodelc",
        )
    decoder = ms.DecoderStep(w, 1, args.decoder_codebooks)
    _export(
        decoder,
        {
            "codes": ((1, args.decoder_codebooks, 1), np.int32),
            "conv": ((1, decoder.size), np.float16),
            **_per_call(1),
        },
        ["pcm", *state_out],
        args.out_dir / "mimi-decoder.mlmodelc",
    )


if __name__ == "__main__":
    main()
