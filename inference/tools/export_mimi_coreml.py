"""Export Mimi's streaming encoder and decoder steps (`mimi_step.py`) as Core ML packages for
the Apple Neural Engine: fp16 throughout, fixed shapes, state in and out.

    python inference/tools/export_mimi_coreml.py weights/mimi-pytorch-e351c8d8@125.safetensors \
        MODEL_DIR --frames 2 --decoder-codebooks 8

writes MODEL_DIR/mimi-decoder.mlpackage, which the Swift runtime loads (`NeuralDecoder`), and
MODEL_DIR/mimi-encoder.mlpackage, which it does not: on the Neural Engine the encoder's codes
match fp32 on only 94% of frames in codebook 0 and ~17% in codebooks 8-15, so Mimi encodes in
fp32 on the CPU. Inputs and outputs are named as `mimi_step` names them; the host feeds `conv`,
`keys` and `values` back (`new_*`), starting from zeros, and computes `mask`, `cos`, `sin` and
`first` per call as `mimi_step.Inputs.at` does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

import mimi_step as ms


def _convert(module: torch.nn.Module, inputs: dict[str, tuple[tuple[int, ...], type]],
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
    model.save(str(path))
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path, help="the model directory")
    parser.add_argument("--frames", type=int, default=2, help="frames per call")
    parser.add_argument("--decoder-codebooks", type=int, required=True,
                        help="the LM's dep_q: 16 for upstream Hibiki-Zero, 8 for the 8-codebook cuts")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    w = ms.load(args.checkpoint)
    k, steps = args.frames, args.frames * ms.STEPS_PER_FRAME
    window = (ms.LAYERS, ms.HEADS, ms.WINDOW, ms.HEAD_DIM)
    f16 = np.float16
    per_call = {
        "mask": ((steps, ms.WINDOW + steps), f16),
        "cos": ((steps, ms.HEAD_DIM // 2), f16),
        "sin": ((steps, ms.HEAD_DIM // 2), f16),
    }
    state_out = ["new_conv", "new_keys", "new_values"]

    encoder = ms.EncoderStep(w, k)
    _convert(
        encoder,
        {
            "pcm": ((1, 1, k * ms.FRAME), f16),
            "conv": ((1, encoder.size), f16),
            "keys": (window, f16),
            "values": (window, f16),
            **per_call,
            "first": ((1,), f16),
        },
        ["codes", *state_out],
        args.out_dir / "mimi-encoder.mlpackage",
    )
    decoder = ms.DecoderStep(w, k, args.decoder_codebooks)
    _convert(
        decoder,
        {
            "codes": ((1, args.decoder_codebooks, k), np.int32),
            "conv": ((1, decoder.size), f16),
            "keys": (window, f16),
            "values": (window, f16),
            **per_call,
        },
        ["pcm", *state_out],
        args.out_dir / "mimi-decoder.mlpackage",
    )


if __name__ == "__main__":
    main()
