"""The aligner's transformer on the Neural Engine, beside Mimi on the GPU, for the MLX cache build
(`cache_build.MLXModels`, `cache_codes --ane-ctc`): the base layout's encoder LayerNorm, 12
post-norm layers, CTC head and log-softmax as a float16 Core ML program, run on the transformer
input that `wav2vec2_mlx` computes on the GPU in float32 (`Wav2Vec2CTC.transformer_input`). Only
the columns the Viterbi reads come back (`text_timing.vocab_columns`).

Four things keep float16 on the Neural Engine near float32 (the Vietnamese aligner against MLX
float32 on 600 `en_speech` rows, 2026-11-22: 99.90-99.93% of argmaxes equal, 6 of 15,983 word
starts a 12.5 Hz frame apart, scores within 0.009):
- the positional convolution stays on the GPU: in float16 the 8,192-term sums of omniASR's (the
  former aligner) put a 3.5% error on every frame, 60% after its 24 layers;
- each LayerNorm's input is scaled by 1/64, or its sum of squares overflows float16 (LayerNorm is
  scale-invariant but for its epsilon, which float16 rounds up to 2^-24: 2.4e-4 at the model's
  scale against its 1e-5);
- the log-softmax is written out (max, exp, sum, log): Core ML's `reduce_log_sum_exp` is ~15 nats
  off on the Neural Engine;
- the head's columns are a linear layer of their own (95 for Vietnamese), which the Neural Engine
  runs; a gather of them would go to the CPU.

The program takes one shape per length bucket up to MAX_BUCKETS (enumerated shapes); a window's
clips go in bucket order, since switching shapes costs the Neural Engine ~40% of its speed. It is
converted once into weights/coreml/, under a name keyed by the aligner's revision, its columns,
the shapes, `convert`'s source and the coremltools version, and the Neural Engine compiles it on
its first load (~1.5 min), which macOS keeps for later loads of the same file. It runs in a
process of its own, which exits with the build: Core ML's predict holds the GIL, which stalled the
MLX calls that queue Mimi (with omniASR the build ran 207x in a thread, 293x in a process).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
import tempfile
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import REPO_ROOT
from training.prep.text_timing import Aligner, ctc_frames, vocab_columns
from training.prep.wav2vec2_mlx import BUCKET_SAMPLES

MAX_BUCKETS = 19  # 24.3 s; the GPU runs longer clips' CTC
SHAPES = [ctc_frames(buckets * BUCKET_SAMPLES) for buckets in range(1, MAX_BUCKETS + 1)]
LN_SCALE = 1 / 64
MODEL_DIR = REPO_ROOT / "weights" / "coreml"


def convert(aligner: Aligner, files: Path, out: Path) -> None:
    """The aligner's transformer as a compiled float16 Core ML program at `out` (an .mlmodelc),
    from the positional convolution's output [1, T, dim] and the clip's frames' mask [1, T] to the
    columns' log-probabilities [1, T, columns]."""
    import coremltools as ct
    import torch
    from transformers import Wav2Vec2ForCTC

    def layer_norm(norm: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x * LN_SCALE, norm.normalized_shape, norm.weight, norm.bias, norm.eps * LN_SCALE**2
        )

    class Body(torch.nn.Module):
        """HF Wav2Vec2ForCTC's encoder LayerNorm, post-norm layers and head (the base layout)."""

        def __init__(self, model: torch.nn.Module, columns: list[int]):
            super().__init__()
            encoder = model.wav2vec2.encoder
            self.norm, self.layers, self.head = encoder.layer_norm, encoder.layers, model.lm_head
            self.heads = model.config.num_attention_heads
            self.columns = torch.nn.Linear(self.head.in_features, len(columns))
            with torch.no_grad():
                self.columns.weight.copy_(self.head.weight[columns])
                self.columns.bias.copy_(self.head.bias[columns])

        def forward(self, hs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            batch, steps, dim = hs.shape
            head = dim // self.heads
            bias = ((1.0 - valid) * -1e4)[:, None, None, :]  # the padding's keys

            def split(x: torch.Tensor) -> torch.Tensor:
                return x.reshape(batch, steps, self.heads, head).transpose(1, 2)

            hs = layer_norm(self.norm, hs)
            for layer in self.layers:
                attention, ff = layer.attention, layer.feed_forward
                q, k, v = (split(attention.q_proj(hs)) * head**-0.5, split(attention.k_proj(hs)),
                           split(attention.v_proj(hs)))
                weights = torch.softmax(q @ k.transpose(-1, -2) + bias, dim=-1)
                hs = layer_norm(layer.layer_norm,
                                hs + attention.out_proj((weights @ v).transpose(1, 2).reshape(batch, steps, dim)))
                hs = layer_norm(layer.final_layer_norm,
                                hs + ff.output_dense(torch.nn.functional.gelu(ff.intermediate_dense(hs))))
            logits = self.head(hs)
            top = logits.max(dim=-1, keepdim=True).values
            return self.columns(hs) - (top + torch.log(torch.exp(logits - top).sum(dim=-1, keepdim=True)))

    model = Wav2Vec2ForCTC.from_pretrained(files, dtype=torch.float32).eval()
    body = Body(model, vocab_columns(aligner, files)).eval()
    dim = model.config.hidden_size
    with torch.no_grad():
        traced = torch.jit.trace(body, (torch.zeros(1, SHAPES[0], dim), torch.ones(1, SHAPES[0])))
    program = ct.convert(
        traced,
        inputs=[
            ct.TensorType("hs", shape=ct.EnumeratedShapes([[1, t, dim] for t in SHAPES]), dtype=np.float32),
            ct.TensorType("valid", shape=ct.EnumeratedShapes([[1, t] for t in SHAPES]), dtype=np.float32),
        ],
        outputs=[ct.TensorType("lp", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        package = Path(tmp) / "body.mlpackage"
        program.save(str(package))
        Path(ct.utils.compile_model(str(package), str(Path(tmp) / out.name))).rename(out)


_model: Any = None  # the Neural Engine process's loaded program


def _load(path: str) -> None:
    import coremltools as ct

    global _model
    _model = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    # A killed build leaves its spawned workers running; this one holds ~1 GB.
    parent = multiprocessing.parent_process().sentinel
    threading.Thread(target=lambda: (wait([parent]), os._exit(1)), daemon=True).start()


def _predict(clips: list[tuple[str, np.ndarray, int]]) -> list[np.ndarray]:
    out = []
    for name, hs, frames in clips:
        valid = (np.arange(hs.shape[1]) < frames).astype(np.float32)[None]
        log_probs = _model.predict({"hs": hs, "valid": valid})["lp"][0, :frames]
        if not np.isfinite(log_probs).all():  # float16 overflow: a NaN score would pass the Viterbi's floor
            raise FloatingPointError(f"{name}: the Neural Engine gave non-finite log-probabilities")
        out.append(log_probs)
    return out


class Wav2Vec2ANE:
    """The aligner's transformer on the Neural Engine, in its own process."""

    max_samples = MAX_BUCKETS * BUCKET_SAMPLES  # the longest clip it takes

    def __init__(self, aligner: Aligner, files: Path):
        if json.loads((files / "config.json").read_text())["do_stable_layer_norm"]:
            raise ValueError(f"{aligner.repo}: only the base (post-norm) layout runs on the Neural Engine")
        program = json.dumps([vocab_columns(aligner, files), SHAPES, LN_SCALE, inspect.getsource(convert),
                              version("coremltools")])
        key = hashlib.sha256(program.encode()).hexdigest()[:8]
        path = MODEL_DIR / f"{aligner.name}-{aligner.revision[:8]}-{key}.mlmodelc"
        if not path.exists():
            convert(aligner, files, path)
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_load,
                                        initargs=(str(path),))
        self.pool.submit(_predict, []).result()  # loaded (and compiled, the first time) before the build starts

    def submit(self, clips: list[tuple[str, np.ndarray, int]]) -> Future:
        """The [frames, columns] float32 log-probabilities of each clip, given as its name (for
        errors), `Wav2Vec2CTC.transformer_input` [1, T, dim] and frames, in order."""
        return self.pool.submit(_predict, clips)
