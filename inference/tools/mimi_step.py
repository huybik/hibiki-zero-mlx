"""Mimi's streaming step as a pure function of explicit state, for neural accelerators.

    encoder:  pcm [1, 1, 1920 k], state, mask, cos, sin, first  ->  codes [1, 16, k], state
    decoder:  codes [1, n, k], state, mask, cos, sin            ->  pcm [1, 1, 1920 k], state

k frames per call, fixed when the module is built. Every op is a convolution, matmul, norm,
softmax or elementwise op on fixed shapes, and the state goes in and out as three tensors the
host feeds back unchanged: the convolutions' left context and the transposed convolutions'
overlap packed in `conv`, and the transformer's keys and values of the last `context - 1` steps
in `keys` and `values`. So the same module converts to Core ML for the Apple Neural Engine and,
through ONNX, to Qualcomm's NPU. The host computes the attention mask and the RoPE tables from
the step count (`Inputs`), so positions, which grow without bound, never enter a half-precision
graph.

Matches moshi 0.2.13's PyTorch Mimi (exact GELU; SEANet convolutions left-padded with zeros, the
25 -> 12.5 Hz downsample with the first value; the transformer causal over 250 steps). The
query and key features of each head are reordered from RoPE's interleaved pairs to two halves
(rows of `in_proj` permuted), which leaves attention unchanged and keeps every tensor 4-D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

FRAME = 1920  # samples per 12.5 Hz frame at 24 kHz
RATIOS = (4, 5, 6, 8)  # SEANet encoder strides; the decoder runs them reversed
DIM, HEADS, HEAD_DIM, LAYERS, CONTEXT = 512, 8, 64, 8, 250
WINDOW = CONTEXT - 1  # past steps each transformer keeps
STEPS_PER_FRAME = 2  # transformer steps (25 Hz) per frame
CODEBOOK_EPS = 1e-5


class _Conv(nn.Module):
    """Causal conv1d whose left context of `ctx` input steps is part of the packed state."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor | None, stride: int = 1):
        super().__init__()
        self.register_buffer("weight", w)
        self.register_buffer("bias", b)
        self.stride = stride
        self.channels = w.shape[1]
        self.ctx = w.shape[2] - stride

    def forward(self, x: torch.Tensor, state: torch.Tensor | None):
        if self.ctx == 0:
            return F.conv1d(x, self.weight, self.bias), None
        xs = torch.cat([state, x], dim=-1)
        return F.conv1d(xs, self.weight, self.bias, stride=self.stride), xs[..., -self.ctx :]


class _ConvTr(nn.Module):
    """Transposed conv1d, kernel = 2 x stride; its overlap with the next call (`stride` steps,
    without the bias) is part of the packed state."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor | None, stride: int, groups: int = 1):
        super().__init__()
        self.register_buffer("weight", w)
        self.register_buffer("bias", b)
        self.stride, self.groups = stride, groups
        self.channels = w.shape[1] * groups
        self.ctx = stride

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        s, t = self.stride, x.shape[-1]
        y = F.conv_transpose1d(x, self.weight, stride=s, groups=self.groups)  # (t + 1) s steps
        y = torch.cat([y[..., :s] + state, y[..., s:]], dim=-1)
        out = y[..., : t * s]
        if self.bias is not None:
            out = out + self.bias[:, None]
        return out, y[..., t * s :]


class _Transformer(nn.Module):
    """Mimi's 8-layer transformer over conv-layout input [1, 512, T], attending to the host's
    window of the last 249 steps' keys and values plus this call's."""

    def __init__(self, w: dict[str, torch.Tensor], prefix: str):
        super().__init__()
        # Interleaved RoPE pairs (2i, 2i + 1) -> halves (i, i + 32), within each head.
        half = torch.arange(HEAD_DIM // 2)
        perm = torch.cat([2 * half, 2 * half + 1])
        rows = torch.cat([h * HEAD_DIM + perm for h in range(HEADS)])
        self.layers = nn.ModuleList()
        for i in range(LAYERS):
            p = f"{prefix}.transformer.layers.{i}."
            layer = nn.Module()
            q, k, v = w[p + "self_attn.in_proj_weight"].split(DIM)
            for name, tensor in (
                ("q", q[rows]),
                ("k", k[rows]),
                ("v", v),
                ("out", w[p + "self_attn.out_proj.weight"]),
                ("ff1", w[p + "linear1.weight"]),
                ("ff2", w[p + "linear2.weight"]),
                ("norm1_w", w[p + "norm1.weight"]),
                ("norm1_b", w[p + "norm1.bias"]),
                ("norm2_w", w[p + "norm2.weight"]),
                ("norm2_b", w[p + "norm2.bias"]),
                ("scale1", w[p + "layer_scale_1.scale"]),
                ("scale2", w[p + "layer_scale_2.scale"]),
            ):
                layer.register_buffer(name, tensor)
            self.layers.append(layer)

    def forward(self, x, keys, values, mask, cos, sin):
        steps = x.shape[-1]
        x = x.transpose(1, 2)  # [1, T, C]
        cos = torch.cat([cos, cos], dim=-1)  # [T, D]
        sin = torch.cat([sin, sin], dim=-1)
        heads = lambda t: t.reshape(1, steps, HEADS, HEAD_DIM).transpose(1, 2)  # [1, H, T, D]
        rope = lambda t: t * cos + torch.cat([-t[..., HEAD_DIM // 2 :], t[..., : HEAD_DIM // 2]], -1) * sin
        new_keys, new_values = [], []
        for i, p in enumerate(self.layers):
            h = F.layer_norm(x, (DIM,), p.norm1_w, p.norm1_b, 1e-5)
            q, k, v = rope(heads(h @ p.q.T)), rope(heads(h @ p.k.T)), heads(h @ p.v.T)
            all_k = torch.cat([keys[i : i + 1], k], dim=2)  # [1, H, W + T, D]
            all_v = torch.cat([values[i : i + 1], v], dim=2)
            scores = (q @ all_k.transpose(2, 3)) * HEAD_DIM**-0.5 + mask
            attn = (scores.softmax(dim=-1) @ all_v).transpose(1, 2).reshape(1, steps, DIM)
            x = x + p.scale1 * (attn @ p.out.T)
            h = F.layer_norm(x, (DIM,), p.norm2_w, p.norm2_b, 1e-5)
            x = x + p.scale2 * (F.gelu(h @ p.ff1.T) @ p.ff2.T)
            new_keys.append(all_k[:, :, steps:])
            new_values.append(all_v[:, :, steps:])
        return x.transpose(1, 2), torch.cat(new_keys), torch.cat(new_values)


def _codebooks(w: dict[str, torch.Tensor], count: int) -> list[torch.Tensor]:
    books = []
    for i in range(count):
        c = (
            "quantizer.rvq_first.vq.layers.0._codebook"
            if i == 0
            else f"quantizer.rvq_rest.vq.layers.{i - 1}._codebook"
        )
        books.append(w[c + ".embedding_sum"] / w[c + ".cluster_usage"].clamp(min=CODEBOOK_EPS)[:, None])
    return books


class _Stateful(nn.Module):
    """Packs the streaming layers' conv states into one [1, size] tensor, in layer order."""

    def _register(self, layers: list[nn.Module]) -> None:
        self.size = 0
        self.slots = []
        for layer in layers:
            if layer.ctx:
                self.slots.append((self.size, layer.channels, layer.ctx))
                self.size += layer.channels * layer.ctx

    def _unpack(self, conv: torch.Tensor) -> list[torch.Tensor]:
        return [conv[:, o : o + c * n].reshape(1, c, n) for o, c, n in self.slots]

    @staticmethod
    def _pack(states: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([s.reshape(1, -1) for s in states], dim=1)


class EncoderStep(_Stateful):
    def __init__(self, w: dict[str, torch.Tensor], frames: int, codebooks: int = 16):
        super().__init__()
        self.frames, self.codebooks = frames, codebooks
        conv = lambda name, stride=1: _Conv(
            w[f"{name}.conv.conv.weight"], w.get(f"{name}.conv.conv.bias"), stride
        )
        self.init = conv("encoder.model.0")
        self.blocks = nn.ModuleList()
        for (res, down), stride in zip(((1, 3), (4, 6), (7, 9), (10, 12)), RATIOS):
            self.blocks.append(
                nn.ModuleList(
                    [
                        conv(f"encoder.model.{res}.block.1"),
                        conv(f"encoder.model.{res}.block.3"),
                        conv(f"encoder.model.{down}", stride),
                    ]
                )
            )
        self.final = conv("encoder.model.14")
        self.transformer = _Transformer(w, "encoder_transformer")
        self.downsample = _Conv(w["downsample.conv.conv.conv.weight"], None, stride=2)
        self.register_buffer("proj_first", w["quantizer.rvq_first.input_proj.weight"][:, :, 0])
        self.register_buffer("proj_rest", w["quantizer.rvq_rest.input_proj.weight"][:, :, 0])
        books = _codebooks(w, codebooks)
        self.register_buffer("books", torch.stack(books))  # [Q, 2048, 256]
        self.register_buffer("half_norms", torch.stack([(b * b).sum(-1) / 2 for b in books]))
        self.convs = [self.init, *[c for b in self.blocks for c in b], self.final, self.downsample]
        self._register(self.convs)

    def forward(self, pcm, conv, keys, values, mask, cos, sin, first):
        states = iter(self._unpack(conv))
        new_states = []

        def run(layer, x):
            y, s = layer(x, next(states) if layer.ctx else None)
            if s is not None:
                new_states.append(s)
            return y

        x = run(self.init, pcm)
        for res1, res2, down in self.blocks:
            x = x + run(res2, F.elu(run(res1, F.elu(x))))
            x = run(down, F.elu(x))
        x = run(self.final, F.elu(x))
        x, new_keys, new_values = self.transformer(x, keys, values, mask, cos, sin)
        # The downsample's left context starts as copies of its first input (edge padding).
        state = next(states)
        state = first * x[..., :1].expand_as(state) + (1 - first) * state
        y, s = self.downsample(x, state)
        new_states.append(s)

        codes = []
        for i in range(self.codebooks):
            if i < 2:
                residual = (self.proj_first if i == 0 else self.proj_rest) @ y[0]  # [256, k]
            index = (self.half_norms[i][:, None] - self.books[i] @ residual).argmin(dim=0)
            residual = residual - self.books[i][index].T
            codes.append(index)
        codes = torch.stack(codes)[None].to(torch.int32)  # [1, Q, k]
        return codes, self._pack(new_states), new_keys, new_values


class DecoderStep(_Stateful):
    def __init__(self, w: dict[str, torch.Tensor], frames: int, codebooks: int):
        super().__init__()
        self.frames, self.codebooks = frames, codebooks
        self.register_buffer("books", torch.stack(_codebooks(w, codebooks)))  # [Q, 2048, 256]
        self.register_buffer("proj_first", w["quantizer.rvq_first.output_proj.weight"][:, :, 0])
        self.register_buffer("proj_rest", w["quantizer.rvq_rest.output_proj.weight"][:, :, 0])
        self.upsample = _ConvTr(w["upsample.convtr.convtr.convtr.weight"], None, 2, groups=DIM)
        self.transformer = _Transformer(w, "decoder_transformer")
        conv = lambda name: _Conv(w[f"{name}.conv.conv.weight"], w[f"{name}.conv.conv.bias"])
        self.init = conv("decoder.model.0")
        self.blocks = nn.ModuleList()
        for (up, res), stride in zip(((2, 3), (5, 6), (8, 9), (11, 12)), reversed(RATIOS)):
            self.blocks.append(
                nn.ModuleList(
                    [
                        _ConvTr(
                            w[f"decoder.model.{up}.convtr.convtr.weight"],
                            w[f"decoder.model.{up}.convtr.convtr.bias"],
                            stride,
                        ),
                        conv(f"decoder.model.{res}.block.1"),
                        conv(f"decoder.model.{res}.block.3"),
                    ]
                )
            )
        self.final = conv("decoder.model.14")
        self.convs = [self.upsample, self.init, *[c for b in self.blocks for c in b], self.final]
        self._register(self.convs)

    def forward(self, codes, conv, keys, values, mask, cos, sin):
        states = iter(self._unpack(conv))
        new_states = []

        def run(layer, x):
            y, s = layer(x, next(states) if layer.ctx else None)
            if s is not None:
                new_states.append(s)
            return y

        first = self.books[0][codes[0, 0]]  # [k, 256]
        rest = sum(self.books[i][codes[0, i]] for i in range(1, self.codebooks))
        x = (self.proj_first @ first.T + self.proj_rest @ rest.T)[None]  # [1, 512, k]
        x = run(self.upsample, x)
        x, new_keys, new_values = self.transformer(x, keys, values, mask, cos, sin)
        x = run(self.init, x)
        for up, res1, res2 in self.blocks:
            x = run(up, F.elu(x))
            x = x + run(res2, F.elu(run(res1, F.elu(x))))
        pcm = run(self.final, F.elu(x))
        return pcm, self._pack(new_states), new_keys, new_values


@dataclass
class Inputs:
    """The per-call inputs the host derives from the number of transformer steps already run:
    the additive attention mask over [window, new steps], the RoPE tables and the first-call
    flag. Window slot j holds step `offset - WINDOW + j`; a query at step s sees steps
    s - 249 ... s, and slots before step 0 are empty."""

    mask: np.ndarray  # [T, WINDOW + T]
    cos: np.ndarray  # [T, HEAD_DIM / 2]
    sin: np.ndarray
    first: np.ndarray  # [1], 1.0 on a session's first call

    @staticmethod
    def at(offset: int, steps: int) -> Inputs:
        query = offset + np.arange(steps)[:, None]
        key = np.concatenate([offset - WINDOW + np.arange(WINDOW), offset + np.arange(steps)])
        delta = query - key[None]
        visible = (delta >= 0) & (delta < CONTEXT) & (key[None] >= 0)
        freqs = np.exp(np.arange(HEAD_DIM // 2) * (-math.log(10000.0) * 2 / HEAD_DIM))
        angles = query * freqs[None]  # float64: exact for any session length
        return Inputs(
            mask=np.where(visible, 0.0, -1e4).astype(np.float32),
            cos=np.cos(angles).astype(np.float32),
            sin=np.sin(angles).astype(np.float32),
            first=np.array([1.0 if offset == 0 else 0.0], np.float32),
        )


def load(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    return {k: v.float() for k, v in load_file(str(checkpoint)).items()}


def initial_state(step: _Stateful) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A new session's conv state and empty key/value windows."""
    window = torch.zeros(LAYERS, HEADS, WINDOW, HEAD_DIM)
    return torch.zeros(1, step.size), window, window.clone()
