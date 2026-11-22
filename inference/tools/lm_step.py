"""The parallel-heads student's LM step (`training/student/heads.py`) as fixed-shape PyTorch modules,
for the Apple Neural Engine: one frame per call, the KV cache as Core ML state.

    host:   x = text_emb[text] + sum of audio_embs[q][audio q]        (the tables in `embeddings`)
    Trunk:  x, cos, sin, mask, slot  ->  hidden, text_logits          (state: keys, values)
    host:   text token sampled from text_logits
    Heads:  hidden, text_emb[sampled text]  ->  audio_logits [dep_q, card]

Text is sampled between the two calls because the heads read its embedding, as
`moshi_mlx.models.lm.ParallelHeads` does. The host computes the RoPE tables of the frame's
position and the mask of its cache slots (`Inputs`), so positions, which grow without bound,
never enter a half-precision graph. The cache is a ring of `context` slots: frame t writes slot
t mod context and attends to the frames t - context + 1 .. t, as moshi's training does.

Loads a model directory staged by `convert_mlx.py` in bf16 (its MLX weight names) and computes in
fp32 here; the export casts to fp16.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


@dataclass(frozen=True)
class Config:
    dim: int
    heads: int
    kv_heads: int
    layers: int
    context: int
    max_period: float
    card: int
    text_card: int
    n_q: int
    dep_q: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @staticmethod
    def read(model_dir: Path) -> Config:
        c = json.loads((model_dir / "config.json").read_text())
        if c.get("audio_heads") != "parallel":
            raise ValueError(f"{model_dir} is not a parallel-heads model")
        if c["norm"] != "rms_norm_f32" or c["positional_embedding"] != "rope_concat" or c["gating"] != "silu":
            raise ValueError("only RMS norms, rope_concat and SiLU gating are implemented")
        if c.get("layer_scale") is not None or c.get("cross_attention") or c.get("conditioners"):
            raise ValueError("layer scale, cross-attention and conditioners are not implemented")
        if c.get("demux_second_stream") or c.get("text_card_out") is not None:
            raise ValueError("a demuxed or separate text output vocabulary is not implemented")
        return Config(
            dim=c["dim"], heads=c["num_heads"], kv_heads=c["num_heads"] // c["kv_repeat"],
            layers=c["num_layers"], context=c["context"], max_period=c["max_period"],
            card=c["card"], text_card=c["text_card"], n_q=c["n_q"], dep_q=c["dep_q"],
        )


def load(model_dir: Path) -> tuple[Config, dict[str, torch.Tensor]]:
    """The config and the fp32 weights of a bf16 model directory."""
    cfg = Config.read(model_dir)
    name = json.loads((model_dir / "config.json").read_text())["moshi_name"]
    w = load_file(str(model_dir / name))
    if any(k.endswith(".scales") for k in w):
        raise ValueError(f"{model_dir} is quantized: export from the bf16 staging")
    return cfg, {k: v.float() for k, v in w.items()}


def embeddings(cfg: Config, w: dict[str, torch.Tensor]) -> np.ndarray:
    """The input tables the host sums, fp16, in order: text_emb [text_card + 1, dim], then
    audio_embs 0 .. n_q - 1, each [card + 1, dim]."""
    tables = [w["text_emb.weight"]] + [w[f"audio_embs.{q}.weight"] for q in range(cfg.n_q)]
    return torch.cat(tables).to(torch.float16).numpy()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * weight


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """rope_concat (MLX `traditional=False`): each head's two halves rotate as pairs."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Trunk(nn.Module):
    """The main transformer, its output norm and the text head, one frame per call.

    x [1, dim]; cos, sin [1, head_dim / 2]; mask [1, context] (0 where a slot holds a frame this
    one sees, -1e4 elsewhere); slot [1] int32, where this frame's key and value go
    -> hidden [1, dim] (normed), text_logits [1, text_card].
    """

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        for name, t in w.items():
            if name.startswith("transformer.") or name in ("out_norm.weight", "text_linear.weight"):
                self.register_buffer(name.replace(".", "_"), t)
        shape = (cfg.layers, cfg.kv_heads, cfg.context, cfg.head_dim)
        self.register_buffer("keys", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("values", torch.zeros(shape, dtype=torch.float16))

    def _w(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "_"))

    def forward(self, x, cos, sin, mask, slot):
        cfg = self.cfg
        h, hkv, d = cfg.heads, cfg.kv_heads, cfg.head_dim
        for i in range(cfg.layers):
            p = f"transformer.layers.{i}."
            qkv = _rms_norm(x, self._w(p + "norm1.weight")) @ self._w(p + "self_attn.in_proj.weight").T
            q = _rope(qkv[:, : h * d].reshape(h, d), cos, sin)
            k = _rope(qkv[:, h * d : (h + hkv) * d].reshape(hkv, d), cos, sin)
            v = qkv[:, (h + hkv) * d :].reshape(hkv, d)
            self.keys[i : i + 1, :, slot : slot + 1] = k.reshape(1, hkv, 1, d).to(self.keys.dtype)
            self.values[i : i + 1, :, slot : slot + 1] = v.reshape(1, hkv, 1, d).to(self.values.dtype)
            keys, values = self.keys[i].to(x.dtype), self.values[i].to(x.dtype)
            # Grouped queries: query head j reads key/value head j // (h / hkv).
            scores = q.reshape(hkv, h // hkv, d) @ keys.transpose(1, 2) / math.sqrt(d) + mask
            attn = (F.softmax(scores, dim=-1) @ values).reshape(1, h * d)
            x = x + attn @ self._w(p + "self_attn.out_proj.weight").T
            gate, up = (_rms_norm(x, self._w(p + "norm2.weight")) @ self._w(p + "gating.linear_in.weight").T).chunk(2, -1)
            x = x + (F.silu(gate) * up) @ self._w(p + "gating.linear_out.weight").T
        hidden = _rms_norm(x, self._w("out_norm.weight"))
        return hidden, hidden @ self._w("text_linear.weight").T


class Heads(nn.Module):
    """The 16 linear heads: hidden [1, dim], text [1, dim] (the sampled text token's text_emb row)
    -> audio_logits [dep_q, card]."""

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor]):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("weight", w["depformer.linear.weight"])

    def forward(self, hidden, text):
        x, card = hidden + text, self.cfg.card
        # One matmul a head: the Neural Engine takes none as wide as all 16 together.
        return torch.cat([x @ self.weight[j * card : (j + 1) * card].T for j in range(self.cfg.dep_q)])


@dataclass
class Inputs:
    """The host's per-frame inputs to `Trunk` at frame t (from 0), fp16 as the model takes them and
    rounded as `NeuralLM` rounds them (Double to fp16)."""

    cos: np.ndarray
    sin: np.ndarray
    mask: np.ndarray
    slot: np.ndarray

    @staticmethod
    def at(cfg: Config, t: int) -> Inputs:
        half = cfg.head_dim // 2
        angle = t * np.exp(-np.log(cfg.max_period) * np.arange(half, dtype=np.float64) / half)
        mask = np.zeros((1, cfg.context), np.float16)
        if t < cfg.context - 1:
            mask[0, t + 1 :] = -1e4  # slots not written yet
        return Inputs(
            cos=np.cos(angle)[None].astype(np.float16),
            sin=np.sin(angle)[None].astype(np.float16),
            mask=mask,
            slot=np.array([t % cfg.context], np.int32),
        )
