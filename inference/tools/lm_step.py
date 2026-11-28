"""A phone student's LM step as fixed-shape PyTorch modules, for the Apple Neural Engine: one
frame per call, the KV cache as Core ML state. Its audio comes from parallel heads
(`"audio_heads": "parallel"`, `training/student/heads.py`) or from a depformer whose last slice
emits the last codebooks at once (`"depformer_parallel"`):

    host:   x = text_emb[text] + sum of audio_embs[q][audio q]        (the tables in `embeddings`)
    Trunk:  x, cos, sin, mask, slot  ->  hidden, text_logits          (state: keys, values)
    host:   text token sampled from text_logits
    Heads:  hidden, text_emb[sampled text]  ->  audio_logits [dep_q, card]
  or, for a depformer (`DepformerStep`):
    host:   emb = slice 0's table [the sampled text]; noise = temp * Gumbel(0, 1) [dep_q, card]
    DepformerStep: hidden, emb, noise -> tokens [dep_q] (all slices + top-k sampling on ANE)

Text is sampled before the audio because the heads and the first slice read its embedding, as
`moshi_mlx.models.lm.ParallelHeads` and `DepFormer` do. The host computes the RoPE tables of the frame's
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
    depformer: Depformer | None = None  # None: parallel heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @staticmethod
    def read(model_dir: Path) -> Config:
        c = json.loads((model_dir / "config.json").read_text())
        if c.get("audio_heads") not in ("parallel", "depformer_parallel"):
            raise ValueError(f"{model_dir} is not a parallel-heads or depformer_parallel model")
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
            depformer=Depformer.read(c) if c["audio_heads"] == "depformer_parallel" else None,
        )


@dataclass(frozen=True)
class Depformer:
    """A `depformer_parallel` model's depformer: `slices` sequential steps a frame, one codebook
    each, the last emitting the last `parallel` codebooks at once. Each slice has its own weights
    (moshi's `depformer_weights_per_step`, `depformer_multi_linear`), RMS-normed layers, no
    positional embedding and a LayerNorm before its output (`depformer_norm`)."""

    dim: int
    heads: int
    layers: int
    parallel: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @staticmethod
    def read(c: dict) -> Depformer:
        parallel = c["parallel_codebooks"]
        slices = c["dep_q"] - parallel + 1
        if not 1 <= parallel <= c["dep_q"]:
            raise ValueError(f"parallel_codebooks {parallel} for dep_q {c['dep_q']}")
        if not (c["depformer_weights_per_step"] and c["depformer_multi_linear"]):
            raise ValueError("a depformer_parallel model has weights and an input linear per slice")
        # moshi's schedule, codebook -> weight set (none: one a codebook)
        if (c.get("depformer_weights_per_step_schedule") or list(range(c["dep_q"]))) != [
            min(j, slices - 1) for j in range(c["dep_q"])
        ]:
            raise ValueError("a depformer_parallel model has a slice a codebook, the last one the last codebooks'")
        if c["depformer_pos_emb"] != "none" or c.get("depformer_kv_repeat", c["kv_repeat"]) != 1:
            raise ValueError("a depformer positional embedding or grouped queries are not implemented")
        if c.get("depformer_norm") != "layer_norm" or c.get("depformer_low_rank_embeddings"):
            raise ValueError("only a depformer LayerNorm before each output and full embeddings are implemented")
        return Depformer(
            dim=c["depformer_dim"], heads=c["depformer_num_heads"], layers=c["depformer_num_layers"],
            parallel=parallel,
        )

    def slices(self, dep_q: int) -> int:
        return dep_q - self.parallel + 1


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


def depformer_text_embeddings(w: dict[str, torch.Tensor]) -> np.ndarray:
    """Slice 0's input table [text_card + 1, depformer dim] in fp16, which the host reads for
    `DepformerStep`; its later slices embed their sampled tokens inside the graph."""
    return w["depformer.slices.0.emb.weight"].to(torch.float16).numpy()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8) * weight


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """rope_concat (MLX `traditional=False`): each head's two halves rotate as pairs."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# KV-cache chunk size for online-softmax attention in `Trunk`: summing 3,000 fp16 values in one
# accumulator drifts past ~1,500 frames on the Neural Engine; splitting into chunks of 250 keeps
# fp16 headroom per partial sum with zero 6-bit latency cost (`lm-npu.md#precision`).
_ATTN_CHUNK = 250


class Trunk(nn.Module):
    """The main transformer, its output norm and the text head, one frame per call.

    x [1, dim]; cos, sin [1, head_dim / 2]; mask [1, context] (0 where a slot holds a frame this
    one sees, -1e4 elsewhere); slot [1] int32, where this frame's key and value go
    -> hidden [1, dim] (normed), text_logits [1, text_card].
    """

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor]):
        super().__init__()
        if cfg.context % _ATTN_CHUNK:
            raise ValueError(f"a context of {cfg.context} is not whole chunks of {_ATTN_CHUNK}")
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
            # Online-softmax across `.split(_ATTN_CHUNK)` (Core ML MLState rejects reshaping
            # state directly to 4-D, whereas split stays 100% on the Neural Engine).
            outs, lses = [], []
            for sc, vc in zip(scores.split(_ATTN_CHUNK, dim=-1), values.split(_ATTN_CHUNK, dim=1)):
                mc = sc.max(dim=-1, keepdim=True).values
                wc = torch.exp(sc - mc)
                lc = wc.sum(dim=-1, keepdim=True)
                outs.append((wc / lc) @ vc)
                lses.append(mc + torch.log(lc))
            cw = F.softmax(torch.cat(lses, dim=-1), dim=-1)
            attn = sum(o * cw[..., idx : idx + 1] for idx, o in enumerate(outs)).reshape(1, h * d)
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


class DepformerStep(nn.Module):
    """All slices of a `depformer_parallel` model in one Core ML call, with top-k Gumbel-max
    sampling and inter-slice embedding lookups inside the graph using Neural-Engine-native ops
    only (no `reduce_argmax` or `gather`, which would split the graph onto the CPU).

    hidden [1, dim]; emb [1, d] (slice 0's embedding of the sampled text token);
    noise [dep_q, card] (fp16: temp * Gumbel(0, 1), or zeros for greedy temp = 0)
    -> tokens [dep_q] (exact integers 0 .. card - 1 in fp16; requires card <= 2048).
    """

    def __init__(self, cfg: Config, w: dict[str, torch.Tensor], top_k: int | None = None):
        super().__init__()
        if cfg.depformer is None:
            raise ValueError("Depformer requires a depformer_parallel config")
        if cfg.card > 2048:
            raise ValueError(f"fp16 exact token indices require card <= 2048, got {cfg.card}")
        self.cfg = cfg
        self.slices = cfg.depformer.slices(cfg.dep_q)
        self.top_k = top_k if (top_k is not None and 0 < top_k < cfg.card) else None
        # Exact fp16 integers 1 .. card for tie-free one-hot selection and token readout.
        self.register_buffer("ids", torch.arange(1, cfg.card + 1, dtype=torch.float32).reshape(1, cfg.card))
        for k in range(self.slices):
            prefix = f"depformer.slices.{k}."
            for name, t in w.items():
                if not name.startswith(prefix):
                    continue
                if name == prefix + "emb.weight":
                    if k > 0:
                        # Generated tokens are in 0 .. card - 1 (never the padding row `card`),
                        # stored as [d, card] so `oh @ W.T` keeps its MIL weight name.
                        self.register_buffer(f"s{k}_emb_weight", t[: cfg.card].T.contiguous())
                    continue
                self.register_buffer(f"s{k}_{name[len(prefix):].replace('.', '_')}", t)
            codebooks = cfg.depformer.parallel if k == self.slices - 1 else 1
            if self._w(k, "linear_out.weight").shape[0] != codebooks * cfg.card:
                raise ValueError(f"slice {k}'s linear_out has {self._w(k, 'linear_out.weight').shape[0]} rows "
                                 f"for {codebooks} codebooks of {cfg.card}")

    def _w(self, k: int, name: str) -> torch.Tensor:
        return getattr(self, f"s{k}_{name.replace('.', '_')}")

    def forward(self, hidden, emb, noise):
        d = self.cfg.depformer
        h, hd, card = d.heads, d.head_dim, self.cfg.card
        past_k: list[torch.Tensor | None] = [None] * d.layers
        past_v: list[torch.Tensor | None] = [None] * d.layers
        tokens = []
        for k in range(self.slices):
            x = hidden @ self._w(k, "linear_in.weight").T + emb
            for i in range(d.layers):
                p = f"transformer.layers.{i}."
                qkv = _rms_norm(x, self._w(k, p + "norm1.weight")) @ self._w(k, p + "self_attn.in_proj.weight").T
                q, key, value = (t.reshape(h, 1, hd) for t in qkv.chunk(3, -1))
                if past_k[i] is None:
                    past_k[i], past_v[i] = key, value
                else:
                    past_k[i] = torch.cat([past_k[i], key], dim=1)
                    past_v[i] = torch.cat([past_v[i], value], dim=1)
                attn = (F.softmax(q @ past_k[i].transpose(1, 2) / math.sqrt(hd), dim=-1) @ past_v[i]).reshape(1, d.dim)
                x = x + attn @ self._w(k, p + "self_attn.out_proj.weight").T
                gate, up = (_rms_norm(x, self._w(k, p + "norm2.weight")) @ self._w(k, p + "gating.linear_in.weight").T).chunk(2, -1)
                x = x + (F.silu(gate) * up) @ self._w(k, p + "gating.linear_out.weight").T
            x = F.layer_norm(x, (d.dim,), self._w(k, "norm.weight"), self._w(k, "norm.bias"), 1e-5)
            out = self._w(k, "linear_out.weight")
            ncb = d.parallel if k == self.slices - 1 else 1
            # One matmul a codebook: the Neural Engine takes none as wide as several together.
            logits = torch.cat([x @ out[j * card : (j + 1) * card].T for j in range(ncb)], dim=0)
            # Center at max = 0 for peak fp16 precision around the top logits and safe -1e4 masking.
            logits = logits - logits.max(dim=-1, keepdim=True).values
            nk = noise[k : k + ncb]
            if self.top_k is not None:
                vals, _ = torch.topk(logits, self.top_k, dim=-1)
                thresh = vals[:, -1:]
                pert = torch.where(logits >= thresh, logits + nk, torch.full_like(logits, -1e4))
            else:
                pert = logits + nk
            m = pert.max(dim=-1, keepdim=True).values
            tied = (pert >= m).to(x.dtype)
            scored = tied * self.ids
            winner = scored.max(dim=-1, keepdim=True).values
            tokens.append((winner - 1.0).reshape(ncb))
            if k < self.slices - 1:
                oh = (scored >= winner).to(x.dtype)
                emb = oh @ self._w(k + 1, "emb.weight").T
        return torch.cat(tokens, dim=0)


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
