"""Faster moshi LM for training and generation: the same modules and weights, the same math.

`install()` (called when `common` is imported) replaces four things in moshi 0.2.13:

- Causal attention. moshi passes SDPA a boolean mask, which rules out the flash kernel; at
  offset 0 that mask is exactly lower-triangular, so it becomes `is_causal=True`.
- Grouped-query attention. The main transformer has 8 key/value heads for 16 query heads, and
  moshi copies each key and value to 16 heads before attending: in generation that is the whole
  3,000-frame KV cache, twice per layer and step (44% of a generation step at batch 16). SDPA
  groups the heads itself with `enable_gqa=True`.
- The depformer's training forward. moshi runs its 16 codebook steps as slices `x[:, t:t+1]` of
  a `[B*T, 16, C]` tensor, three times per layer, and the backward pass of every slice allocates
  and zero-fills a full-size gradient before adding it in. Here the activations are step-major,
  `[16, B*T, C]`: a step is a view from `unbind`, steps join with `stack`, and the backward pass
  of both is one copy. Steps that share a `depformer_in` projection share its output. The layers
  run compiled (dynamic shapes, since batches vary), which fuses the norms, casts and residuals.
- The gating FFN in training. `x[..., 0, :]` and `x[..., 1, :]` become one `unbind`, for the
  same reason as the depformer's slices.

    moshi (per layer, 16 steps)            here
    x[:, t:t+1] ─► linear_t ─┐ ×16         x[16, N, C] ─unbind─► linear_t ─┐ ×16
    backward: 16 zero-filled grads + adds  backward: one stack ───────────┘

Generation (`LMGen`) keeps moshi's own depformer and gating.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from moshi.models.lm import LMModel
from moshi.modules import transformer as moshi_transformer
from moshi.modules.gating import ActivationGating
from torch.nn.attention import SDPBackend, sdpa_kernel


class _CausalSDPA:
    """Namespace proxy injected as moshi.modules.transformer.F (rebinds only that module's
    global, not torch.nn.functional itself).

    In training (offset 0, T << cfg.context) moshi's boolean attention mask is exactly
    lower-triangular, so it becomes is_causal=True, which the flash kernel accepts. The verdict
    is checked once per mask shape and cached (checking every call would sync the host per
    layer); every same-shape mask moshi builds at offset 0 is identical, and streaming decode has
    T_q=1 != T_k so it falls through. Keys and values with fewer heads than the queries are
    grouped (`_keep_kv_heads`).
    """

    def __init__(self):
        self._causal_shapes: dict[tuple[int, ...], bool] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(F, name)

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kwargs):
        kwargs["enable_gqa"] = q.shape[-3] != k.shape[-3]
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and q.shape[-2] == k.shape[-2]
            and attn_mask.shape[-2:] == (q.shape[-2], k.shape[-2])
        ):
            key = tuple(attn_mask.shape)
            causal = self._causal_shapes.get(key)
            if causal is None:
                T = q.shape[-2]
                tril = torch.ones(T, T, dtype=torch.bool, device=attn_mask.device).tril()
                causal = bool((attn_mask == tril).all())
                self._causal_shapes[key] = causal
            if causal:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs)
        return F.scaled_dot_product_attention(q, k, v, attn_mask, **kwargs)


def _keep_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """moshi's expand_repeated_kv without the copy: SDPA groups query heads over these, in the
    same order (kv head h serves query heads h*n_rep .. h*n_rep + n_rep - 1)."""
    return x


def _gating(module: ActivationGating, x: torch.Tensor) -> torch.Tensor:
    gate, value = F.linear(x, module.linear_in.weight).unflatten(-1, (2, -1)).unbind(-2)
    return F.linear(module.activation(gate) * value, module.linear_out.weight)


_moshi_gating_forward = ActivationGating.forward


def _gating_forward(self: ActivationGating, x: torch.Tensor) -> torch.Tensor:
    if self.training and isinstance(self.linear_in, torch.nn.Linear):
        return _gating(self, x)
    return _moshi_gating_forward(self, x)


def _schedule(owner: Any, steps: int) -> list[int]:
    schedule = owner.weights_per_step_schedule
    return list(schedule) if schedule is not None else list(range(steps))


def _per_step(modules: Any, schedule: list[int], xs: torch.Tensor, function: Any) -> torch.Tensor:
    """xs [K, N, C]: step k through function(modules[schedule[k]], step)."""
    return torch.stack([function(modules[index], x) for index, x in zip(schedule, xs.unbind(0))])


def _linear(module: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def _depformer_layer(layer: Any, xs: torch.Tensor) -> torch.Tensor:
    """One moshi StreamingTransformerLayer on step-major xs [K, N, C], without streaming."""
    steps, n, channels = xs.shape
    attention = layer.self_attn
    schedule = _schedule(attention, steps)
    projected = _per_step(attention.in_projs, schedule, layer.norm1(xs), _linear)  # [K, N, 3C]
    q, k, v = projected.view(steps, n, 3, attention.num_heads, -1).permute(2, 1, 3, 0, 4)
    # 16 positions per sequence and N sequences: the memory-efficient kernel's backward is faster
    # here than flash's (measured on the H100).
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [N, H, K, D]
    out = out.permute(2, 0, 1, 3).reshape(steps, n, channels)
    update = _per_step(attention.out_projs, schedule, out, _linear)
    xs = xs.to(update) + layer.layer_scale_1(update)
    update = _per_step(layer.gating, _schedule(layer, steps), layer.norm2(xs), _gating)
    return xs.to(update) + layer.layer_scale_2(update)


def _depformer_layers(depformer: Any, xs: torch.Tensor) -> torch.Tensor:
    for layer in depformer.layers:
        xs = _depformer_layer(layer, xs)
    return xs


# Compiled on first call; every batch shape shares one dynamic graph.
_compiled_depformer_layers = torch.compile(_depformer_layers, dynamic=True)


def _check_depformer(depformer: Any) -> None:
    if depformer._streaming_state is not None or depformer.checkpointing:
        raise RuntimeError("The training depformer runs without streaming or checkpointing")
    if depformer.positional_embedding in {"sin", "sin_rope"}:
        raise RuntimeError("The training depformer has no positional embedding")
    for layer in depformer.layers:
        attention = layer.self_attn
        if attention.kv_repeat != 1 or attention.rope is not None or attention.cross_attention:
            raise RuntimeError("Unsupported depformer attention")
        if layer.cross_attention is not None or layer.skip_self_attn or layer.gating is None:
            raise RuntimeError("Unsupported depformer layer")


def forward_depformer_training(
    self: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor
) -> torch.Tensor:
    """moshi's LMModel.forward_depformer_training: logits [B, dep_q, T, card]."""
    B, K, T = sequence.shape
    if K != self.num_codebooks:
        raise RuntimeError(f"Depformer training needs all {self.num_codebooks} codebooks, got {K}")
    _check_depformer(self.depformer)
    steps = self.dep_q
    projections: dict[int, torch.Tensor] = {}
    inputs = []
    for cb in range(steps):
        index = 0
        if self.depformer_multi_linear:
            index = cb
            if self.depformer_weights_per_step_schedule is not None:
                index = self.depformer_weights_per_step_schedule[cb]
        if index not in projections:
            projections[index] = self.depformer_in[index](transformer_out)
        if cb == 0:
            token_in = self.depformer_text_emb(sequence[:, 0])
        else:
            token_in = self.depformer_emb[cb - 1](sequence[:, cb + self.audio_offset - 1])
        inputs.append(token_in + projections[index])
    xs = torch.stack(inputs, 0).view(steps, B * T, -1)
    out = _compiled_depformer_layers(self.depformer, xs).to(xs.dtype).unbind(0)
    logits = [
        self.linears[cb](self.depformer_norms[cb](out[cb])).view(B, T, -1) for cb in range(steps)
    ]
    return torch.stack(logits, 1)


def install() -> None:
    moshi_transformer.F = _CausalSDPA()
    moshi_transformer.expand_repeated_kv = _keep_kv_heads
    LMModel.forward_depformer_training = forward_depformer_training
    ActivationGating.forward = _gating_forward
