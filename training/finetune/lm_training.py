"""Faster moshi LM for training and generation: the same modules and weights, the same math.

`install()` (called when `common` is imported) replaces five things in moshi 0.2.13:

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
- The main transformer's training forward. Its 28 layers run as one compiled function with
  dynamic shapes, with moshi's RMS norms and rope inline and `is_causal` attention (training is
  at offset 0, T within its 3,000-frame context). Compiling moshi's own forward instead
  recompiles for every batch shape, until dynamo's limit leaves new shapes eager.

    moshi (per layer, 16 steps)            here
    x[:, t:t+1] ─► linear_t ─┐ ×16         x[16, N, C] ─unbind─► linear_t ─┐ ×16
    backward: 16 zero-filled grads + adds  backward: one stack ───────────┘

In eval mode (teacher-forced validation, generation) the main transformer is moshi's own, and
generation (`LMGen`) also keeps moshi's depformer and gating. `install()` also lets each
compiled function keep 64 graphs instead of dynamo's 8, so none of moshi's runs eagerly.
"""
from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn.functional as F
from moshi.models.lm import LMModel
from moshi.modules import transformer as moshi_transformer
from moshi.modules.gating import ActivationGating
from moshi.modules.transformer import RMSNorm, StreamingTransformer
from torch.nn.attention import SDPBackend, sdpa_kernel


class _CausalSDPA:
    """Namespace proxy injected as moshi.modules.transformer.F (rebinds only that module's
    global, not torch.nn.functional itself).

    At offset 0 (teacher-forced validation, and Mimi's encoder) with T within the context, moshi's
    boolean attention mask is exactly lower-triangular, so it becomes is_causal=True, which the
    flash kernel accepts; Mimi's windowed masks are not and fall through. The verdict
    is checked once per query and mask shape and cached (checking every call would sync the host
    per layer). At offset 0 a model's mask depends only on T and its context, and the models
    here differ in heads as well as context (Mimi: 8 heads of 64, 250 frames; the LM: 16 of 128,
    3,000), so a shape's mask is fixed. Streaming decode has T_q=1 != T_k, so it falls through.
    Keys and values with fewer heads than the queries are grouped (`_keep_kv_heads`).
    """

    def __init__(self):
        self._causal_shapes: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] = {}

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
            key = (tuple(q.shape), tuple(attn_mask.shape))
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


def _rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """moshi's `_rms_norm`, traced inline rather than through its own compiled function."""
    y = x if norm.dtype is None else x.to(norm.dtype)
    variance = norm.eps + torch.mean(y**2, dim=2, keepdim=True)
    return (y * (norm.alpha.to(variance) * torch.rsqrt(variance))).to(x.dtype)


def _rope(q: torch.Tensor, k: torch.Tensor, max_period: float) -> tuple[torch.Tensor, torch.Tensor]:
    """moshi's `apply_rope` at offset 0 on [B, H, T, D], halves concatenated (`rope_concat`)."""
    steps, dim = q.shape[-2], q.shape[-1]
    ds = torch.arange(dim // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / dim))
    ts = torch.arange(steps, device=q.device, dtype=torch.float32).view(-1, 1)
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        xr, xi = x[..., : dim // 2].float(), x[..., dim // 2 :].float()
        real, imaginary = xr * rotr - xi * roti, xr * roti + xi * rotr
        return torch.cat([real.to(q.dtype), imaginary.to(q.dtype)], -1)

    return rotate(q), rotate(k)


def _transformer_layer(layer: Any, x: torch.Tensor, max_period: float) -> torch.Tensor:
    """One moshi StreamingTransformerLayer on x [B, T, C] at offset 0, without streaming."""
    attention = layer.self_attn
    channels = x.shape[-1]
    kv_heads = attention.num_heads // attention.kv_repeat
    projected = attention.in_projs[0](_rms_norm(layer.norm1, x))
    q = projected[..., :channels].unflatten(-1, (attention.num_heads, -1)).transpose(1, 2)
    k, v = projected[..., channels:].unflatten(-1, (2, kv_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k = _rope(q, k, max_period)
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=attention.kv_repeat != 1
    )  # [B, H, T, D]
    update = attention.out_projs[0](out.transpose(1, 2).flatten(2))
    x = x.to(update) + layer.layer_scale_1(update)
    update = _gating(layer.gating, _rms_norm(layer.norm2, x))
    return x.to(update) + layer.layer_scale_2(update)


def _transformer_layers(transformer: Any, x: torch.Tensor) -> torch.Tensor:
    out = x
    for layer in transformer.layers:
        out = _transformer_layer(layer, out, transformer.rope.max_period)
    return out.to(x.dtype)


# Compiled on first call with dynamic shapes: two graphs, as batches under 4,096 frames get
# their own (an inductor fusion guard).
_compiled_transformer_layers = torch.compile(_transformer_layers, dynamic=True)


def _check_transformer(transformer: Any, steps: int, args: tuple, kwargs: dict) -> None:
    if transformer._streaming_state is not None or transformer.checkpointing:
        raise RuntimeError("The training transformer runs without streaming or checkpointing")
    if args or kwargs.get("cross_attention_src") is not None:
        raise RuntimeError("The training transformer has no cross-attention")
    if transformer.positional_embedding != "rope_concat":
        raise RuntimeError("The training transformer needs rope_concat positions")
    for layer in transformer.layers:
        if layer.cross_attention is not None or layer.skip_self_attn or layer.weights_per_step:
            raise RuntimeError("Unsupported transformer layer")
        attention = layer.self_attn
        if not attention.causal or attention.cross_attention or len(attention.in_projs) != 1:
            raise RuntimeError("Unsupported transformer attention")
        if attention.context is not None and steps > attention.context:
            raise RuntimeError(f"{steps} frames exceed the attention context {attention.context}")
        gating = layer.gating
        if not isinstance(gating, ActivationGating) or not (
            isinstance(gating.linear_in, torch.nn.Linear)
            and isinstance(gating.linear_out, torch.nn.Linear)
        ):
            raise RuntimeError("Unsupported transformer feed-forward")
        if not (isinstance(layer.norm1, RMSNorm) and isinstance(layer.norm2, RMSNorm)):
            raise RuntimeError("Unsupported transformer norms")


_moshi_transformer_forward = StreamingTransformer.forward


def _transformer_forward(self: StreamingTransformer, x: torch.Tensor, *args, **kwargs):
    if not self.training:
        return _moshi_transformer_forward(self, x, *args, **kwargs)
    _check_transformer(self, x.shape[1], args, kwargs)
    return _compiled_transformer_layers(self, x)


def install() -> None:
    # moshi compiles _rms_norm, apply_rope and gating_forward_kernel with automatic dynamic
    # shapes: one graph per dtype, rank, grad mode, autocast state and shape class. A training
    # process that validates builds 8 of _rms_norm, dynamo's default limit, past which a
    # function's new variants run eagerly for the rest of the process.
    torch._dynamo.config.recompile_limit = 64
    moshi_transformer.F = _CausalSDPA()
    moshi_transformer.expand_repeated_kv = _keep_kv_heads
    LMModel.forward_depformer_training = forward_depformer_training
    ActivationGating.forward = _gating_forward
    StreamingTransformer.forward = _transformer_forward
