"""Faster moshi LM for training and generation: the same modules and weights, the same math.

`install()` (called when `common` is imported) replaces six things in moshi 0.2.13:

- Causal attention. moshi passes SDPA a boolean mask, which rules out the flash kernel; at
  offset 0 that mask is exactly lower-triangular, so it becomes `is_causal=True`.
- Grouped-query attention. The main transformer has 8 key/value heads for 16 query heads, and
  moshi copies each key and value to 16 heads before attending: in generation that is the whole
  3,000-frame KV cache, twice per layer and step (44% of a generation step at batch 16). SDPA
  groups the heads itself with `enable_gqa=True`.
- A generation step's attention (`_step_attention`). moshi attends over the whole KV cache at
  every step, whatever is written: 38 of the 46 ms of kernels a step at 320 rows of 1,208 slots,
  profiled. Here a Triton kernel (`training.tranche.decode_attention`) reads the written slots
  only, their count on the device, so the main transformer's CUDA graph replays at every step,
  and the query heads that share a key/value head read that head's cache once, together.
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

In eval mode (teacher-forced validation, generation) the main transformer is moshi's own but for
that attention, and generation (`LMGen`) also keeps moshi's depformer and gating. `install()` also
lets each compiled function keep 64 graphs instead of dynamo's 8, so none of moshi's runs eagerly.
"""
from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn.functional as F
from moshi.models.compression import MimiModel
from moshi.models.lm import LMGen, LMModel
from moshi.models.lm_utils import _delay_sequence
from moshi.modules import transformer as moshi_transformer
from moshi.modules.gating import ActivationGating
from moshi.modules.transformer import RMSNorm, StreamingMultiheadAttention, StreamingTransformer
from torch.nn.attention import SDPBackend, sdpa_kernel

from training.finetune.fp8_linear import (
    FrozenFP8Linear,
    _fp8_frozen_linear_op,
    fp8_frozen_linear,
    fp8_linear,
)

_ROPE_CACHE: dict[tuple[torch.device, int, float], tuple[torch.Tensor, torch.Tensor]] = {}
_ROPE_MAX_LEN = 4096


def _rope_tables(
    steps: int, dim: int, max_period: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached (cos, sin) tables for rope_concat at offset 0: computed once per device/dim/period
    instead of inside every transformer layer."""
    key = (device, dim, float(max_period))
    cached = _ROPE_CACHE.get(key)
    if cached is None or cached[0].shape[0] < steps:
        length = max(_ROPE_MAX_LEN, steps)
        ds = torch.arange(dim // 2, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2 / dim))
        ts = torch.arange(length, device=device, dtype=torch.float32).view(-1, 1)
        angles = freqs * ts
        cached = (torch.cos(angles), torch.sin(angles))
        _ROPE_CACHE[key] = cached
    return cached[0][:steps], cached[1][:steps]


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


def _linear_apply(module: Any, x: torch.Tensor, use_fp8: bool = False) -> torch.Tensor:
    if isinstance(module, FrozenFP8Linear):
        return module(x)
    if use_fp8:
        wq = getattr(module, "_fp8_w_cache", None)
        if wq is not None and not torch.is_grad_enabled():
            return fp8_frozen_linear(x, wq, module._fp8_s_cache)
        return fp8_linear(x, module.weight)
    return F.linear(x, module.weight, module.bias)


def _gating(module: ActivationGating, x: torch.Tensor, use_fp8: bool = False) -> torch.Tensor:
    gate, value = _linear_apply(module.linear_in, x, use_fp8).unflatten(-1, (2, -1)).unbind(-2)
    return _linear_apply(module.linear_out, module.activation(gate) * value, use_fp8)


_moshi_gating_forward = ActivationGating.forward


def _gating_forward(self: ActivationGating, x: torch.Tensor) -> torch.Tensor:
    if self.training and isinstance(self.linear_in, (torch.nn.Linear, FrozenFP8Linear)):
        return _gating(self, x)
    return _moshi_gating_forward(self, x)


def _schedule(owner: Any, steps: int) -> list[int]:
    schedule = owner.weights_per_step_schedule
    return list(schedule[:steps]) if schedule is not None else list(range(steps))


def _per_step(
    modules: Any, schedule: list[int], xs: torch.Tensor, function: Any, use_fp8: bool
) -> torch.Tensor:
    """xs [K, N, C]: step k through function(modules[schedule[k]], step, use_fp8).
    Contiguous runs of identical schedule indices (e.g. the 3B teacher's steps 8..15 sharing slice 8)
    are batched into one call, eliminating 7/8 of slice-8 GEMMs and weight HBM reads."""
    runs: list[tuple[int, int]] = []
    for idx in schedule:
        if runs and runs[-1][0] == idx:
            runs[-1] = (idx, runs[-1][1] + 1)
        else:
            runs.append((idx, 1))
    if len(runs) == len(schedule):
        return torch.stack([function(modules[index], x, use_fp8) for index, x in zip(schedule, xs.unbind(0))])
    n, c = xs.shape[1], xs.shape[2]
    chunks = xs.split([count for _, count in runs], dim=0)
    outs = []
    for (idx, count), chunk in zip(runs, chunks, strict=True):
        if count == 1:
            outs.append(function(modules[idx], chunk.squeeze(0), use_fp8).unsqueeze(0))
        else:
            outs.append(function(modules[idx], chunk.reshape(count * n, c), use_fp8).view(count, n, -1))
    return torch.cat(outs, dim=0)


def _depformer_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, steps: int) -> torch.Tensor:
    """Causal attention for short depformer sequences (K=9 student, K=16 teacher). With gradients,
    the explicit form, which the compiled layers fuse: forward and backward of the wide slices' K=9
    attention [12,288, 16, 9, 64] take 3.9 ms on the RTX 5090, the memory-efficient kernel 10.7.
    Without (the frozen teacher's forward), cuDNN on CUDA: 1.57x faster on Blackwell sm_120."""
    if torch.is_grad_enabled():
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones((steps, steps), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        return torch.matmul(torch.softmax(scores.float(), dim=-1).to(q.dtype), v)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION if q.is_cuda else SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _depformer_layer(layer: Any, xs: torch.Tensor, use_fp8: bool) -> torch.Tensor:
    """One moshi StreamingTransformerLayer on step-major xs [K, N, C], without streaming."""
    steps, n, channels = xs.shape
    attention = layer.self_attn
    schedule = _schedule(attention, steps)
    projected = _per_step(attention.in_projs, schedule, layer.norm1(xs), _linear_apply, use_fp8)  # [K, N, 3C]
    q, k, v = projected.view(steps, n, 3, attention.num_heads, -1).permute(2, 1, 3, 0, 4)
    out = _depformer_sdpa(q, k, v, steps)  # [N, H, K, D]
    out = out.permute(2, 0, 1, 3).reshape(steps, n, channels)
    update = _per_step(attention.out_projs, schedule, out, _linear_apply, use_fp8)
    xs = xs.to(update) + layer.layer_scale_1(update)
    update = _per_step(layer.gating, _schedule(layer, steps), layer.norm2(xs), _gating, use_fp8)
    return xs.to(update) + layer.layer_scale_2(update)


def _depformer_layers(depformer: Any, xs: torch.Tensor, use_fp8: bool) -> torch.Tensor:
    for layer in depformer.layers:
        xs = _depformer_layer(layer, xs, use_fp8)
    return xs


def _depformer_layers_ckpt_graph(depformer: Any, xs: torch.Tensor, use_fp8: bool) -> torch.Tensor:
    for layer in depformer.layers:
        xs = torch.utils.checkpoint.checkpoint(_depformer_layer, layer, xs, use_fp8, use_reentrant=False)
    return xs


# Compiled on first call; every batch shape shares one dynamic graph.
_compiled_depformer_layers = torch.compile(_depformer_layers, dynamic=True)
_compiled_depformer_layers_ckpt = torch.compile(_depformer_layers_ckpt_graph, dynamic=True)


def depformer_fp8(depformer: Any) -> bool:
    """Whether a depformer's training forward runs its linears in FP8 (`train.py --fp8` sets
    `use_fp8`): in training only, as the main transformer's, so validation stays bf16."""
    return bool(getattr(depformer, "use_fp8", False)) and depformer.training


def check_depformer(depformer: Any) -> None:
    if depformer._streaming_state is not None:
        raise RuntimeError("The training depformer runs without streaming")
    if depformer.positional_embedding in {"sin", "sin_rope"}:
        raise RuntimeError("The training depformer has no positional embedding")
    use_fp8 = depformer_fp8(depformer)
    for layer in depformer.layers:
        attention = layer.self_attn
        if attention.kv_repeat != 1 or attention.rope is not None or attention.cross_attention:
            raise RuntimeError("Unsupported depformer attention")
        if layer.cross_attention is not None or layer.skip_self_attn or layer.gating is None:
            raise RuntimeError("Unsupported depformer layer")
        if use_fp8:
            gatings = [linear for gating in layer.gating for linear in (gating.linear_in, gating.linear_out)]
            if any(m.bias is not None for m in [*attention.in_projs, *attention.out_projs, *gatings]):
                raise RuntimeError("FP8 depformer linears have no bias")


def run_depformer(depformer: Any, xs: torch.Tensor) -> torch.Tensor:
    """The depformer's layers on step-major xs [K, N, C], compiled (checkpointed if the depformer
    is; its linears in FP8 per `depformer_fp8`), as unbound steps."""
    check_depformer(depformer)
    run = _compiled_depformer_layers_ckpt if getattr(depformer, "checkpointing", False) else _compiled_depformer_layers
    return run(depformer, xs, depformer_fp8(depformer)).to(xs.dtype).unbind(0)


def project_depformer_in(
    depformer_in: Any, transformer_out: torch.Tensor, use_fp8: bool = False
) -> tuple[torch.Tensor, ...]:
    """Run all `depformer_in` projections on `transformer_out` [B, T, C] in a single fused GEMM
    (FP8 with `use_fp8`), reading `transformer_out` once instead of S times. Returns a tuple of S
    tensors [B, T, D]."""
    B, T, _ = transformer_out.shape
    count = len(depformer_in)
    if count == 1:
        return (depformer_in[0](transformer_out),)
    if all(isinstance(m, torch.nn.Linear) and m.bias is None for m in depformer_in):
        w = torch.cat([m.weight for m in depformer_in], dim=0)
        out = fp8_linear(transformer_out, w) if use_fp8 else F.linear(transformer_out, w)
        return out.view(B, T, count, -1).unbind(2)
    if all(isinstance(m, FrozenFP8Linear) for m in depformer_in):
        if transformer_out.is_cuda and ((B * T) % 16 == 0):
            wq = torch.cat([m.weight_fp8 for m in depformer_in], dim=0)
            sw = torch.cat([m.scale_fp8 for m in depformer_in], dim=0)
            if (wq.shape[0] % 16 == 0) and (wq.shape[1] % 16 == 0):
                return _fp8_frozen_linear_op(transformer_out, wq, sw).view(B, T, count, -1).unbind(2)
    return tuple(m(transformer_out) for m in depformer_in)


def _apply_norms(norms: Any, out: tuple[torch.Tensor, ...], B: int, T: int) -> tuple[torch.Tensor, ...]:
    return tuple(norm(x).view(B, T, -1) for norm, x in zip(norms, out, strict=True))


_compiled_apply_norms = torch.compile(_apply_norms, dynamic=True)


def forward_depformer_hidden(
    lm: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Normed depformer outputs `[S]` of `[B, T, dep_dim]` before `lm.linears`, for chunked
    linear+KL distillation. Dispatches to a slices model's custom `forward_depformer_hidden`."""
    custom = getattr(lm, "forward_depformer_hidden", None)
    if custom is not None:
        return custom(sequence, transformer_out)
    B, K, T = sequence.shape
    if K != lm.num_codebooks:
        raise RuntimeError(f"Depformer training needs all {lm.num_codebooks} codebooks, got {K}")
    steps = lm.dep_q
    projected = project_depformer_in(lm.depformer_in, transformer_out, depformer_fp8(lm.depformer))
    inputs = []
    for cb in range(steps):
        index = 0
        if lm.depformer_multi_linear:
            index = cb
            if lm.depformer_weights_per_step_schedule is not None:
                index = lm.depformer_weights_per_step_schedule[cb]
        if cb == 0:
            token_in = lm.depformer_text_emb(sequence[:, 0])
        else:
            token_in = lm.depformer_emb[cb - 1](sequence[:, cb + lm.audio_offset - 1])
        inputs.append(token_in + projected[index])
    out = run_depformer(lm.depformer, torch.stack(inputs, 0).view(steps, B * T, -1))
    return _compiled_apply_norms(lm.depformer_norms[:steps], out, B, T)


def forward_depformer_training(
    self: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor
) -> torch.Tensor:
    """moshi's LMModel.forward_depformer_training: logits [B, dep_q, T, card]."""
    normed = forward_depformer_hidden(self, sequence, transformer_out)
    logits = [self.linears[cb](normed[cb]) for cb in range(self.dep_q)]
    return torch.stack(logits, 1)


def _delayed(lm: LMModel, codes: torch.Tensor) -> torch.Tensor:
    """moshi's delayed sequence of `codes`: the initial token, then each row at its delay."""
    initial = lm._get_initial_token().expand(codes.shape[0], -1, -1)
    return torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)


def _embed_inputs(lm: LMModel, sequence: torch.Tensor) -> torch.Tensor:
    """Sum text + all audio embeddings for `sequence` [B, K, T] in one compiled graph."""
    input_ = lm.text_emb(sequence[:, 0])
    for cb_index in range(lm.num_audio_codebooks):
        input_ = input_ + lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
    return input_


_compiled_embed_inputs = torch.compile(_embed_inputs, dynamic=True)


def forward_trunk_hidden(
    lm: LMModel, codes: torch.Tensor, condition_tensors: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Main transformer output `[B, T, dim]` (after `out_norm`, skipping `text_linear`) and
    `delayed` `[B, K, T + 1]`. Used by chunked distillation and `--freeze-trunk`."""
    delayed = _delayed(lm, codes)
    sequence = delayed[:, :, :-1]
    sum_condition = cross_attention_src = None
    if condition_tensors is not None:
        sum_condition = lm.fuser.get_sum(condition_tensors)
        cross_attention_src = lm.fuser.get_cross(condition_tensors)
    elif lm.fuser is not None:
        raise ValueError("this model needs its condition tensors")
    input_ = _compiled_embed_inputs(lm, sequence)
    if sum_condition is not None:
        input_ = input_ + sum_condition.to(input_)
    if cross_attention_src is not None:
        cross_attention_src = cross_attention_src.to(input_)
    transformer_out = lm.transformer(input_, cross_attention_src=cross_attention_src)
    if lm.out_norm:
        transformer_out = lm.out_norm(transformer_out)
    return transformer_out, delayed


def _rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """moshi's `_rms_norm`, traced inline rather than through its own compiled function."""
    y = x if norm.dtype is None else x.to(norm.dtype)
    variance = norm.eps + torch.mean(y**2, dim=2, keepdim=True)
    return (y * (norm.alpha.to(variance) * torch.rsqrt(variance))).to(x.dtype)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed (cos, sin) rope_concat tables to [B, H, T, D] q and k."""
    half = q.shape[-1] // 2

    def rotate(x: torch.Tensor) -> torch.Tensor:
        xr, xi = x[..., :half].float(), x[..., half:].float()
        real, imaginary = xr * rotr - xi * roti, xr * roti + xi * rotr
        return torch.cat([real.to(q.dtype), imaginary.to(q.dtype)], -1)

    return rotate(q), rotate(k)


def _transformer_layer(
    layer: Any,
    x: torch.Tensor,
    rotr: torch.Tensor,
    roti: torch.Tensor,
    use_fp8: bool = False,
) -> torch.Tensor:
    """One moshi StreamingTransformerLayer on x [B, T, C] at offset 0, without streaming."""
    attention = layer.self_attn
    channels = x.shape[-1]
    kv_heads = attention.num_heads // attention.kv_repeat
    projected = _linear_apply(attention.in_projs[0], _rms_norm(layer.norm1, x), use_fp8)
    q = projected[..., :channels].unflatten(-1, (attention.num_heads, -1)).transpose(1, 2)
    k, v = projected[..., channels:].unflatten(-1, (2, kv_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k = _apply_rope(q, k, rotr, roti)
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=attention.kv_repeat != 1
    )  # [B, H, T, D]
    update = _linear_apply(attention.out_projs[0], out.transpose(1, 2).flatten(2), use_fp8)
    x = x.to(update) + layer.layer_scale_1(update)
    update = _gating(layer.gating, _rms_norm(layer.norm2, x), use_fp8=use_fp8)
    return x.to(update) + layer.layer_scale_2(update)


def _transformer_layers(
    transformer: Any, x: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor, use_fp8: bool
) -> torch.Tensor:
    out = x
    for layer in transformer.layers:
        out = _transformer_layer(layer, out, rotr, roti, use_fp8)
    return out.to(x.dtype)


def _transformer_layers_ckpt_loop(
    transformer: Any, x: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor, use_fp8: bool
) -> torch.Tensor:
    """Whole-loop compiled selective checkpointing (fuses norms/residuals across layers)."""
    out = x
    ckpt_n = int(getattr(transformer, "checkpoint_layers", 0)) or len(transformer.layers)
    for index, layer in enumerate(transformer.layers):
        if index < ckpt_n:
            out = torch.utils.checkpoint.checkpoint(
                _transformer_layer, layer, out, rotr, roti, use_fp8, use_reentrant=False
            )
        else:
            out = _transformer_layer(layer, out, rotr, roti, use_fp8)
    return out.to(x.dtype)


# Whole-loop compiled functions: since fp8_linear is a single opaque custom_op, both bf16 and
# FP8 compile cleanly in one graph and fuse RMSNorm + residuals + SiLU across layers.
_compiled_transformer_layers = torch.compile(_transformer_layers, dynamic=True)
_compiled_transformer_layers_ckpt = torch.compile(_transformer_layers_ckpt_loop, dynamic=True)


def _check_transformer(transformer: Any, steps: int, args: tuple, kwargs: dict) -> None:
    if transformer._streaming_state is not None:
        raise RuntimeError("The training transformer runs without streaming")
    if args or kwargs.get("cross_attention_src") is not None:
        raise RuntimeError("The training transformer has no cross-attention")
    if transformer.positional_embedding != "rope_concat":
        raise RuntimeError("The training transformer needs rope_concat positions")
    linear_types = (torch.nn.Linear, FrozenFP8Linear)
    for layer in transformer.layers:
        if layer.cross_attention is not None or layer.skip_self_attn or layer.weights_per_step:
            raise RuntimeError("Unsupported transformer layer")
        attention = layer.self_attn
        if not attention.causal or attention.cross_attention or len(attention.in_projs) != 1:
            raise RuntimeError("Unsupported transformer attention")
        if not isinstance(attention.in_projs[0], linear_types) or not isinstance(attention.out_projs[0], linear_types):
            raise RuntimeError("Unsupported transformer attention linear")
        if attention.in_projs[0].bias is not None or attention.out_projs[0].bias is not None:
            raise RuntimeError("Unsupported transformer attention bias")
        if attention.context is not None and steps > attention.context:
            raise RuntimeError(f"{steps} frames exceed the attention context {attention.context}")
        gating = layer.gating
        if not isinstance(gating, ActivationGating) or not (
            isinstance(gating.linear_in, linear_types)
            and isinstance(gating.linear_out, linear_types)
        ):
            raise RuntimeError("Unsupported transformer feed-forward")
        if not (isinstance(layer.norm1, RMSNorm) and isinstance(layer.norm2, RMSNorm)):
            raise RuntimeError("Unsupported transformer norms")


_moshi_transformer_forward = StreamingTransformer.forward


def _transformer_forward(self: StreamingTransformer, x: torch.Tensor, *args, **kwargs):
    if not self.training:
        return _moshi_transformer_forward(self, x, *args, **kwargs)
    _check_transformer(self, x.shape[1], args, kwargs)
    head_dim = x.shape[-1] // self.layers[0].self_attn.num_heads
    rotr, roti = _rope_tables(x.shape[1], head_dim, self.rope.max_period, x.device)
    use_fp8 = bool(getattr(self, "use_fp8", False))
    if self.checkpointing:
        return _compiled_transformer_layers_ckpt(self, x, rotr, roti, use_fp8)
    return _compiled_transformer_layers(self, x, rotr, roti, use_fp8)


_moshi_attention_forward = StreamingMultiheadAttention.forward
_moshi_lm_gen_step = LMGen._step


def _step_attention(
    self: StreamingMultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """moshi's attention for one streamed step, over the cache's written slots only when `_narrow`
    set `live_keys` on the streaming state (moshi's own without it). The caller's contract: only
    LMGen steps it, one frame a call and every row in step, so rows share one offset (no partial
    exec mask or reset), and the attention context is not changed while streaming. `_narrow` keeps
    the ring from wrapping, so slot j holds position j, and slots 0..offset are the live ones. The
    Triton kernel (`decode_attention.attend`) reads that length on the device, so one CUDA graph of
    the main transformer serves every step, and the query heads that share a key/value head read
    its cache once, together."""
    state = self._streaming_state
    if not getattr(state, "live_keys", False):
        return _moshi_attention_forward(self, query, key, value)
    if query.shape[1] != 1:
        raise RuntimeError("A narrowed attention step takes one frame")
    from training.tranche.decode_attention import attend

    batch, _, channels = query.shape
    groups, dim = self.num_heads // self.kv_repeat, channels // self.num_heads
    projected = self.in_projs[0](query)
    q = projected[..., :channels].view(batch, 1, self.num_heads, dim).transpose(1, 2)
    k, v = projected[..., channels:].view(batch, 1, 2, groups, dim).permute(2, 0, 3, 1, 4)
    q, k = self.rope(q, k, state.offset, time_before_heads=False)
    keys, values = state.kv_cache.cache
    position = state.offset[:1]
    keys.index_copy_(2, position, k)
    values.index_copy_(2, position, v)
    out = attend(q.reshape(batch, self.num_heads, dim), keys, values, state.offset + 1, dim**-0.5)
    out = self.out_projs[0](out.reshape(batch, 1, channels))
    state.kv_cache.end_offset.add_(1)
    state.offset.add_(1)
    state.offset_cpu += 1
    return out


def _check_attention(attention: StreamingMultiheadAttention) -> None:
    if not attention.causal or attention.rope is None or attention.weights_per_step:
        raise RuntimeError("Unsupported attention for a narrowed generation step")


def _narrow(lm_gen: LMGen) -> None:
    """Whether the coming step's attention reads only the written slots (`_step_attention`): on
    CUDA until the cache's ring wraps, and never with `support_out_of_sync`, whose rows step apart.
    At a change the main transformer's CUDA graph is captured again."""
    state = lm_gen._streaming_state
    attentions = [layer.self_attn for layer in lm_gen.lm_model.transformer.layers]
    cache = attentions[0]._streaming_state.kv_cache
    live = cache.cache.is_cuda and state.offset_cpu < cache.capacity and not lm_gen.support_out_of_sync
    current = getattr(attentions[0]._streaming_state, "live_keys", None)
    if live == current:
        return
    for attention in attentions:
        if live:
            _check_attention(attention)
        attention._streaming_state.live_keys = live
    if current is not None:  # None: a new streaming state, its graph not captured yet
        state.graphed_main.reset()


def _lm_gen_step(
    self: LMGen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None
) -> Any:
    if self._streaming_state is not None:
        _narrow(self)
    return _moshi_lm_gen_step(self, input_tokens, depformer_replace_tokens)


_moshi_mimi_encode = MimiModel.encode
_moshi_mimi_apply = MimiModel._apply


def _fast_mimi_encode(self: MimiModel, x: torch.Tensor) -> torch.Tensor:
    """Accelerated stateless CUDA Mimi encode (`training.cache.mimi_cuda.FastMimiEncoder`) for
    non-streaming `MimiModel.encode` in fine-tuning, room encoding, validation, and eval; moshi's
    stock encode on CPU, MPS or in streaming mode."""
    if self._streaming_state is None and x.is_cuda:
        fast = getattr(self, "_fast_encoder", None)
        if fast is None or fast._device != x.device:
            from training.cache.mimi_cuda import FastMimiEncoder

            fast = FastMimiEncoder.from_moshi(self, device=x.device)
            object.__setattr__(self, "_fast_encoder", fast)
        return fast.encode(x, num_codebooks=self.num_codebooks)
    return _moshi_mimi_encode(self, x)


def _fast_mimi_apply(self: MimiModel, fn: Any) -> MimiModel:
    # Drop cached fast encoder on `.to("cpu")` / `.to(device)` so parking Mimi on CPU during
    # training frees all GPU weights immediately (~85 MB saved on 32 GB RTX 5090).
    if hasattr(self, "_fast_encoder"):
        object.__setattr__(self, "_fast_encoder", None)
    return _moshi_mimi_apply(self, fn)


def install() -> None:
    # moshi compiles _rms_norm, apply_rope and gating_forward_kernel with automatic dynamic
    # shapes: one graph per dtype, rank, grad mode, autocast state and shape class. A training
    # process that validates builds 8 of _rms_norm, dynamo's default limit, past which a
    # function's new variants run eagerly for the rest of the process.
    torch._dynamo.config.recompile_limit = 64
    moshi_transformer.F = _CausalSDPA()
    moshi_transformer.expand_repeated_kv = _keep_kv_heads
    StreamingMultiheadAttention.forward = _step_attention
    LMGen._step = _lm_gen_step
    LMModel.forward_depformer_training = forward_depformer_training
    ActivationGating.forward = _gating_forward
    StreamingTransformer.forward = _transformer_forward
    MimiModel.encode = _fast_mimi_encode
    MimiModel._apply = _fast_mimi_apply
