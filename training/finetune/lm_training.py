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
  profiled. Here it reads the written slots only, rounded up to STEP_SLOTS (`_lm_gen_step` narrows
  them), and the query heads that share a key/value head become that head's query positions, so
  the cache is read once and not once a query head. Seeded rollouts come out token for token the
  same, at 16 ms a step instead of 44 (the heads student at 320 rows; the 3B at 32 rows is
  identical too).
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
from moshi.models.lm import LMGen, LMModel, LMOutput
from moshi.models.lm_utils import _delay_sequence, _undelay_sequence
from moshi.modules import transformer as moshi_transformer
from moshi.modules.gating import ActivationGating
from moshi.modules.transformer import RMSNorm, StreamingMultiheadAttention, StreamingTransformer
from torch.nn.attention import SDPBackend, sdpa_kernel

from training.finetune.fp8_linear import fp8_linear


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


def _gating(module: ActivationGating, x: torch.Tensor, use_fp8: bool = False) -> torch.Tensor:
    lin = fp8_linear if use_fp8 else F.linear
    gate, value = lin(x, module.linear_in.weight).unflatten(-1, (2, -1)).unbind(-2)
    return lin(module.activation(gate) * value, module.linear_out.weight)


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


def _depformer_layers_ckpt_graph(depformer: Any, xs: torch.Tensor) -> torch.Tensor:
    for layer in depformer.layers:
        xs = torch.utils.checkpoint.checkpoint(_depformer_layer, layer, xs, use_reentrant=False)
    return xs


# Compiled on first call; every batch shape shares one dynamic graph.
_compiled_depformer_layers = torch.compile(_depformer_layers, dynamic=True)
_compiled_depformer_layers_ckpt = torch.compile(_depformer_layers_ckpt_graph, dynamic=True)


def _check_depformer(depformer: Any) -> None:
    if depformer._streaming_state is not None:
        raise RuntimeError("The training depformer runs without streaming")
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
    run_dep = (
        _compiled_depformer_layers_ckpt
        if getattr(self.depformer, "checkpointing", False)
        else _compiled_depformer_layers
    )
    out = run_dep(self.depformer, xs).to(xs.dtype).unbind(0)
    logits = [
        self.linears[cb](self.depformer_norms[cb](out[cb])).view(B, T, -1) for cb in range(steps)
    ]
    return torch.stack(logits, 1)


def _delayed(lm: LMModel, codes: torch.Tensor) -> torch.Tensor:
    """moshi's delayed sequence of `codes`: the initial token, then each row at its delay."""
    initial = lm._get_initial_token().expand(codes.shape[0], -1, -1)
    return torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)


def _main_transformer(
    lm: LMModel, codes: torch.Tensor, condition_tensors: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """`lm.forward_text` as moshi's forward calls it: each step's output and text logits, read
    from the steps before it."""
    sum_condition = cross_attention_src = None
    if condition_tensors is not None:
        sum_condition = lm.fuser.get_sum(condition_tensors)
        cross_attention_src = lm.fuser.get_cross(condition_tensors)
    elif lm.fuser is not None:
        raise ValueError("this model needs its condition tensors")
    return lm.forward_text(_delayed(lm, codes)[:, :, :-1], sum_condition, cross_attention_src)


def _text_output(
    lm: LMModel, text_logits: torch.Tensor, codes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    text_logits, text_mask = _undelay_sequence(lm.delays[:1], text_logits, fill_value=float("NaN"))
    return text_logits, text_mask & (codes[:, :1] != lm.zero_token_id)


def text_logits(
    lm: LMModel, codes: torch.Tensor, condition_tensors: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The text logits [B, 1, T, card] and their mask, as moshi's forward gives them, from the main
    transformer alone: no depformer."""
    _, logits = _main_transformer(lm, codes, condition_tensors)
    return _text_output(lm, logits, codes)


def forward_history(
    lm: LMModel, history: torch.Tensor, codes: torch.Tensor, condition_tensors: Any = None
) -> LMOutput:
    """moshi's `LMModel.forward` with two inputs, for scheduled sampling of the text: the main
    transformer reads `history` (the steps before each one, their text partly the model's own
    samples) and the depformer, or a heads student's text input, reads `codes` (each step's own
    tokens, the dataset's), which also give the masks; `history` holds -1 where `codes` do. So a
    step's audio is predicted after the text the dataset has at that frame, as its targets are."""
    transformer_out, text = _main_transformer(lm, history, condition_tensors)
    logits = lm.forward_depformer_training(_delayed(lm, codes)[:, :, 1:], transformer_out)
    audio = slice(lm.audio_offset, lm.audio_offset + lm.dep_q)
    logits, mask = _undelay_sequence(lm.delays[audio], logits, fill_value=float("NaN"))
    mask &= codes[:, audio] != lm.zero_token_id
    text_logits, text_mask = _text_output(lm, text, codes)
    return LMOutput(logits, mask, text_logits, text_mask)


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


def _transformer_layer(
    layer: Any, x: torch.Tensor, max_period: float, use_fp8: bool = False
) -> torch.Tensor:
    """One moshi StreamingTransformerLayer on x [B, T, C] at offset 0, without streaming."""
    attention = layer.self_attn
    channels = x.shape[-1]
    kv_heads = attention.num_heads // attention.kv_repeat
    lin = fp8_linear if use_fp8 else F.linear
    projected = lin(_rms_norm(layer.norm1, x), attention.in_projs[0].weight)
    q = projected[..., :channels].unflatten(-1, (attention.num_heads, -1)).transpose(1, 2)
    k, v = projected[..., channels:].unflatten(-1, (2, kv_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k = _rope(q, k, max_period)
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=attention.kv_repeat != 1
    )  # [B, H, T, D]
    update = lin(out.transpose(1, 2).flatten(2), attention.out_projs[0].weight)
    x = x.to(update) + layer.layer_scale_1(update)
    update = _gating(layer.gating, _rms_norm(layer.norm2, x), use_fp8=use_fp8)
    return x.to(update) + layer.layer_scale_2(update)


def _transformer_layers(transformer: Any, x: torch.Tensor) -> torch.Tensor:
    out = x
    max_period = transformer.rope.max_period
    for layer in transformer.layers:
        out = _transformer_layer(layer, out, max_period, False)
    return out.to(x.dtype)


def _transformer_layers_ckpt_bf16(transformer: Any, x: torch.Tensor) -> torch.Tensor:
    """Whole-loop compiled checkpointing for pure bf16 (+6.5% faster than per-layer compile via
    cross-layer norm/residual fusion)."""
    out = x
    max_period = transformer.rope.max_period
    ckpt_n = int(getattr(transformer, "checkpoint_layers", 0)) or len(transformer.layers)
    for index, layer in enumerate(transformer.layers):
        if index < ckpt_n:
            out = torch.utils.checkpoint.checkpoint(
                _transformer_layer, layer, out, max_period, False, use_reentrant=False
            )
        else:
            out = _transformer_layer(layer, out, max_period, False)
    return out.to(x.dtype)


# Whole-loop compile for bf16 (fuses norms/residuals across layers); per-layer compile for --fp8
# (avoids unrolling 28 FP8 quantize+scaled_mm blocks, ~20x faster compile).
_compiled_transformer_layers = torch.compile(_transformer_layers, dynamic=True)
_compiled_transformer_layers_ckpt_bf16 = torch.compile(_transformer_layers_ckpt_bf16, dynamic=True)
_compiled_transformer_layer = torch.compile(_transformer_layer, dynamic=True)


def _transformer_layers_ckpt(transformer: Any, x: torch.Tensor) -> torch.Tensor:
    """Checkpoint the first `checkpoint_layers` main layers (all layers when 0 or unset), leaving
    later layers uncheckpointed to use spare GPU memory for speed."""
    if not getattr(transformer, "use_fp8", False):
        return _compiled_transformer_layers_ckpt_bf16(transformer, x)
    out = x
    max_period = transformer.rope.max_period
    ckpt_n = int(getattr(transformer, "checkpoint_layers", 0)) or len(transformer.layers)
    for index, layer in enumerate(transformer.layers):
        if index < ckpt_n:
            out = torch.utils.checkpoint.checkpoint(
                _compiled_transformer_layer, layer, out, max_period, True, use_reentrant=False
            )
        else:
            out = _compiled_transformer_layer(layer, out, max_period, True)
    return out.to(x.dtype)


def _check_transformer(transformer: Any, steps: int, args: tuple, kwargs: dict) -> None:
    if transformer._streaming_state is not None:
        raise RuntimeError("The training transformer runs without streaming")
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
        if attention.in_projs[0].bias is not None or attention.out_projs[0].bias is not None:
            raise RuntimeError("Unsupported transformer attention bias")
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
    if self.checkpointing:
        return _transformer_layers_ckpt(self, x)
    if getattr(self, "use_fp8", False):
        # Per-layer compile avoids unrolling 14-28 FP8 quantize+scaled_mm blocks (~20x faster compile).
        out = x
        max_period = self.rope.max_period
        for layer in self.layers:
            out = _compiled_transformer_layer(layer, out, max_period, True)
        return out.to(x.dtype)
    return _compiled_transformer_layers(self, x)


# A generation step's attention reads its cache in whole STEP_SLOTS. The graph is captured again at
# each change, so a finer width costs captures and a coarser one reads unwritten slots: 320 rows of
# 1,208 steps take 19.7, 19.2, 19.4 and 20.5 s at 32, 64, 128 and 256, 29.9 s at the whole cache.
STEP_SLOTS = 64

_moshi_attention_forward = StreamingMultiheadAttention.forward
_moshi_lm_gen_step = LMGen._step


def _step_attention(
    self: StreamingMultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """moshi's attention for one streamed step, over the `step_slots` slots `_narrow` set on the
    streaming state (moshi's own without them). The caller's contract: only LMGen steps it, one
    frame a call and every row in step, so rows share one offset (no partial exec mask or reset),
    and the attention context is not changed while streaming. `_narrow` keeps the ring from
    wrapping, so slot j holds position j."""
    state = self._streaming_state
    slots = getattr(state, "step_slots", None)
    if slots is None:
        return _moshi_attention_forward(self, query, key, value)
    if query.shape[1] != 1:
        raise RuntimeError("A narrowed attention step takes one frame")
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
    mask = torch.arange(slots, device=position.device) <= position
    # Query heads kh * kv_repeat .. kh * kv_repeat + kv_repeat - 1 use key/value head kh: as its
    # kv_repeat query positions, they read its cache once.
    out = F.scaled_dot_product_attention(
        q.reshape(batch, groups, self.kv_repeat, dim),
        keys[:, :, :slots], values[:, :, :slots], mask[None, None, None])
    out = self.out_projs[0](out.reshape(batch, 1, channels))
    state.kv_cache.end_offset.add_(1)
    state.offset.add_(1)
    state.offset_cpu += 1
    return out


def _check_attention(attention: StreamingMultiheadAttention) -> None:
    if not attention.causal or attention.rope is None or attention.weights_per_step:
        raise RuntimeError("Unsupported attention for a narrowed generation step")


def _narrow(lm_gen: LMGen) -> None:
    """Set the slots the coming step's attention reads (`_step_attention`): the written ones and
    this step's, up to a whole STEP_SLOTS, or none (moshi's attention) once the cache's ring wraps
    or with `support_out_of_sync`, whose rows step apart. At a change the main transformer's CUDA
    graph is captured again."""
    state = lm_gen._streaming_state
    attentions = [layer.self_attn for layer in lm_gen.lm_model.transformer.layers]
    capacity = attentions[0]._streaming_state.kv_cache.capacity
    slots = None
    if state.offset_cpu < capacity and not lm_gen.support_out_of_sync:
        slots = min(STEP_SLOTS * (state.offset_cpu // STEP_SLOTS + 1), capacity)
    current = getattr(attentions[0]._streaming_state, "step_slots", None)
    if slots == current:
        return
    for attention in attentions:
        if slots is not None:
            _check_attention(attention)
        attention._streaming_state.step_slots = slots
    # A full reset after a wrap (current None) leaves moshi's graph until the next change: it
    # reads every slot, the same attention.
    if current is not None:
        state.graphed_main.reset()


def _lm_gen_step(
    self: LMGen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None
) -> Any:
    if self._streaming_state is not None:
        _narrow(self)
    return _moshi_lm_gen_step(self, input_tokens, depformer_replace_tokens)


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
