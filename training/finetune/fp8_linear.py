"""Rowwise FP8 linear for Blackwell (sm_120) via fused Triton quantization and `torch._scaled_mm`.

On RTX 5090 / RTX PRO 6000 (GB202, sm_120), rowwise FP8 GEMMs run at ~650-735 TFLOP/s vs
~380-444 TFLOP/s in bf16 (~1.65x faster).

Two paths:
1. Trainable `fp8_linear(x, weight)` (`--fp8`): single-pass Triton rowwise E4M3 quantization on
   forward and tiled transpose+quantization on backward (no fp32 HBM intermediates and no
   `.T.contiguous()` HBM passes), registered as a `torch.library.custom_op` so `torch.compile`
   traces cleanly across layers with zero graph breaks.
2. Pre-quantized frozen teacher `FrozenFP8Linear` (`--fp8-teacher trunk|all`): quantizes each
   frozen teacher linear weight once at startup into `(weight_fp8, scale_fp8)` and drops the bf16
   weight copy, eliminating all runtime weight quantization and halving teacher weight VRAM.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0


if _HAS_TRITON:

    @triton.jit
    def _quantize_rowwise_1pass_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        K,
        stride_xm,
        BLOCK_K: tl.constexpr,
    ):
        """Single-pass rowwise E4M3 quantization for K <= 8192: reads bf16 once into registers."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(amax, 1e-12) / 448.0
        q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + row * K + offs, q, mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _quantize_rowwise_2pass_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        K,
        stride_xm,
        BLOCK_K: tl.constexpr,
    ):
        """Two-pass rowwise E4M3 quantization for wide rows (K > 8192) without fp32 HBM buffers."""
        row = tl.program_id(0)
        row_x = x_ptr + row * stride_xm
        amax = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
        scale = tl.maximum(amax, 1e-12) / 448.0
        inv_scale = 1.0 / scale
        tl.store(s_ptr + row, scale)
        row_q = q_ptr + row * K
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
            q = tl.clamp(x * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
            tl.store(row_q + offs, q, mask=mask)

    @triton.jit
    def _column_amax_kernel(
        x_ptr,
        amax_ptr,
        M,
        K,
        stride_xm,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """amax [K] (zeroed) of x [M, K]'s columns: each program one [BLOCK_M, BLOCK_K] tile, so a
        tall x fills every SM, the tiles' maxima joined by atomic max (exact in any order)."""
        m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        tile = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :],
            mask=(m_offs[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.atomic_max(amax_ptr + k_offs, tl.max(tl.abs(tile), axis=0), mask=k_mask)

    @triton.jit
    def _transpose_quantize_kernel(
        x_ptr,
        amax_ptr,
        qt_ptr,
        s_ptr,
        M,
        K,
        stride_xm,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One [BLOCK_M, BLOCK_K] tile of x [M, K] quantized by its columns' amax and stored
        transposed into qt [K, M], coalesced along M; the first row of tiles stores s [K]."""
        pid_m = tl.program_id(0)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        scale = tl.maximum(tl.load(amax_ptr + k_offs, mask=k_mask, other=0.0), 1e-12) / 448.0
        inv_scale = 1.0 / scale
        if pid_m == 0:
            tl.store(s_ptr + k_offs, scale, mask=k_mask)
        tile = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :],
            mask=(m_offs[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_t = tl.trans(tl.clamp(tile * inv_scale[None, :], -448.0, 448.0).to(tl.float8e4nv))
        tl.store(
            qt_ptr + k_offs[:, None] * M + m_offs[None, :],
            q_t,
            mask=k_mask[:, None] & (m_offs[None, :] < M),
        )


def _quantize_rowwise_eager(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch rowwise E4M3 quantization of 2D [M, K] -> (q [M, K], s [M, 1])."""
    x_f32 = x.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_MAX
    q = (x_f32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return q, scale


def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast Triton rowwise E4M3 quantization of 2D [M, K] -> (q [M, K], s [M, 1])."""
    if not (x.is_cuda and _HAS_TRITON):
        return _quantize_rowwise_eager(x)
    x_c = x if x.stride(-1) == 1 else x.contiguous()
    M, K = x_c.shape
    q = torch.empty((M, K), device=x_c.device, dtype=_FP8_DTYPE)
    s = torch.empty((M, 1), device=x_c.device, dtype=torch.float32)
    if K <= 8192:
        block_k = triton.next_power_of_2(K)
        num_warps = 8 if block_k >= 2048 else 4
        _quantize_rowwise_1pass_kernel[(M,)](
            x_c, q, s, K, x_c.stride(0), BLOCK_K=block_k, num_warps=num_warps
        )
    else:
        _quantize_rowwise_2pass_kernel[(M,)](
            x_c, q, s, K, x_c.stride(0), BLOCK_K=4096, num_warps=8
        )
    return q, s


def transpose_quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiled transpose + rowwise E4M3 quantization of 2D [M, K] -> (qt [K, M], s [K, 1]): the
    columns' amax over tiles of [128, 64] (128-byte bf16 loads, M/128 x K/64 programs), then each
    tile quantized and stored transposed, reading x from L2 the second time."""
    if not (x.is_cuda and _HAS_TRITON):
        return _quantize_rowwise_eager(x.T.contiguous())
    x_c = x if x.stride(-1) == 1 else x.contiguous()
    M, K = x_c.shape
    qt = torch.empty((K, M), device=x_c.device, dtype=_FP8_DTYPE)
    s = torch.empty((K, 1), device=x_c.device, dtype=torch.float32)
    amax = torch.zeros((K,), device=x_c.device, dtype=torch.float32)
    block_m, block_k = 128, 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    _column_amax_kernel[grid](x_c, amax, M, K, x_c.stride(0), BLOCK_M=block_m, BLOCK_K=block_k, num_warps=4)
    _transpose_quantize_kernel[grid](
        x_c, amax, qt, s, M, K, x_c.stride(0), BLOCK_M=block_m, BLOCK_K=block_k, num_warps=4
    )
    return qt, s


def _scaled_mm_rowwise(
    aq: torch.Tensor, sa: torch.Tensor, bq_t: torch.Tensor, sb_t: torch.Tensor
) -> torch.Tensor:
    return torch._scaled_mm(aq, bq_t, scale_a=sa, scale_b=sb_t, out_dtype=torch.bfloat16)


@torch.library.custom_op("hibiki::fp8_linear_fwd", mutates_args=())
def _fp8_linear_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    xq, sx = quantize_rowwise(x_2d)
    wq, sw = quantize_rowwise(weight)
    out = _scaled_mm_rowwise(xq, sx, wq.T, sw.T)
    out = out.reshape(*orig_shape[:-1], weight.shape[0])
    return out if out.dtype == x.dtype else out.to(x.dtype)


@_fp8_linear_fwd.register_fake
def _fp8_linear_fwd_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op("hibiki::fp8_linear_bwd", mutates_args=())
def _fp8_linear_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    need_x: bool,
    need_w: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    dy_2d = grad_out.reshape(-1, weight.shape[0])

    if need_x:
        dyq, sdy = quantize_rowwise(dy_2d)
        wtq, swt = transpose_quantize_rowwise(weight)  # [K, N]
        gx = _scaled_mm_rowwise(dyq, sdy, wtq.T, swt.T).reshape(orig_shape)
        if gx.dtype != x.dtype:
            gx = gx.to(x.dtype)
    else:
        gx = x.new_empty((0,))

    if need_w:
        dytq, sdyt = transpose_quantize_rowwise(dy_2d)  # [N, M]
        xtq, sxt = transpose_quantize_rowwise(x_2d)  # [K, M]
        gw = _scaled_mm_rowwise(dytq, sdyt, xtq.T, sxt.T)
        if gw.dtype != weight.dtype:
            gw = gw.to(weight.dtype)
    else:
        gw = weight.new_empty((0,))

    return gx, gw


@_fp8_linear_bwd.register_fake
def _fp8_linear_bwd_fake(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    need_x: bool,
    need_w: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    gx = x.new_empty(x.shape) if need_x else x.new_empty((0,))
    gw = weight.new_empty(weight.shape) if need_w else weight.new_empty((0,))
    return gx, gw


def _fp8_linear_setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _fp8_linear_backward(
    ctx: Any, grad_out: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    x, weight = ctx.saved_tensors
    need_x, need_w = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
    gx, gw = _fp8_linear_bwd(grad_out, x, weight, need_x, need_w)
    return (gx if need_x else None), (gw if need_w else None)


_fp8_linear_fwd.register_autograd(_fp8_linear_backward, setup_context=_fp8_linear_setup_context)


def fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `F.linear(x, weight)` (no bias) using fused Triton rowwise FP8 on
    CUDA when shapes are 16-aligned; falls back to standard `F.linear` on CPU or unaligned shapes."""
    if x.is_cuda and weight.ndim == 2:
        m = x.numel() // x.shape[-1]
        n, k = weight.shape
        if (m % 16 == 0) and (n % 16 == 0) and (k % 16 == 0):
            return _fp8_linear_fwd(x, weight)
    return torch.nn.functional.linear(x, weight)


@torch.library.custom_op("hibiki::fp8_frozen_linear", mutates_args=())
def _fp8_frozen_linear_op(
    x: torch.Tensor, weight_fp8: torch.Tensor, scale_fp8: torch.Tensor
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    xq, sx = quantize_rowwise(x_2d)
    out = _scaled_mm_rowwise(xq, sx, weight_fp8.T, scale_fp8.T)
    out = out.reshape(*orig_shape[:-1], weight_fp8.shape[0])
    return out if out.dtype == x.dtype else out.to(x.dtype)


@_fp8_frozen_linear_op.register_fake
def _fp8_frozen_linear_op_fake(
    x: torch.Tensor, weight_fp8: torch.Tensor, scale_fp8: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_fp8.shape[0]))


def fp8_frozen_linear(
    x: torch.Tensor, weight_fp8: torch.Tensor, scale_fp8: torch.Tensor
) -> torch.Tensor:
    """Apply pre-quantized `(weight_fp8, scale_fp8)` to `x` using rowwise FP8 on CUDA."""
    if x.is_cuda:
        m = x.numel() // x.shape[-1]
        if (m % 16 == 0) and (weight_fp8.shape[0] % 16 == 0) and (weight_fp8.shape[1] % 16 == 0):
            return _fp8_frozen_linear_op(x, weight_fp8, scale_fp8)
    w_bf16 = (weight_fp8.float() * scale_fp8).to(x.dtype)
    return torch.nn.functional.linear(x, w_bf16)


class FrozenFP8Linear(nn.Module):
    """Pre-quantized rowwise E4M3 linear for a frozen distillation teacher.
    Quantizes the weight once at startup, freeing the bf16 weight copy and avoiding all runtime
    weight quantization."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"Expected 2D weight, got {tuple(weight.shape)}")
        wq, sw = quantize_rowwise(weight.detach())
        self.register_buffer("weight_fp8", wq)
        self.register_buffer("scale_fp8", sw)
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])

    @property
    def bias(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_frozen_linear(x, self.weight_fp8, self.scale_fp8)


def cache_frozen_fp8_weights(model: Any) -> list[torch.Tensor]:
    """Attach plain (non-buffer) rowwise FP8 weight caches to frozen trunk `nn.Linear` modules.
    Leaves `state_dict()` and the original bf16 weights untouched for checkpointing and eval,
    while eliminating per-step weight quantization during `--freeze-trunk --fp8` training."""
    cached: list[torch.Tensor] = []

    def _attach(mod: Any) -> None:
        if (
            isinstance(mod, nn.Linear)
            and mod.bias is None
            and not mod.weight.requires_grad
            and (mod.weight.shape[0] % 16 == 0)
            and (mod.weight.shape[1] % 16 == 0)
        ):
            wq, sw = quantize_rowwise(mod.weight.detach())
            mod._fp8_w_cache = wq
            mod._fp8_s_cache = sw
            cached.append(wq)
            cached.append(sw)

    for layer in model.transformer.layers:
        for mod in layer.self_attn.in_projs:
            _attach(mod)
        for mod in layer.self_attn.out_projs:
            _attach(mod)
        if layer.gating is not None:
            _attach(layer.gating.linear_in)
            _attach(layer.gating.linear_out)
    return cached


def _replace_linear(parent: Any, attr: str) -> int:
    mod = getattr(parent, attr)
    if isinstance(mod, nn.Linear) and mod.bias is None:
        n, k = mod.weight.shape
        if (n % 16 == 0) and (k % 16 == 0):
            setattr(parent, attr, FrozenFP8Linear(mod.weight))
            return 1
    return 0


def _replace_linear_list(module_list: nn.ModuleList) -> int:
    count = 0
    for idx, mod in enumerate(module_list):
        if isinstance(mod, nn.Linear) and mod.bias is None:
            n, k = mod.weight.shape
            if (n % 16 == 0) and (k % 16 == 0):
                module_list[idx] = FrozenFP8Linear(mod.weight)
                count += 1
    return count


def prepare_fp8_teacher(teacher: Any, mode: str) -> int:
    """Pre-quantize frozen teacher linear weights to rowwise E4M3 in place.
    `mode`:
      - 'none': no-op.
      - 'trunk': main transformer's 4 linears per layer (in_proj, out_proj, gating linear_in/out).
      - 'all': main transformer + depformer's per-step linears and gating + depformer_in.
    Keeps `text_linear` and audio `linears` in bf16 for clean target distributions.
    Returns number of quantized linears."""
    if mode == "none":
        return 0
    if mode not in ("trunk", "all"):
        raise ValueError(f"Unsupported fp8 teacher mode: {mode!r}")
    count = 0
    for layer in teacher.transformer.layers:
        count += _replace_linear_list(layer.self_attn.in_projs)
        count += _replace_linear_list(layer.self_attn.out_projs)
        if layer.gating is not None:
            count += _replace_linear(layer.gating, "linear_in")
            count += _replace_linear(layer.gating, "linear_out")
    if mode == "all":
        if getattr(teacher, "depformer_in", None) is not None:
            count += _replace_linear_list(teacher.depformer_in)
        if getattr(teacher, "depformer", None) is not None and hasattr(teacher.depformer, "layers"):
            for layer in teacher.depformer.layers:
                count += _replace_linear_list(layer.self_attn.in_projs)
                count += _replace_linear_list(layer.self_attn.out_projs)
                for gating in layer.gating:
                    count += _replace_linear(gating, "linear_in")
                    count += _replace_linear(gating, "linear_out")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return count
