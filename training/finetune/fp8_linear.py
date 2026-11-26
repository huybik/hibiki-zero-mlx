"""Rowwise FP8 linear for Blackwell (sm_120) main-transformer matmuls via `torch._scaled_mm`.

On RTX 5090 / RTX PRO 6000 (GB202, sm_120), rowwise FP8 GEMMs run at ~650-735 TFLOP/s vs
~380-444 TFLOP/s in bf16 (~1.65x faster). This module applies dynamic per-row E4M3 quantization
to the main transformer's large 2D linears (in_proj, out_proj, gating linear_in/linear_out)
while keeping norms, RoPE, attention, embeddings, heads, and depformer in bf16.

All dimensions of Hibiki-Zero's main transformer (2560, 5120, 7168, 14336) and stream frame
buckets (multiples of 16) satisfy `torch._scaled_mm`'s 16-element alignment requirement.
"""
from __future__ import annotations

import torch

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0


def _quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D tensor [M, K] per row to E4M3 with fp32 row scales [M, 1]."""
    x_f32 = x.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_MAX
    q = (x_f32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return q, scale


class _FP8RowwiseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        w_2d = weight.contiguous()
        ctx.save_for_backward(x_2d, w_2d)
        ctx.orig_shape = orig_shape
        ctx.out_dtype = x.dtype

        xq, sx = _quantize_rowwise(x_2d)
        wq, sw = _quantize_rowwise(w_2d)
        out = torch._scaled_mm(xq, wq.T, scale_a=sx, scale_b=sw.T, out_dtype=torch.bfloat16)
        return out.reshape(*orig_shape[:-1], w_2d.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x_2d, w_2d = ctx.saved_tensors
        dy_2d = grad_out.reshape(-1, w_2d.shape[0]).contiguous()
        grad_x = grad_w = None

        if ctx.needs_input_grad[0]:
            # dX = dY @ W  ([M, N] @ [N, K] -> [M, K])
            dyq, sdy = _quantize_rowwise(dy_2d)
            wt_2d = w_2d.T.contiguous()  # [K, N]
            wtq, swt = _quantize_rowwise(wt_2d)
            gx = torch._scaled_mm(dyq, wtq.T, scale_a=sdy, scale_b=swt.T, out_dtype=torch.bfloat16)
            grad_x = gx.reshape(ctx.orig_shape).to(ctx.out_dtype)

        if ctx.needs_input_grad[1]:
            # dW = dY.T @ X  ([N, M] @ [M, K] -> [N, K])
            dyt_2d = dy_2d.T.contiguous()  # [N, M]
            xt_2d = x_2d.T.contiguous()  # [K, M]
            dytq, sdyt = _quantize_rowwise(dyt_2d)
            xtq, sxt = _quantize_rowwise(xt_2d)
            gw = torch._scaled_mm(dytq, xtq.T, scale_a=sdyt, scale_b=sxt.T, out_dtype=torch.bfloat16)
            grad_w = gw.to(w_2d.dtype)

        return grad_x, grad_w


def fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `F.linear(x, weight)` (no bias) using rowwise FP8 on CUDA when
    shapes are 16-aligned; falls back to standard `F.linear` on CPU or unaligned shapes."""
    if x.is_cuda and weight.ndim == 2:
        m = x.numel() // x.shape[-1]
        n, k = weight.shape
        if (m % 16 == 0) and (n % 16 == 0) and (k % 16 == 0):
            return _FP8RowwiseLinear.apply(x, weight)
    return torch.nn.functional.linear(x, weight)
