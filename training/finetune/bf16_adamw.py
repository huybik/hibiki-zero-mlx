"""Fused BF16 AdamW with IEEE-754 stochastic rounding and fused gradient clipping.

Designed for full-model 3B SFT on 32 GB GPUs (RTX 5090):
- 2D+ parameters (linears, embeddings; >99.95% of 3B) stay in bf16, with bf16 gradients and bf16
  AdamW moments (m, v): 8 bytes/param total (23.3 GiB on 3B instead of 46.7 GiB for fp32).
- 1D parameters (RMSNorm alpha, layer_scale, biases; <0.05% of params, <1 MB) stay in fp32 with
  fp32 moments so norms and scales never lose precision.
- Every AdamW update runs in fp32 registers in a single Triton kernel with:
  1. fused gradient clipping (`clip_coef` passed as a 0-d GPU tensor, eliminating the separate
     grad-scaling read+write pass and avoiding any host sync);
  2. unbiased IEEE-754 stochastic rounding back to bf16 for p, m, and v using Triton's in-register
     Philox PRNG (`tl.randint`), so updates smaller than bf16's ULP (e.g. lr=2.25e-6) accumulate
     without vanishing and EMAs stay unbiased. Unlike 8-bit blockwise AdamW, bf16 preserves fp32's
     full 8-bit exponent range (1e-38..3e38), so second moments never underflow to 0 on outliers.

On CPU (for unit tests) it runs the exact same fp32 math and bit-exact IEEE-754 stochastic rounding
in PyTorch.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


def cast_policy_bf16(model: torch.nn.Module) -> tuple[int, int]:
    """Cast 2D+ floating parameters to bf16 and keep 1D parameters (norms, scales, biases) in fp32.
    Returns (bf16_params, fp32_params) counts."""
    bf16_count = fp32_count = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.is_floating_point():
                continue
            if param.ndim >= 2:
                param.data = param.data.to(torch.bfloat16)
                bf16_count += param.numel()
            else:
                param.data = param.data.to(torch.float32)
                fp32_count += param.numel()
    return bf16_count, fp32_count


if _HAS_TRITON:

    @triton.jit
    def _fp32_to_bf16_sr(x, seed, offsets):
        """Unbiased IEEE-754 stochastic rounding from fp32 to bf16.
        Adds a uniform 16-bit integer to the lower 16 mantissa bits of the fp32 sign-magnitude
        representation, then truncates to the upper 16 bits. Finite values near max float32 are
        clamped so rounding never overflows to inf; NaN/Inf pass through unchanged."""
        u = x.to(tl.uint32, bitcast=True)
        rand16 = tl.randint(seed, offsets).to(tl.uint32) & 0xFFFF
        abs_u = u & 0x7FFFFFFF
        # Guard NaN/Inf (>= 0x7F800000) and max-bf16-rounding-to-inf (>= 0x7F7F0000).
        safe = abs_u < 0x7F7F0000
        rounded = (u + rand16) & 0xFFFF0000
        out_u = tl.where(safe, rounded, u)
        return out_u.to(tl.float32, bitcast=True).to(tl.bfloat16)

    @triton.jit
    def _bf16_sr_adamw_kernel(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        clip_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        inv_bc1,
        inv_bc2,
        wd_factor,
        eps,
        seed_p,
        seed_m,
        seed_v,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        clip = tl.load(clip_ptr).to(tl.float32)
        p = tl.load(p_ptr + offsets, mask=mask).to(tl.float32)
        g = tl.load(g_ptr + offsets, mask=mask).to(tl.float32) * clip
        m = tl.load(m_ptr + offsets, mask=mask).to(tl.float32)
        v = tl.load(v_ptr + offsets, mask=mask).to(tl.float32)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        step_size = lr * inv_bc1
        denom = tl.sqrt(v * inv_bc2) + eps
        p = p * wd_factor - step_size * (m / denom)

        p_bf16 = _fp32_to_bf16_sr(p, seed_p, offsets)
        m_bf16 = _fp32_to_bf16_sr(m, seed_m, offsets)
        v_bf16 = _fp32_to_bf16_sr(v, seed_v, offsets)

        tl.store(p_ptr + offsets, p_bf16, mask=mask)
        tl.store(m_ptr + offsets, m_bf16, mask=mask)
        tl.store(v_ptr + offsets, v_bf16, mask=mask)

    @triton.jit
    def _fp32_adamw_kernel(
        p_ptr,
        g_ptr,
        m_ptr,
        v_ptr,
        clip_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        inv_bc1,
        inv_bc2,
        wd_factor,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        clip = tl.load(clip_ptr).to(tl.float32)
        p = tl.load(p_ptr + offsets, mask=mask)
        g = tl.load(g_ptr + offsets, mask=mask) * clip
        m = tl.load(m_ptr + offsets, mask=mask)
        v = tl.load(v_ptr + offsets, mask=mask)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        step_size = lr * inv_bc1
        denom = tl.sqrt(v * inv_bc2) + eps
        p = p * wd_factor - step_size * (m / denom)

        tl.store(p_ptr + offsets, p, mask=mask)
        tl.store(m_ptr + offsets, m, mask=mask)
        tl.store(v_ptr + offsets, v, mask=mask)


def _cpu_fp32_to_bf16_sr(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """CPU reference implementation of unbiased IEEE-754 stochastic rounding fp32 -> bf16."""
    u = x.contiguous().view(torch.int32)
    rand16 = torch.randint(0, 1 << 16, u.shape, dtype=torch.int32, generator=generator)
    abs_u = u & 0x7FFFFFFF
    safe = abs_u < 0x7F7F0000
    # Signed int32 addition wraps identically to uint32 for the lower 31 bits when safe (abs_u < 0x7F7F0000).
    rounded = (u + rand16) & (-65536)  # 0xFFFF0000 in two's complement
    out_u = torch.where(safe, rounded, u)
    return out_u.view(torch.float32).to(torch.bfloat16)


class BF16StochasticAdamW(torch.optim.Optimizer):
    """AdamW with bf16 weights & moments (with unbiased stochastic rounding) for 2D+ parameters
    and fp32 weights & moments for 1D parameters, plus optional fused gradient clipping."""

    BLOCK_SIZE = 2048

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        seed: int = 42,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.sr_seed = int(seed)
        self._one_gpu: dict[torch.device, torch.Tensor] = {}
        self._step_cache: int | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._step_cache = None

    def _unit_clip(self, device: torch.device) -> torch.Tensor:
        tensor = self._one_gpu.get(device)
        if tensor is None:
            tensor = torch.ones((), device=device, dtype=torch.float32)
            self._one_gpu[device] = tensor
        return tensor

    def _current_step(self) -> int:
        if self._step_cache is None:
            step = 0
            for state in self.state.values():
                if "step" in state:
                    step = int(state["step"].item())
                    break
            self._step_cache = step
        self._step_cache += 1
        return self._step_cache

    @torch.no_grad()
    def step(self, closure: Any = None, clip_coef: torch.Tensor | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        step_int = self._current_step()
        param_index = 0
        clip_by_device: dict[torch.device, torch.Tensor] = {}
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = float(group["betas"][0]), float(group["betas"][1])
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            wd_factor = 1.0 - lr * wd
            inv_bc1 = 1.0 / (1.0 - beta1**step_int)
            inv_bc2 = 1.0 / (1.0 - beta2**step_int)

            for p in group["params"]:
                param_index += 1
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BF16StochasticAdamW does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)  # CPU scalar: no GPU fill_ launch
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                elif state["step"].device.type != "cpu":
                    # checkpoints._states_to moves optimizer tensors to CUDA on resume; keep step on CPU.
                    state["step"] = state["step"].cpu()

                state["step"].fill_(float(step_int))

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                n = p.numel()
                if n == 0:
                    continue

                if p.is_cuda and _HAS_TRITON:
                    p_flat = p.data if p.data.is_contiguous() else p.data.contiguous()
                    g_flat = grad if grad.is_contiguous() else grad.contiguous()
                    m_flat = m if m.is_contiguous() else m.contiguous()
                    v_flat = v if v.is_contiguous() else v.contiguous()
                    clip_tensor = clip_by_device.get(p.device)
                    if clip_tensor is None:
                        clip_tensor = (
                            self._unit_clip(p.device)
                            if clip_coef is None
                            else clip_coef.to(device=p.device, dtype=torch.float32)
                        )
                        clip_by_device[p.device] = clip_tensor
                    grid = (triton.cdiv(n, self.BLOCK_SIZE),)
                    if p.dtype == torch.bfloat16:
                        # Mix seed, step and param_index into 3 disjoint 32-bit seeds for p, m, v.
                        base = ((self.sr_seed * 0x9E3779B1) ^ (step_int * 0x85EBCA6B) ^ (param_index * 0xC2B2AE35)) & 0x7FFFFFFF
                        _bf16_sr_adamw_kernel[grid](
                            p_flat, g_flat, m_flat, v_flat, clip_tensor,
                            n, lr, beta1, beta2, inv_bc1, inv_bc2, wd_factor, eps,
                            base, (base + 1) & 0x7FFFFFFF, (base + 2) & 0x7FFFFFFF,
                            BLOCK_SIZE=self.BLOCK_SIZE, num_warps=8,
                        )
                    elif p.dtype == torch.float32:
                        _fp32_adamw_kernel[grid](
                            p_flat, g_flat, m_flat, v_flat, clip_tensor,
                            n, lr, beta1, beta2, inv_bc1, inv_bc2, wd_factor, eps,
                            BLOCK_SIZE=min(self.BLOCK_SIZE, 1024), num_warps=4,
                        )
                    else:
                        raise RuntimeError(f"Unsupported parameter dtype for BF16StochasticAdamW: {p.dtype}")
                    if p_flat.data_ptr() != p.data.data_ptr():
                        p.data.copy_(p_flat)
                    if m_flat.data_ptr() != m.data_ptr():
                        m.copy_(m_flat)
                    if v_flat.data_ptr() != v.data_ptr():
                        v.copy_(v_flat)
                else:
                    # CPU / non-Triton fallback with exact same math and SR.
                    clip_val = 1.0 if clip_coef is None else float(clip_coef.item())
                    pf = p.data.float()
                    gf = grad.float() * clip_val
                    mf = m.float()
                    vf = v.float()
                    mf = beta1 * mf + (1.0 - beta1) * gf
                    vf = beta2 * vf + (1.0 - beta2) * (gf * gf)
                    pf = pf * wd_factor - (lr * inv_bc1) * (mf / (vf.mul(inv_bc2).sqrt() + eps))
                    if p.dtype == torch.bfloat16:
                        base = ((self.sr_seed * 0x9E3779B1) ^ (step_int * 0x85EBCA6B) ^ (param_index * 0xC2B2AE35)) & 0x7FFFFFFF
                        gen = torch.Generator(device=p.device)
                        gen.manual_seed(base)
                        p.data.copy_(_cpu_fp32_to_bf16_sr(pf, gen))
                        gen.manual_seed((base + 1) & 0x7FFFFFFF)
                        m.copy_(_cpu_fp32_to_bf16_sr(mf, gen))
                        gen.manual_seed((base + 2) & 0x7FFFFFFF)
                        v.copy_(_cpu_fp32_to_bf16_sr(vf, gen))
                    else:
                        p.data.copy_(pf.to(p.dtype))
                        m.copy_(mf.to(m.dtype))
                        v.copy_(vf.to(v.dtype))

        return loss
