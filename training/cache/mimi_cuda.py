"""Stateless channels-last CUDA Mimi encoder optimized for Blackwell (RTX 5090, sm_120).

Replaces moshi's streaming `MimiModel.encode()` for non-streaming batch encoding across cache
generation (`cuda.cuda_mimi`: `cuda.py`, `noisy.py`) and fine-tuning /
evaluation (`lm_training`'s `MimiModel.encode` on CUDA: `streams.py`, `hibiki_helpers.py`,
`eval.py`, `train.py`, `grpo.py`):

- Zero `StreamingConv1d` state allocations or `torch.cat` padding copies.
- Channels-last `(B, T, C)` end-to-end (eliminates NCL<->NLC transposes through SEANet,
  encoder transformer, and 16-codebook RVQ).
- FP16 SEANet activations with FP32 Tensor Core accumulation (same 10-bit mantissa as TF32,
  halves SEANet HBM traffic & activation VRAM; keeps transformer norms/residuals and RVQ in fp32).
- Fused Triton kernels for `init_conv1d_k7`, Stages 0-2 SEANet residual+ELU blocks, strided
  downsample shift-add+bias, and RVQ nearest-centroid lookup; stage 3's residual block (C=512,
  few frames) runs in PyTorch.
- Zero-copy strided downsampling (`kernel = 2 * stride`) via a single cuBLAS GEMM per stage.
- Cached RoPE tables + cached sliding-window SDPA masks, eager.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
FRAME_SAMPLES = 1920
RATIOS = (4, 5, 6, 8)  # SEANet strides, 24 kHz -> 25 Hz (product = 960)
SEANET_INDICES = ((1, 3), (4, 6), (7, 9), (10, 12))  # (residual, downsample) in encoder.model
CONTEXT = 250  # encoder transformer sliding-window attention span (25 Hz frames)
NUM_HEADS = 8
CODEBOOK_EPS = 1e-5
MAX_PERIOD = 10_000.0
DTYPE = torch.float16  # SEANet activations and the transformer's linears; norms, residuals and RVQ in fp32


# --------------------------------------------------------------------------- Triton kernels


@triton.jit
def _elu_f32(x):
    # Clamp argument to exp <= 0 so the unused branch never overflows for large x > 0.
    return tl.where(x > 0.0, x, tl.exp(tl.minimum(x, 0.0)) - 1.0)


@triton.jit
def _init_conv1d_k7_nlc_kernel(
    x_ptr,   # [B, T] fp32
    w_ptr,   # [7, 64] fp32 (tap k, out_c)
    b_ptr,   # [64] fp32
    y_ptr,   # [B, T, 64] fp16
    T,
    BLOCK_T: tl.constexpr,
):
    """Causal Conv1d(1 -> 64, k=7, s=1) writing channels-last [B, T, 64] with zero pad copy."""
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    c_offs = tl.arange(0, 64)

    xb = x_ptr + pid_b.to(tl.int64) * T
    yb = y_ptr + pid_b.to(tl.int64) * T * 64

    acc = tl.broadcast_to(tl.load(b_ptr + c_offs)[None, :], (BLOCK_T, 64)).to(tl.float32)
    for k in tl.static_range(7):
        row = t_offs - 6 + k
        in_mask = (row >= 0) & t_mask
        xv = tl.load(xb + row, mask=in_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + k * 64 + c_offs).to(tl.float32)
        acc += xv[:, None] * wv[None, :]

    tl.store(yb + t_offs[:, None].to(tl.int64) * 64 + c_offs[None, :], acc.to(tl.float16), mask=t_mask[:, None])


@triton.jit
def _seanet_residual_elu_kernel(
    x_ptr,   # [B, T, C] fp16
    w1_ptr,  # [3, C, H] fp16
    b1_ptr,  # [H] fp32
    w2_ptr,  # [H, C] fp16
    b2_ptr,  # [C] fp32
    y_ptr,   # [B, T, C] fp16
    T,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused elu(x + W2 @ elu(conv3(elu(x)) + b1) + b2) keeping H in registers/SRAM."""
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T
    h_offs = tl.arange(0, H)

    xb = x_ptr + pid_b.to(tl.int64) * T * C
    yb = y_ptr + pid_b.to(tl.int64) * T * C

    mid = tl.zeros((BLOCK_T, H), dtype=tl.float32)
    for k in tl.static_range(3):
        row = t_offs - 2 + k
        row_valid = (row >= 0) & t_mask
        row_i64 = tl.maximum(row, 0).to(tl.int64)
        for c0 in tl.static_range(0, C, BLOCK_C):
            c_offs = c0 + tl.arange(0, BLOCK_C)
            xv = tl.load(
                xb + row_i64[:, None] * C + c_offs[None, :],
                mask=row_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            # Causal zero-padding: rows < 0 are 0.0, and elu(0.0) == 0.0.
            a = _elu_f32(xv)
            a_dot = a.to(tl.float16)
            w1 = tl.load(w1_ptr + (k * C + c_offs[:, None]) * H + h_offs[None, :])
            w1_dot = w1.to(tl.float16)
            mid = tl.dot(a_dot, w1_dot, acc=mid)

    b1 = tl.load(b1_ptr + h_offs).to(tl.float32)
    mid = _elu_f32(mid + b1[None, :])
    mid_dot = mid.to(tl.float16)

    row_cur = tl.minimum(t_offs, T - 1).to(tl.int64)
    for c0 in tl.static_range(0, C, BLOCK_C):
        c_offs = c0 + tl.arange(0, BLOCK_C)
        w2 = tl.load(w2_ptr + h_offs[:, None] * C + c_offs[None, :])
        w2_dot = w2.to(tl.float16)
        acc = tl.dot(mid_dot, w2_dot, out_dtype=tl.float32)
        b2 = tl.load(b2_ptr + c_offs).to(tl.float32)
        x_cur = tl.load(
            xb + row_cur[:, None] * C + c_offs[None, :],
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        out = _elu_f32(acc + b2[None, :] + x_cur)
        tl.store(
            yb + t_offs[:, None].to(tl.int64) * C + c_offs[None, :],
            out.to(tl.float16),
            mask=t_mask[:, None],
        )


@triton.jit
def _shift_add_bias_kernel(
    y_ptr,    # [B, T, 2 * C_out]
    b_ptr,    # [C_out]
    out_ptr,  # [B, T, C_out] fp16
    T,
    C_OUT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out[b, t, c] = (y[b, t-1, c] if t > 0 else 0) + y[b, t, C_OUT + c] + b[c]."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offs < T
    c_mask = c_offs < C_OUT
    mask = t_mask[:, None] & c_mask[None, :]

    stride_y = 2 * C_OUT
    yb = y_ptr + pid_b.to(tl.int64) * T * stride_y
    ob = out_ptr + pid_b.to(tl.int64) * T * C_OUT

    curr = tl.load(
        yb + t_offs[:, None].to(tl.int64) * stride_y + (C_OUT + c_offs)[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    prev_t = t_offs - 1
    prev_mask = (prev_t >= 0)[:, None] & mask
    prev = tl.load(
        yb + tl.maximum(prev_t, 0)[:, None].to(tl.int64) * stride_y + c_offs[None, :],
        mask=prev_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    res = prev + curr + bias[None, :]
    tl.store(ob + t_offs[:, None].to(tl.int64) * C_OUT + c_offs[None, :], res.to(tl.float16), mask=mask)


@triton.jit
def _rvq_argmin_kernel(
    dot_ptr,        # [N, K] fp32
    half_norm_ptr,  # [K] fp32
    idx_ptr,        # [N] int32
    N,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """idx[n] = argmin_k (half_norm[k] - dot[n, k]) without materializing [N, K] diff."""
    row = tl.program_id(0)
    if row >= N:
        return
    row_ptr = dot_ptr + row.to(tl.int64) * K
    best_val = float("inf")
    best_idx = 0
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        hn = tl.load(half_norm_ptr + k_offs, mask=k_mask, other=float("inf")).to(tl.float32)
        dv = tl.load(row_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        dist = tl.where(k_mask, hn - dv, float("inf"))
        blk_min = tl.min(dist, axis=0)
        if blk_min < best_val:
            # First index achieving blk_min (matches PyTorch argmin tie-breaking).
            is_min = dist == blk_min
            blk_idx = tl.min(tl.where(is_min, k_offs, K), axis=0)
            best_val = blk_min
            best_idx = blk_idx
    tl.store(idx_ptr + row, best_idx.to(tl.int32))


# --------------------------------------------------------------------------- Helpers


def _rope_tables(
    steps: int, dim: int, device: torch.device, cache: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, dim)
    entry = cache.get(key)
    if entry is None or entry[0].shape[0] < steps:
        length = max(2048, steps)
        ds = torch.arange(dim // 2, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(MAX_PERIOD) * 2.0 / dim))
        ts = torch.arange(length, device=device, dtype=torch.float32).view(-1, 1)
        angles = freqs * ts
        entry = (torch.cos(angles), torch.sin(angles))
        cache[key] = entry
    return entry[0][:steps], entry[1][:steps]


def _window_mask(
    steps: int, device: torch.device, cache: dict[tuple[torch.device, int], torch.Tensor]
) -> torch.Tensor:
    key = (device, steps)
    mask = cache.get(key)
    if mask is None:
        pos = torch.arange(steps, device=device)
        delta = pos[:, None] - pos[None, :]
        mask = ((delta >= 0) & (delta < CONTEXT)).unsqueeze(0).unsqueeze(0)
        cache[key] = mask
    return mask


def _apply_rope_interleaved(
    q: torch.Tensor, k: torch.Tensor, rotr: torch.Tensor, roti: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved RoPE (`moshi/modules/rope.py` `interleave=True`) on `[B, H, T, D]` or `[B, T, H, D]`."""
    dtype = q.dtype
    qv = q.view(*q.shape[:-1], -1, 2)
    kv = k.view(*k.shape[:-1], -1, 2)
    qr, qi = qv[..., 0].float(), qv[..., 1].float()
    kr, ki = kv[..., 0].float(), kv[..., 1].float()
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    qo = torch.stack((qor.to(dtype), qoi.to(dtype)), dim=-1).view_as(q)
    ko = torch.stack((kor.to(dtype), koi.to(dtype)), dim=-1).view_as(k)
    return qo, ko


def _pad_for_frame_size(x: torch.Tensor) -> torch.Tensor:
    rem = x.shape[-1] % FRAME_SAMPLES
    if rem:
        x = F.pad(x, (0, FRAME_SAMPLES - rem))
    return x


def _causal_conv3_nlc(x: torch.Tensor, w_3ic: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Causal kernel-3 conv in channels-last [B, T, C_in] -> [B, T, C_out] without padding copies."""
    # w_3ic: [3, C_in, C_out] -> 3 matmuls over shifted views
    B, T, _ = x.shape
    out = F.linear(x, w_3ic[2].T, b)
    if T > 1:
        out[:, 1:, :] = out[:, 1:, :] + F.linear(x[:, :-1, :], w_3ic[1].T)
    if T > 2:
        out[:, 2:, :] = out[:, 2:, :] + F.linear(x[:, :-2, :], w_3ic[0].T)
    return out


# --------------------------------------------------------------------------- Fast Mimi Encoder


class _TransformerLayer(nn.Module):
    """One Mimi encoder transformer layer in `(B, T, C)`: fp16 linears, fp32 LayerNorm/LayerScale."""

    def __init__(
        self,
        norm1_w: torch.Tensor,
        norm1_b: torch.Tensor,
        in_proj_w: torch.Tensor,
        out_proj_w: torch.Tensor,
        ls1: torch.Tensor,
        norm2_w: torch.Tensor,
        norm2_b: torch.Tensor,
        lin1_w: torch.Tensor,
        lin2_w: torch.Tensor,
        ls2: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("norm1_w", norm1_w.float())
        self.register_buffer("norm1_b", norm1_b.float())
        self.register_buffer("ls1", ls1.float())
        self.register_buffer("norm2_w", norm2_w.float())
        self.register_buffer("norm2_b", norm2_b.float())
        self.register_buffer("ls2", ls2.float())

        dev = in_proj_w.device
        self.in_proj = nn.Linear(in_proj_w.shape[1], in_proj_w.shape[0], bias=False, device=dev, dtype=DTYPE)
        self.out_proj = nn.Linear(out_proj_w.shape[1], out_proj_w.shape[0], bias=False, device=dev, dtype=DTYPE)
        self.lin1 = nn.Linear(lin1_w.shape[1], lin1_w.shape[0], bias=False, device=dev, dtype=DTYPE)
        self.lin2 = nn.Linear(lin2_w.shape[1], lin2_w.shape[0], bias=False, device=dev, dtype=DTYPE)
        with torch.no_grad():
            self.in_proj.weight.copy_(in_proj_w.to(DTYPE))
            self.out_proj.weight.copy_(out_proj_w.to(DTYPE))
            self.lin1.weight.copy_(lin1_w.to(DTYPE))
            self.lin2.weight.copy_(lin2_w.to(DTYPE))

    def forward(
        self,
        xs: torch.Tensor,
        rotr: torch.Tensor,
        roti: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, D = xs.shape
        head = D // NUM_HEADS
        h = F.layer_norm(xs, (D,), self.norm1_w, self.norm1_b, 1e-5).to(DTYPE)
        qkv = self.in_proj(h)
        q, k, v = qkv.view(B, T, 3, NUM_HEADS, head).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = _apply_rope_interleaved(q, k, rotr[None, None, :, :], roti[None, None, :, :])
        if attn_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        xs = xs + self.ls1 * self.out_proj(attn).float()
        h = F.layer_norm(xs, (D,), self.norm2_w, self.norm2_b, 1e-5).to(DTYPE)
        ff = self.lin2(F.gelu(self.lin1(h)))
        return xs + self.ls2 * ff.float()


class FastMimiEncoder(nn.Module):
    """Stateless, encode-only Mimi optimized for RTX 5090 / Blackwell CUDA, on a CUDA device.

    Drop-in encode interface matching `moshi.models.MimiModel`:
      `codes = encoder.encode(x)` for `x: [B, 1, T]` float32 -> `[B, num_codebooks, ceil(T/1920)]` int64.
    """

    sample_rate: int = SAMPLE_RATE
    frame_rate: float = FRAME_RATE
    frame_size: int = FRAME_SAMPLES

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        num_codebooks: int = 16,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"FastMimiEncoder runs on CUDA only, not {self._device}")

        self._num_codebooks = num_codebooks
        total_codebooks = 1 + sum(
            1 for k in weights if k.startswith("quantizer.rvq_rest.vq.layers.") and k.endswith("._codebook.embedding_sum")
        )
        self.cardinality = int(weights["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"].shape[0])

        # 1. Init conv: [64, 1, 7] -> w_triton [7, 64] fp32
        w0 = weights["encoder.model.0.conv.conv.weight"].to(self._device, dtype=torch.float32)
        b0 = weights["encoder.model.0.conv.conv.bias"].to(self._device, dtype=torch.float32)
        self.register_buffer("init_w_k7", w0[:, 0, :].T.contiguous())  # [7, 64]
        self.register_buffer("init_b", b0)

        # 2. SEANet stages (4 stages: C=64, 128, 256, 512; strides 4, 5, 6, 8)
        self.stage_strides: list[int] = list(RATIOS)
        for i, ((res_idx, down_idx), stride) in enumerate(zip(SEANET_INDICES, RATIOS, strict=True)):
            w1 = weights[f"encoder.model.{res_idx}.block.1.conv.conv.weight"].to(self._device)  # [H, C, 3]
            b1 = weights[f"encoder.model.{res_idx}.block.1.conv.conv.bias"].to(self._device, dtype=torch.float32)
            w2 = weights[f"encoder.model.{res_idx}.block.3.conv.conv.weight"].to(self._device)  # [C, H, 1]
            b2 = weights[f"encoder.model.{res_idx}.block.3.conv.conv.bias"].to(self._device, dtype=torch.float32)
            # Triton / NLC layouts: w1 [3, C, H], w2 [H, C]
            w1_nlc = w1.permute(2, 1, 0).contiguous().to(DTYPE)
            w2_nlc = w2[:, :, 0].T.contiguous().to(DTYPE)
            self.register_buffer(f"s{i}_w1", w1_nlc)
            self.register_buffer(f"s{i}_b1", b1)
            self.register_buffer(f"s{i}_w2", w2_nlc)
            self.register_buffer(f"s{i}_b2", b2)

            # Downsample conv: [C_out, C_in, 2*s] -> w_both [2 * C_out, s * C_in]
            wd = weights[f"encoder.model.{down_idx}.conv.conv.weight"].to(self._device)
            bd = weights[f"encoder.model.{down_idx}.conv.conv.bias"].to(self._device, dtype=torch.float32)
            c_out, c_in, k = wd.shape
            assert k == 2 * stride
            wd_halves = wd.permute(0, 2, 1).contiguous().view(c_out, 2, stride * c_in)
            w_both = torch.cat([wd_halves[:, 0, :], wd_halves[:, 1, :]], dim=0).contiguous().to(DTYPE)
            self.register_buffer(f"s{i}_wd_both", w_both)
            self.register_buffer(f"s{i}_bd", bd)

        # 3. Final SEANet conv: 1024 -> 512, k=3
        wf = weights["encoder.model.14.conv.conv.weight"].to(self._device)  # [512, 1024, 3]
        bf = weights["encoder.model.14.conv.conv.bias"].to(self._device, dtype=torch.float32)
        self.register_buffer("final_w", wf.permute(2, 1, 0).contiguous().to(DTYPE))
        self.register_buffer("final_b", bf)

        # 4. Encoder transformer (8 layers, d=512, 8 heads, ff=2048, ctx=250)
        prefix = "encoder_transformer.transformer.layers"
        n_layers = 1 + max(int(k.split(".")[3]) for k in weights if k.startswith(prefix))
        layers = []
        for i in range(n_layers):
            p = f"{prefix}.{i}"
            in_key = f"{p}.self_attn.in_proj_weight"
            if in_key not in weights:
                in_key = f"{p}.self_attn.in_projs.0.weight"
            out_key = f"{p}.self_attn.out_proj.weight"
            if out_key not in weights:
                out_key = f"{p}.self_attn.out_projs.0.weight"
            layers.append(
                _TransformerLayer(
                    weights[f"{p}.norm1.weight"].to(self._device),
                    weights[f"{p}.norm1.bias"].to(self._device),
                    weights[in_key].to(self._device),
                    weights[out_key].to(self._device),
                    weights[f"{p}.layer_scale_1.scale"].to(self._device),
                    weights[f"{p}.norm2.weight"].to(self._device),
                    weights[f"{p}.norm2.bias"].to(self._device),
                    weights[f"{p}.linear1.weight"].to(self._device),
                    weights[f"{p}.linear2.weight"].to(self._device),
                    weights[f"{p}.layer_scale_2.scale"].to(self._device),
                )
            )
        self.transformer_layers = nn.ModuleList(layers)

        # 5. Frame-rate downsample 25 Hz -> 12.5 Hz (512 -> 512, k=4, s=2, replicate left-pad 2)
        self.register_buffer(
            "ds_w", weights["downsample.conv.conv.conv.weight"].to(self._device, dtype=torch.float32)
        )

        # 6. Split RVQ (keep in FP32 for exact codebook boundary precision)
        self.register_buffer(
            "rvq_first_proj",
            weights["quantizer.rvq_first.input_proj.weight"][:, :, 0].to(self._device, dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "rvq_rest_proj",
            weights["quantizer.rvq_rest.input_proj.weight"][:, :, 0].to(self._device, dtype=torch.float32).contiguous(),
        )
        first_emb, first_hn = self._load_codebook(weights, "quantizer.rvq_first.vq.layers.0._codebook")
        self.register_buffer("rvq_first_emb", first_emb)
        self.register_buffer("rvq_first_hn", first_hn)
        rest_embs, rest_hns = [], []
        for i in range(total_codebooks - 1):
            e, hn = self._load_codebook(weights, f"quantizer.rvq_rest.vq.layers.{i}._codebook")
            rest_embs.append(e)
            rest_hns.append(hn)
        self.register_buffer("rvq_rest_embs", torch.stack(rest_embs, dim=0))  # [Q-1, K, D]
        self.register_buffer("rvq_rest_hns", torch.stack(rest_hns, dim=0))    # [Q-1, K]

        self._rope_cache: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def _load_codebook(self, weights: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        emb_sum = weights[f"{prefix}.embedding_sum"].to(self._device, dtype=torch.float32)
        usage = weights[f"{prefix}.cluster_usage"].to(self._device, dtype=torch.float32).clamp(min=CODEBOOK_EPS)
        emb = (emb_sum / usage[:, None]).contiguous()
        half_norm = (0.5 * (emb * emb).sum(dim=-1)).contiguous()
        return emb, half_norm

    @classmethod
    def from_checkpoint(
        cls,
        mimi_weight: str | Path,
        num_codebooks: int = 16,
        device: torch.device | str = "cuda",
    ) -> "FastMimiEncoder":
        from safetensors.torch import load_file

        weights = load_file(str(mimi_weight), device="cpu")
        return cls(weights, num_codebooks=num_codebooks, device=device).eval()

    @classmethod
    def from_moshi(
        cls,
        mimi: Any,
        device: torch.device | str | None = None,
    ) -> "FastMimiEncoder":
        dev = torch.device(device) if device is not None else next(mimi.parameters()).device
        sd = {k: v.detach() for k, v in mimi.state_dict().items()}
        return cls(sd, num_codebooks=int(mimi.num_codebooks), device=dev).eval()

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    # ---- SEANet ops ----

    def _init_conv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T] fp32 -> [B, T, 64] fp16
        B, _, T = x.shape
        x_2d = x[:, 0, :].contiguous()
        y = torch.empty((B, T, 64), device=x.device, dtype=DTYPE)
        block_t = 128
        grid = (triton.cdiv(T, block_t), B)
        _init_conv1d_k7_nlc_kernel[grid](
            x_2d,
            self.init_w_k7,
            self.init_b,
            y,
            T,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return y

    def _residual_elu(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        w1 = getattr(self, f"s{stage}_w1")
        b1 = getattr(self, f"s{stage}_b1")
        w2 = getattr(self, f"s{stage}_w2")
        b2 = getattr(self, f"s{stage}_b2")
        B, T, C = x.shape
        H = w1.shape[2]
        if stage < 3:
            y = torch.empty_like(x)
            block_t = 64
            block_c = 64
            grid = (triton.cdiv(T, block_t), B)
            _seanet_residual_elu_kernel[grid](
                x,
                w1,
                b1,
                w2,
                b2,
                y,
                T,
                C=C,
                H=H,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                    num_warps=4 if H <= 64 else 8,
            )
            return y
        # Stage 3 (C=512, few frames): PyTorch
        h = F.elu(x.float()).to(DTYPE)
        h = _causal_conv3_nlc(h, w1, b1.to(DTYPE))
        h = F.elu(h.float()).to(DTYPE)
        h = F.linear(h, w2.T, b2.to(DTYPE))
        return F.elu((x.float() + h.float())).to(DTYPE)

    def _downsample(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        stride = self.stage_strides[stage]
        w_both = getattr(self, f"s{stage}_wd_both")
        bias = getattr(self, f"s{stage}_bd")
        B, T, C = x.shape
        T_out = T // stride
        c_out = w_both.shape[0] // 2
        y = F.linear(x.reshape(B, T_out, stride * C), w_both)  # [B, T_out, 2 * c_out]
        out = torch.empty((B, T_out, c_out), device=x.device, dtype=DTYPE)
        block_t = 64
        block_c = 64
        grid = (triton.cdiv(T_out, block_t), triton.cdiv(c_out, block_c), B)
        _shift_add_bias_kernel[grid](
            y,
            bias,
            out,
            T_out,
            C_OUT=c_out,
            BLOCK_T=block_t,
            BLOCK_C=block_c,
            num_warps=4,
        )
        return out

    def _seanet(self, x: torch.Tensor) -> torch.Tensor:
        hs = self._init_conv(x)
        for stage in range(4):
            hs = self._residual_elu(hs, stage)
            hs = self._downsample(hs, stage)
        hs = F.elu(hs.float()).to(DTYPE)
        hs = _causal_conv3_nlc(hs, self.final_w, self.final_b.to(DTYPE))
        return hs.float()

    # ---- Encoder transformer ----

    def _transformer(self, xs: torch.Tensor) -> torch.Tensor:
        T, D = xs.shape[1], xs.shape[2]
        rotr, roti = _rope_tables(T, D // NUM_HEADS, xs.device, self._rope_cache)
        attn_mask = None if T <= CONTEXT else _window_mask(T, xs.device, self._mask_cache)
        for layer in self.transformer_layers:
            xs = layer(xs, rotr, roti, attn_mask)
        return xs

    # ---- RVQ ----

    @staticmethod
    def _quantize_one(residual: torch.Tensor, emb: torch.Tensor, half_norm: torch.Tensor) -> torch.Tensor:
        # residual: [N, 256] fp32, emb: [2048, 256] fp32, half_norm: [2048] fp32
        dot = torch.mm(residual, emb.T)  # [N, 2048]
        N, K = dot.shape
        idx = torch.empty((N,), device=residual.device, dtype=torch.int32)
        _rvq_argmin_kernel[(N,)](dot, half_norm, idx, N, K=K, BLOCK_K=512, num_warps=4)
        return idx

    def _quantize(self, hs: torch.Tensor, num_codebooks: int) -> torch.Tensor:
        # hs: [B, T_12_5hz, 512] fp32
        B, T, _ = hs.shape
        flat = hs.reshape(B * T, hs.shape[2])
        codes: list[torch.Tensor] = []

        # Semantic codebook 0
        r0 = F.linear(flat, self.rvq_first_proj)
        c0 = self._quantize_one(r0, self.rvq_first_emb, self.rvq_first_hn)
        codes.append(c0)

        # Acoustic residual codebooks 1..num_codebooks-1
        if num_codebooks > 1:
            residual = F.linear(flat, self.rvq_rest_proj)
            n_rest = num_codebooks - 1
            for q in range(n_rest):
                emb = self.rvq_rest_embs[q]
                hn = self.rvq_rest_hns[q]
                idx = self._quantize_one(residual, emb, hn)
                codes.append(idx)
                if q + 1 < n_rest:
                    residual = residual - F.embedding(idx.long(), emb)

        return torch.stack(codes, dim=0).view(num_codebooks, B, T).transpose(0, 1).contiguous()

    # ---- Public API ----

    @torch.no_grad()
    def encode(self, x: torch.Tensor, num_codebooks: int | None = None) -> torch.Tensor:
        """Encode `[B, 1, T]` float32 24 kHz audio -> `[B, K, ceil(T / 1920)]` int64 codes."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T] audio, got {tuple(x.shape)}")
        k = self._num_codebooks if num_codebooks is None else num_codebooks
        x = _pad_for_frame_size(x)
        hs = self._transformer(self._seanet(x))
        # 25 Hz -> 12.5 Hz: replicate left-pad 2, k=4, s=2
        hs_bct = hs.transpose(1, 2)
        hs_pad = torch.cat((hs_bct[..., :1], hs_bct[..., :1], hs_bct), dim=-1)
        hs_12 = F.conv1d(hs_pad, self.ds_w, stride=2).transpose(1, 2).contiguous()
        return self._quantize(hs_12, k).to(torch.int64)
