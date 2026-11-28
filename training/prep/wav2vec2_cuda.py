"""Fast Blackwell CUDA wav2vec2-base CTC aligner & GPU 24 kHz -> 16 kHz resampler.

Optimizations over stock `transformers.Wav2Vec2ForCTC` in eager fp32/TF32:
- FP16 compute (10-bit mantissa, matching TF32; keeps c0 GroupNorm statistics & lm_head in fp32).
- Fused two-pass Triton masked GroupNorm + GELU on `c0` output (20 GB -> ~2-3 GB HBM traffic).
- `weight_norm` folded once at load on `pos_conv_embed.conv`.
- Fused QKV projection (`768 -> 2304`) per encoder layer (36 GEMMs -> 12).
- Packed varlen FlashAttention-2 (`flash_attn_varlen_func`) on valid tokens only (zero padding
  tokens through attention, FFN, lm_head, and D2H transfer!), with compiled broadcastable-mask
  SDPA fallback when `flash_attn` is absent.
- Optional Blackwell FP8 (`FrozenFP8Linear`) on the 12 transformer layers (`use_fp8=True`).
- Pure-PyTorch batched GPU Kaiser-windowed polyphase FIR `24 kHz -> 16 kHz` resampler + per-clip
  masked zero-mean/unit-variance normalization (zero `torchaudio` dependency).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _flash_attn_varlen_func = None
    _HAS_FLASH_ATTN = False


# --------------------------------------------------------------------------- Triton kernels

if _HAS_TRITON:

    @triton.jit
    def _gelu_exact_f32(x):
        return 0.5 * x * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

    @triton.jit
    def _masked_groupnorm_gelu_kernel(
        x_ptr,      # [B, C, T] fp32
        valid_ptr,  # [B] int32
        w_ptr,      # [C] fp32
        b_ptr,      # [C] fp32
        out_ptr,    # [B, C, T] out_dtype
        C,
        T,
        eps,
        BLOCK_T: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        """Two-pass exact masked GroupNorm(num_groups=C) + affine + GELU over [0, valid[b]).

        Grid: (B, C). Uses parallel Welford/centered two-pass in fp32 registers and writes
        directly to fp16/fp32 (zeroing t >= valid[b]).
        """
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        valid_len = tl.load(valid_ptr + pid_b)
        row_offset = (pid_b.to(tl.int64) * C + pid_c.to(tl.int64)) * T
        row_x = x_ptr + row_offset
        row_out = out_ptr + row_offset

        # Pass 1: exact fp32 mean over [0, valid_len)
        total = tl.zeros((), dtype=tl.float32)
        for t0 in range(0, T, BLOCK_T):
            offs = t0 + tl.arange(0, BLOCK_T)
            mask = offs < valid_len
            xv = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
            total += tl.sum(xv, axis=0)
        mean = total / tl.maximum(valid_len.to(tl.float32), 1.0)

        # Pass 2: exact centered fp32 variance over [0, valid_len)
        var_sum = tl.zeros((), dtype=tl.float32)
        for t0 in range(0, T, BLOCK_T):
            offs = t0 + tl.arange(0, BLOCK_T)
            mask = offs < valid_len
            xv = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
            diff = tl.where(mask, xv - mean, 0.0)
            var_sum += tl.sum(diff * diff, axis=0)
        var = var_sum / tl.maximum(valid_len.to(tl.float32), 1.0)
        inv_std = tl.rsqrt(var + eps)

        gamma = tl.load(w_ptr + pid_c).to(tl.float32)
        beta = tl.load(b_ptr + pid_c).to(tl.float32)
        scale = inv_std * gamma

        # Pass 3: normalize + affine + GELU + zero padding + cast to out_dtype
        for t0 in range(0, T, BLOCK_T):
            offs = t0 + tl.arange(0, BLOCK_T)
            in_bounds = offs < T
            valid_m = offs < valid_len
            xv = tl.load(row_x + offs, mask=valid_m, other=0.0).to(tl.float32)
            y = _gelu_exact_f32((xv - mean) * scale + beta)
            y = tl.where(valid_m, y, 0.0)
            out_val = y.to(tl.float16) if OUT_FP16 else y
            tl.store(row_out + offs, out_val, mask=in_bounds)


# --------------------------------------------------------------------------- GPU 24k -> 16k Resampler


def _kaiser_polyphase_24_to_16(lowpass_filter_width: int = 16, rolloff: float = 0.99, beta: float = 14.769656459379492) -> torch.Tensor:
    """Build exact 2/3 rational Kaiser-windowed sinc polyphase filter bank (matches torchaudio.functional.resample).

    For `orig_freq = 24000`, `new_freq = 16000`: gcd = 8000, `up = 2`, `down = 3`.
    Returns kernel of shape `[2, 1, K]` for `F.conv1d(..., stride=3)`.
    """
    orig_freq, new_freq = 3, 2
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64) / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=torch.float64)[:, None] / new_freq + idx[None, :]
    t = t * base_freq
    t = t.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.i0(beta * torch.sqrt((1.0 - (t / lowpass_filter_width) ** 2).clamp(min=0.0))) / torch.i0(
        torch.tensor(beta, dtype=torch.float64)
    )
    kernels = torch.sinc(t) * window * (base_freq / orig_freq)
    return kernels[:, None, :].to(torch.float32).contiguous(), width


class Resample24to16(nn.Module):
    """Batched GPU `24 kHz -> 16 kHz` Kaiser polyphase FIR resampler + masked wav2vec2 normalization."""

    def __init__(self, device: torch.device | str = "cuda"):
        super().__init__()
        kernel, width = _kaiser_polyphase_24_to_16()
        self.width = int(width)
        self.register_buffer("kernel", kernel.to(device))  # [2, 1, K]

    @staticmethod
    def output_lengths(lengths_24k: torch.Tensor) -> torch.Tensor:
        """Output 16 kHz sample count matching `soxr.resample(x, 24000, 16000)` (`round(n * 2 / 3)`)."""
        return (lengths_24k * 2 + 1) // 3

    @torch.no_grad()
    def forward(self, wav24: torch.Tensor, lengths_24k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample right-padded `wav24` `[B, 1, T24]` -> normalized `wav16` `[B, 1, T16]` + `lengths_16k` `[B]`."""
        lengths_16k = self.output_lengths(lengths_24k)
        B, _, T24 = wav24.shape
        # Exact torchaudio padding convention for 3 -> 2
        padded = F.pad(wav24, (self.width, self.width + 3))
        out = F.conv1d(padded, self.kernel, stride=3)  # [B, 2, T_blocks]
        out = out.transpose(1, 2).reshape(B, 1, -1)
        t16 = int(lengths_16k.max().item())
        out = out[:, :, :t16]

        # Per-clip zero-mean, unit-variance normalization over [0, lengths_16k[b]) (matching cache_rows.prepare)
        lens_dev = lengths_16k.to(wav24.device, non_blocking=True)
        mask = (torch.arange(t16, device=wav24.device)[None, :] < lens_dev[:, None])[:, None, :]
        m_f = mask.float()
        count = m_f.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (out * m_f).sum(-1, keepdim=True) / count
        var = ((out - mean).square() * m_f).sum(-1, keepdim=True) / count
        out = torch.where(mask, (out - mean) / torch.sqrt(var + 1e-7), 0.0)
        return out, lengths_16k


# --------------------------------------------------------------------------- Fast Wav2Vec2 CTC


def _can_compile() -> bool:
    return _HAS_TRITON and not os.environ.get("NO_TORCH_COMPILE")


def _remove_weight_norm(conv: nn.Module) -> None:
    if hasattr(nn.utils.parametrize, "is_parametrized") and nn.utils.parametrize.is_parametrized(conv, "weight"):
        nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    elif hasattr(conv, "weight_g"):
        nn.utils.remove_weight_norm(conv)


class _FusedW2VLayer(nn.Module):
    """One post-norm Wav2Vec2EncoderLayer with fused QKV projection and optional FP8 linears."""

    def __init__(self, layer: nn.Module, num_heads: int, dtype: torch.dtype, use_fp8: bool = False):
        super().__init__()
        attn = layer.attention
        ff = layer.feed_forward
        self.num_heads = num_heads
        self.embed_dim = attn.embed_dim
        self.head_dim = attn.head_dim
        self.dtype = dtype
        self.use_fp8 = use_fp8

        qkv_w = torch.cat([attn.q_proj.weight.detach(), attn.k_proj.weight.detach(), attn.v_proj.weight.detach()], dim=0)
        qkv_b = torch.cat([attn.q_proj.bias.detach(), attn.k_proj.bias.detach(), attn.v_proj.bias.detach()], dim=0)
        out_w, out_b = attn.out_proj.weight.detach(), attn.out_proj.bias.detach()
        fc1_w, fc1_b = ff.intermediate_dense.weight.detach(), ff.intermediate_dense.bias.detach()
        fc2_w, fc2_b = ff.output_dense.weight.detach(), ff.output_dense.bias.detach()

        if use_fp8:
            from training.finetune.fp8_linear import FrozenFP8Linear

            self.qkv_proj: nn.Module = FrozenFP8Linear(qkv_w.to(torch.bfloat16))
            self.out_proj: nn.Module = FrozenFP8Linear(out_w.to(torch.bfloat16))
            self.fc1: nn.Module = FrozenFP8Linear(fc1_w.to(torch.bfloat16))
            self.fc2: nn.Module = FrozenFP8Linear(fc2_w.to(torch.bfloat16))
            self.register_buffer("qkv_bias", qkv_b.to(dtype))
            self.register_buffer("out_bias", out_b.to(dtype))
            self.register_buffer("fc1_bias", fc1_b.to(dtype))
            self.register_buffer("fc2_bias", fc2_b.to(dtype))
        else:
            dev = qkv_w.device
            self.qkv_proj = nn.Linear(qkv_w.shape[1], qkv_w.shape[0], bias=True, device=dev, dtype=dtype)
            self.out_proj = nn.Linear(out_w.shape[1], out_w.shape[0], bias=True, device=dev, dtype=dtype)
            self.fc1 = nn.Linear(fc1_w.shape[1], fc1_w.shape[0], bias=True, device=dev, dtype=dtype)
            self.fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0], bias=True, device=dev, dtype=dtype)
            with torch.no_grad():
                self.qkv_proj.weight.copy_(qkv_w.to(dtype))
                self.qkv_proj.bias.copy_(qkv_b.to(dtype))
                self.out_proj.weight.copy_(out_w.to(dtype))
                self.out_proj.bias.copy_(out_b.to(dtype))
                self.fc1.weight.copy_(fc1_w.to(dtype))
                self.fc1.bias.copy_(fc1_b.to(dtype))
                self.fc2.weight.copy_(fc2_w.to(dtype))
                self.fc2.bias.copy_(fc2_b.to(dtype))

        self.ln1_eps = float(layer.layer_norm.eps)
        self.ln2_eps = float(layer.final_layer_norm.eps)
        self.register_buffer("ln1_w", layer.layer_norm.weight.detach().float())
        self.register_buffer("ln1_b", layer.layer_norm.bias.detach().float())
        self.register_buffer("ln2_w", layer.final_layer_norm.weight.detach().float())
        self.register_buffer("ln2_b", layer.final_layer_norm.bias.detach().float())

    def _lin(self, mod: nn.Module, bias_attr: str, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            # Pad rows to multiple of 16 for _scaled_mm if needed (varlen packed length can be arbitrary)
            orig_shape = x.shape
            x2d = x.reshape(-1, orig_shape[-1]).to(torch.bfloat16)
            m = x2d.shape[0]
            rem = m % 16
            if rem:
                x2d = F.pad(x2d, (0, 0, 0, 16 - rem))
            y = mod(x2d)
            if rem:
                y = y[:m]
            y = y.reshape(*orig_shape[:-1], y.shape[-1]).to(self.dtype)
            return y + getattr(self, bias_attr)
        return mod(x)

    def forward_packed(
        self, hs: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """hs: [total_tokens, D] in self.dtype -> [total_tokens, D]."""
        N, D = hs.shape
        qkv = self._lin(self.qkv_proj, "qkv_bias", hs)
        q, k, v = qkv.view(N, 3, self.num_heads, self.head_dim).unbind(dim=1)
        attn_out = _flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=False
        ).reshape(N, D)
        attn_out = self._lin(self.out_proj, "out_bias", attn_out)
        hs = F.layer_norm((hs + attn_out).float(), (D,), self.ln1_w, self.ln1_b, self.ln1_eps).to(self.dtype)
        ff = self._lin(self.fc2, "fc2_bias", F.gelu(self._lin(self.fc1, "fc1_bias", hs)))
        return F.layer_norm((hs + ff).float(), (D,), self.ln2_w, self.ln2_b, self.ln2_eps).to(self.dtype)

    def forward_padded(self, hs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """hs: [B, T, D] in self.dtype, attn_mask: [B, 1, 1, T] additive float mask (0 = attend, -inf = mask)."""
        B, T, D = hs.shape
        qkv = self._lin(self.qkv_proj, "qkv_bias", hs)
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self._lin(self.out_proj, "out_bias", attn_out)
        hs = F.layer_norm((hs + attn_out).float(), (D,), self.ln1_w, self.ln1_b, self.ln1_eps).to(self.dtype)
        ff = self._lin(self.fc2, "fc2_bias", F.gelu(self._lin(self.fc1, "fc1_bias", hs)))
        return F.layer_norm((hs + ff).float(), (D,), self.ln2_w, self.ln2_b, self.ln2_eps).to(self.dtype)


class FastWav2Vec2CTC(nn.Module):
    """Blackwell-optimized wav2vec2-base CTC model for `CUDAModels`."""

    def __init__(
        self,
        files: Path,
        columns: list[int],
        device: torch.device | str = "cuda",
        fp16: bool = True,
        use_fp8: bool = False,
        compile_encoder: bool = True,
        c0_fp16: bool = False,
    ):
        super().__init__()
        from transformers import Wav2Vec2ForCTC

        self._device = torch.device(device)
        is_cuda = self._device.type == "cuda"
        self.dtype = torch.float16 if (is_cuda and fp16) else torch.float32
        self.use_fp8 = bool(is_cuda and use_fp8)
        self.use_varlen_fa2 = bool(is_cuda and _HAS_FLASH_ATTN and self.dtype == torch.float16)
        self._use_nhwc_convs = bool(is_cuda and self.dtype == torch.float16)

        base = Wav2Vec2ForCTC.from_pretrained(files, dtype=torch.float32).to(self._device).eval()
        config = base.config
        if config.do_stable_layer_norm or config.feat_extract_norm != "group":
            raise ValueError(f"{files}: FastWav2Vec2CTC supports only the wav2vec2-base layout")
        if base.wav2vec2.adapter is not None:
            raise ValueError(f"{files} has an adapter, which FastWav2Vec2CTC does not apply")

        self._get_feat_extract_output_lengths = base._get_feat_extract_output_lengths

        # Feature extractor: c0 conv in fp16 (when c0_fp16) or fp32; GroupNorm stats always fp32
        convs = base.wav2vec2.feature_extractor.conv_layers
        self.c0_dtype = self.dtype if (is_cuda and c0_fp16) else torch.float32
        self.c0_conv = convs[0].conv.to(self._device, dtype=self.c0_dtype)
        self.c0_k = int(self.c0_conv.kernel_size[0])
        self.c0_s = int(self.c0_conv.stride[0])
        self.c0_eps = float(convs[0].layer_norm.eps)
        self.register_buffer("c0_gn_w", convs[0].layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("c0_gn_b", convs[0].layer_norm.bias.detach().to(self._device, dtype=torch.float32))
        if self._use_nhwc_convs:
            # Store c1..c6 weights as 4D channels-last [C_out, C_in, 1, k] for cuDNN Tensor Core conv2d
            self.rest_convs = None
            self._rest_strides: list[int] = []
            for idx, layer in enumerate(convs[1:]):
                w = layer.conv.weight.detach().to(self._device, dtype=self.dtype).unsqueeze(2).to(
                    memory_format=torch.channels_last
                )
                self.register_buffer(f"rc_w_{idx}", w)
                b = layer.conv.bias.detach().to(self._device, dtype=self.dtype) if layer.conv.bias is not None else None
                self.register_buffer(f"rc_b_{idx}", b)
                self._rest_strides.append(int(layer.conv.stride[0]))
        else:
            self.rest_convs = nn.ModuleList([layer.to(self._device, dtype=self.dtype) for layer in convs[1:]])
            self._rest_strides = []

        # Feature projection: fp32 LayerNorm + self.dtype Linear
        fp = base.wav2vec2.feature_projection
        self.fp_ln_eps = float(fp.layer_norm.eps)
        self.register_buffer("fp_ln_w", fp.layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("fp_ln_b", fp.layer_norm.bias.detach().to(self._device, dtype=torch.float32))
        self.fp_proj = fp.projection.to(self._device, dtype=self.dtype)

        # Positional conv: fold weight_norm ONCE at load time!
        enc = base.wav2vec2.encoder
        _remove_weight_norm(enc.pos_conv_embed.conv)
        self.pos_conv_embed = enc.pos_conv_embed.to(self._device, dtype=self.dtype)
        self.enc_ln_eps = float(enc.layer_norm.eps)
        self.register_buffer("enc_ln_w", enc.layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("enc_ln_b", enc.layer_norm.bias.detach().to(self._device, dtype=torch.float32))

        # 12 transformer layers with fused QKV
        self.layers = nn.ModuleList(
            [_FusedW2VLayer(layer, int(config.num_attention_heads), self.dtype, use_fp8=self.use_fp8) for layer in enc.layers]
        ).to(self._device)

        # lm_head stays in fp32 for exact log_softmax probabilities
        self.lm_head = base.lm_head.to(self._device, dtype=torch.float32)
        self.register_buffer("columns", torch.tensor(columns, device=self._device, dtype=torch.long))

        self._compiled_rest_convs = None
        self._compiled_padded_enc = None
        if is_cuda and compile_encoder and _can_compile():
            self._compiled_rest_convs = torch.compile(self._run_rest_convs, dynamic=True)
            if not self.use_varlen_fa2:
                self._compiled_padded_enc = torch.compile(self._run_padded_layers, dynamic=True)

    def _masked_groupnorm_gelu(self, hs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # hs: [B, C, T] (fp16 or fp32), valid: [B] CPU int64 -> [B, C, T] self.dtype
        B, C, T = hs.shape
        if hs.is_cuda and _HAS_TRITON:
            valid_i32 = valid.to(torch.int32).pin_memory().to(hs.device, non_blocking=True)
            out = torch.empty((B, C, T), device=hs.device, dtype=self.dtype)
            _masked_groupnorm_gelu_kernel[(B, C)](
                hs.contiguous(),
                valid_i32,
                self.c0_gn_w,
                self.c0_gn_b,
                out,
                C,
                T,
                self.c0_eps,
                BLOCK_T=1024,
                OUT_FP16=(self.dtype == torch.float16),
                num_warps=4,
            )
            return out
        hs_f = hs.float()
        valid_dev = valid.to(hs.device, non_blocking=hs.is_cuda)
        own = (torch.arange(T, device=hs.device)[None, :] < valid_dev[:, None])[:, None, :].float()
        count = own.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (hs_f * own).sum(-1, keepdim=True) / count
        var = ((hs_f - mean).square() * own).sum(-1, keepdim=True) / count
        hs_f = (hs_f - mean) / torch.sqrt(var + self.c0_eps) * self.c0_gn_w[:, None] + self.c0_gn_b[:, None]
        hs_f = F.gelu(hs_f) * own
        return hs_f.to(self.dtype)

    def _run_rest_convs(self, hs: torch.Tensor) -> torch.Tensor:
        if self._use_nhwc_convs:
            h4 = hs.unsqueeze(2).to(memory_format=torch.channels_last)
            for idx, stride in enumerate(self._rest_strides):
                w = getattr(self, f"rc_w_{idx}")
                b = getattr(self, f"rc_b_{idx}")
                h4 = F.gelu(F.conv2d(h4, w, b, stride=(1, stride)))
            return h4.squeeze(2).transpose(1, 2)  # contiguous [B, T_50hz, 512]
        assert self.rest_convs is not None
        for layer in self.rest_convs:
            hs = layer(hs)
        return hs.transpose(1, 2)

    def _run_padded_layers(self, hs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if hs.is_cuda and self.dtype == torch.float16:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                for layer in self.layers:
                    hs = layer.forward_padded(hs, attn_mask)
            return hs
        for layer in self.layers:
            hs = layer.forward_padded(hs, attn_mask)
        return hs

    @torch.no_grad()
    def forward(self, batch: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, list[int], bool]:
        """Run wav2vec2 CTC on right-padded `[B, 1, T]` normalized 16 kHz audio (`lengths`: CPU `[B]`).

        Returns `(log_probs, frame_counts, is_packed)`:
          - if `is_packed` is True (varlen FA2 path): `log_probs` is `[sum(frame_counts), V_sel]` fp32;
          - if `is_packed` is False (SDPA path): `log_probs` is `[B, max(frame_counts), V_sel]` fp32.
        """
        hs = self.c0_conv(batch.to(self.c0_dtype))  # [B, 512, T_c0]
        valid = (lengths - self.c0_k) // self.c0_s + 1
        hs = self._masked_groupnorm_gelu(hs, valid)

        if self._compiled_rest_convs is not None:
            hs = self._compiled_rest_convs(hs)
        else:
            hs = self._run_rest_convs(hs)  # [B, T_50hz, 512]

        # Feature projection
        hs = F.layer_norm(hs.float(), (hs.shape[-1],), self.fp_ln_w, self.fp_ln_b, self.fp_ln_eps).to(self.dtype)
        hs = self.fp_proj(hs)  # [B, T_50hz, 768]

        frames = self._get_feat_extract_output_lengths(lengths)
        frame_list = frames.tolist()
        frames_dev = (
            frames.pin_memory().to(hs.device, non_blocking=True)
            if hs.is_cuda
            else frames.to(hs.device)
        )
        mask = torch.arange(hs.shape[1], device=hs.device)[None, :] < frames_dev[:, None]  # [B, T]
        hs = torch.where(mask[..., None], hs, hs.new_zeros(()))
        hs = hs + self.pos_conv_embed(hs)
        hs = F.layer_norm(hs.float(), (hs.shape[-1],), self.enc_ln_w, self.enc_ln_b, self.enc_ln_eps).to(self.dtype)

        if self.use_varlen_fa2:
            # Pack valid frames only: zero padding waste in all 12 layers + lm_head + D2H!
            hs_packed = hs[mask].contiguous()  # [total_valid, 768]
            cu = torch.zeros(len(frame_list) + 1, dtype=torch.int32)
            torch.cumsum(frames.to(torch.int32), dim=0, out=cu[1:])
            cu_dev = cu.pin_memory().to(hs.device, non_blocking=True)
            max_seqlen = max(frame_list)
            for layer in self.layers:
                hs_packed = layer.forward_packed(hs_packed, cu_dev, max_seqlen)
            log_probs = F.log_softmax(self.lm_head(hs_packed.float()), dim=-1).index_select(-1, self.columns)
            return log_probs, frame_list, True

        # Additive float mask [B, 1, 1, T] enables cuDNN FlashAttention (2.7x faster than bool mask SDPA)
        attn_mask = torch.where(mask[:, None, None, :], hs.new_zeros(()), hs.new_full((), float("-inf")))
        if self._compiled_padded_enc is not None:
            hs = self._compiled_padded_enc(hs, attn_mask)
        else:
            hs = self._run_padded_layers(hs, attn_mask)
        log_probs = F.log_softmax(self.lm_head(hs.float()), dim=-1).index_select(-1, self.columns)
        return log_probs, frame_list, False
