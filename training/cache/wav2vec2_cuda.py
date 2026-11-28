"""Fast CUDA wav2vec2-base CTC aligner, Blackwell-tuned, for `cuda.CUDAModels`.

Optimizations over stock `transformers.Wav2Vec2ForCTC` in eager fp32/TF32:
- FP16 compute (10-bit mantissa, matching TF32; keeps the `c0` conv, its GroupNorm statistics
  and lm_head in fp32).
- Fused two-pass Triton masked GroupNorm + GELU on `c0` output (20 GB -> ~2-3 GB HBM traffic).
- Channels-last cuDNN `conv2d` for `c1..c6`, compiled.
- `weight_norm` folded once at load on `pos_conv_embed.conv`.
- Fused QKV projection (`768 -> 2304`) per encoder layer (36 GEMMs -> 12).
- Additive float key-padding mask, so SDPA runs cuDNN attention; the 12 layers compiled.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------------- Triton kernels


@triton.jit
def _gelu_exact_f32(x):
    return 0.5 * x * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.jit
def _masked_groupnorm_gelu_kernel(
    x_ptr,      # [B, C, T] fp32
    valid_ptr,  # [B] int32
    w_ptr,      # [C] fp32
    b_ptr,      # [C] fp32
    out_ptr,    # [B, C, T] fp16
    C,
    T,
    eps,
    BLOCK_T: tl.constexpr,
):
    """Two-pass exact masked GroupNorm(num_groups=C) + affine + GELU over [0, valid[b]).

    Grid: (B, C). Uses parallel Welford/centered two-pass in fp32 registers and writes
    directly to fp16 (zeroing t >= valid[b]).
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

    # Pass 3: normalize + affine + GELU + zero padding + cast to fp16
    for t0 in range(0, T, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T)
        in_bounds = offs < T
        valid_m = offs < valid_len
        xv = tl.load(row_x + offs, mask=valid_m, other=0.0).to(tl.float32)
        y = _gelu_exact_f32((xv - mean) * scale + beta)
        y = tl.where(valid_m, y, 0.0)
        tl.store(row_out + offs, y.to(tl.float16), mask=in_bounds)


# --------------------------------------------------------------------------- Fast Wav2Vec2 CTC

DTYPE = torch.float16  # everything but the c0 conv, the norms' statistics and lm_head


def _remove_weight_norm(conv: nn.Module) -> None:
    if hasattr(nn.utils.parametrize, "is_parametrized") and nn.utils.parametrize.is_parametrized(conv, "weight"):
        nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    elif hasattr(conv, "weight_g"):
        nn.utils.remove_weight_norm(conv)


class _FusedW2VLayer(nn.Module):
    """One post-norm Wav2Vec2EncoderLayer in FP16 with a fused QKV projection."""

    def __init__(self, layer: nn.Module, num_heads: int):
        super().__init__()
        attn = layer.attention
        ff = layer.feed_forward
        self.num_heads = num_heads
        self.head_dim = attn.head_dim

        qkv_w = torch.cat([attn.q_proj.weight.detach(), attn.k_proj.weight.detach(), attn.v_proj.weight.detach()], dim=0)
        qkv_b = torch.cat([attn.q_proj.bias.detach(), attn.k_proj.bias.detach(), attn.v_proj.bias.detach()], dim=0)
        out_w, out_b = attn.out_proj.weight.detach(), attn.out_proj.bias.detach()
        fc1_w, fc1_b = ff.intermediate_dense.weight.detach(), ff.intermediate_dense.bias.detach()
        fc2_w, fc2_b = ff.output_dense.weight.detach(), ff.output_dense.bias.detach()

        dev = qkv_w.device
        self.qkv_proj = nn.Linear(qkv_w.shape[1], qkv_w.shape[0], bias=True, device=dev, dtype=DTYPE)
        self.out_proj = nn.Linear(out_w.shape[1], out_w.shape[0], bias=True, device=dev, dtype=DTYPE)
        self.fc1 = nn.Linear(fc1_w.shape[1], fc1_w.shape[0], bias=True, device=dev, dtype=DTYPE)
        self.fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0], bias=True, device=dev, dtype=DTYPE)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(qkv_w.to(DTYPE))
            self.qkv_proj.bias.copy_(qkv_b.to(DTYPE))
            self.out_proj.weight.copy_(out_w.to(DTYPE))
            self.out_proj.bias.copy_(out_b.to(DTYPE))
            self.fc1.weight.copy_(fc1_w.to(DTYPE))
            self.fc1.bias.copy_(fc1_b.to(DTYPE))
            self.fc2.weight.copy_(fc2_w.to(DTYPE))
            self.fc2.bias.copy_(fc2_b.to(DTYPE))

        self.ln1_eps = float(layer.layer_norm.eps)
        self.ln2_eps = float(layer.final_layer_norm.eps)
        self.register_buffer("ln1_w", layer.layer_norm.weight.detach().float())
        self.register_buffer("ln1_b", layer.layer_norm.bias.detach().float())
        self.register_buffer("ln2_w", layer.final_layer_norm.weight.detach().float())
        self.register_buffer("ln2_b", layer.final_layer_norm.bias.detach().float())

    def forward(self, hs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """hs: [B, T, D] fp16, attn_mask: [B, 1, 1, T] additive float mask (0 = attend, -inf = mask)."""
        B, T, D = hs.shape
        qkv = self.qkv_proj(hs)
        q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)
        hs = F.layer_norm((hs + attn_out).float(), (D,), self.ln1_w, self.ln1_b, self.ln1_eps).to(DTYPE)
        ff = self.fc2(F.gelu(self.fc1(hs)))
        return F.layer_norm((hs + ff).float(), (D,), self.ln2_w, self.ln2_b, self.ln2_eps).to(DTYPE)


class FastWav2Vec2CTC(nn.Module):
    """Blackwell-optimized wav2vec2-base CTC model for `CUDAModels`, on a CUDA device."""

    def __init__(self, files: Path, columns: list[int], device: torch.device | str = "cuda"):
        super().__init__()
        from transformers import Wav2Vec2ForCTC

        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"FastWav2Vec2CTC runs on CUDA only, not {self._device}")

        base = Wav2Vec2ForCTC.from_pretrained(files, dtype=torch.float32).to(self._device).eval()
        config = base.config
        if config.do_stable_layer_norm or config.feat_extract_norm != "group":
            raise ValueError(f"{files}: FastWav2Vec2CTC supports only the wav2vec2-base layout")
        if base.wav2vec2.adapter is not None:
            raise ValueError(f"{files} has an adapter, which FastWav2Vec2CTC does not apply")

        self._get_feat_extract_output_lengths = base._get_feat_extract_output_lengths

        # Feature extractor: c0 conv and its GroupNorm statistics in fp32
        convs = base.wav2vec2.feature_extractor.conv_layers
        self.c0_conv = convs[0].conv.to(self._device, dtype=torch.float32)
        self.c0_k = int(self.c0_conv.kernel_size[0])
        self.c0_s = int(self.c0_conv.stride[0])
        self.c0_eps = float(convs[0].layer_norm.eps)
        self.register_buffer("c0_gn_w", convs[0].layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("c0_gn_b", convs[0].layer_norm.bias.detach().to(self._device, dtype=torch.float32))
        # c1..c6 weights as 4D channels-last [C_out, C_in, 1, k] for cuDNN Tensor Core conv2d
        self._rest_strides: list[int] = []
        for idx, layer in enumerate(convs[1:]):
            w = layer.conv.weight.detach().to(self._device, dtype=DTYPE).unsqueeze(2).to(
                memory_format=torch.channels_last
            )
            self.register_buffer(f"rc_w_{idx}", w)
            b = layer.conv.bias.detach().to(self._device, dtype=DTYPE) if layer.conv.bias is not None else None
            self.register_buffer(f"rc_b_{idx}", b)
            self._rest_strides.append(int(layer.conv.stride[0]))

        # Feature projection: fp32 LayerNorm + fp16 Linear
        fp = base.wav2vec2.feature_projection
        self.fp_ln_eps = float(fp.layer_norm.eps)
        self.register_buffer("fp_ln_w", fp.layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("fp_ln_b", fp.layer_norm.bias.detach().to(self._device, dtype=torch.float32))
        self.fp_proj = fp.projection.to(self._device, dtype=DTYPE)

        # Positional conv: fold weight_norm ONCE at load time!
        enc = base.wav2vec2.encoder
        _remove_weight_norm(enc.pos_conv_embed.conv)
        self.pos_conv_embed = enc.pos_conv_embed.to(self._device, dtype=DTYPE)
        self.enc_ln_eps = float(enc.layer_norm.eps)
        self.register_buffer("enc_ln_w", enc.layer_norm.weight.detach().to(self._device, dtype=torch.float32))
        self.register_buffer("enc_ln_b", enc.layer_norm.bias.detach().to(self._device, dtype=torch.float32))

        # 12 transformer layers with fused QKV
        self.layers = nn.ModuleList(
            [_FusedW2VLayer(layer, int(config.num_attention_heads)) for layer in enc.layers]
        ).to(self._device)

        # lm_head stays in fp32 for exact log_softmax probabilities
        self.lm_head = base.lm_head.to(self._device, dtype=torch.float32)
        self.register_buffer("columns", torch.tensor(columns, device=self._device, dtype=torch.long))

        self._compiled_rest_convs = torch.compile(self._run_rest_convs, dynamic=True)
        self._compiled_padded_enc = torch.compile(self._run_padded_layers, dynamic=True)

    def _masked_groupnorm_gelu(self, hs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # hs: [B, C, T] fp32, valid: [B] CPU int64 -> [B, C, T] fp16
        B, C, T = hs.shape
        valid_i32 = valid.to(torch.int32).pin_memory().to(hs.device, non_blocking=True)
        out = torch.empty((B, C, T), device=hs.device, dtype=DTYPE)
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
            num_warps=4,
        )
        return out

    def _run_rest_convs(self, hs: torch.Tensor) -> torch.Tensor:
        h4 = hs.unsqueeze(2).to(memory_format=torch.channels_last)
        for idx, stride in enumerate(self._rest_strides):
            w = getattr(self, f"rc_w_{idx}")
            b = getattr(self, f"rc_b_{idx}")
            h4 = F.gelu(F.conv2d(h4, w, b, stride=(1, stride)))
        return h4.squeeze(2).transpose(1, 2)  # contiguous [B, T_50hz, 512]

    def _run_padded_layers(self, hs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for layer in self.layers:
                hs = layer(hs, attn_mask)
        return hs

    @torch.no_grad()
    def forward(self, batch: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Run wav2vec2 CTC on right-padded `[B, 1, T]` normalized 16 kHz audio (`lengths`: CPU `[B]`).

        Returns `(log_probs, frame_counts)`: `log_probs` is `[B, max(frame_counts), V_sel]` fp32.
        """
        hs = self.c0_conv(batch.to(torch.float32))  # [B, 512, T_c0]
        valid = (lengths - self.c0_k) // self.c0_s + 1
        hs = self._masked_groupnorm_gelu(hs, valid)
        hs = self._compiled_rest_convs(hs)  # [B, T_50hz, 512]

        # Feature projection
        hs = F.layer_norm(hs.float(), (hs.shape[-1],), self.fp_ln_w, self.fp_ln_b, self.fp_ln_eps).to(DTYPE)
        hs = self.fp_proj(hs)  # [B, T_50hz, 768]

        frames = self._get_feat_extract_output_lengths(lengths)
        frame_list = frames.tolist()
        frames_dev = frames.pin_memory().to(hs.device, non_blocking=True)
        mask = torch.arange(hs.shape[1], device=hs.device)[None, :] < frames_dev[:, None]  # [B, T]
        hs = torch.where(mask[..., None], hs, hs.new_zeros(()))
        hs = hs + self.pos_conv_embed(hs)
        hs = F.layer_norm(hs.float(), (hs.shape[-1],), self.enc_ln_w, self.enc_ln_b, self.enc_ln_eps).to(DTYPE)

        # Additive float mask [B, 1, 1, T] enables cuDNN FlashAttention (2.7x faster than bool mask SDPA)
        attn_mask = torch.where(mask[:, None, None, :], hs.new_zeros(()), hs.new_full((), float("-inf")))
        hs = self._compiled_padded_enc(hs, attn_mask)
        log_probs = F.log_softmax(self.lm_head(hs.float()), dim=-1).index_select(-1, self.columns)
        return log_probs, frame_list
