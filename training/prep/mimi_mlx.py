"""Mimi encoder in MLX for whole-clip batch encoding on Apple Silicon.

Encode-only port of moshi 0.2.13's PyTorch Mimi (the cache reference) that loads its checkpoint
directly. Activations stay in MLX's native (batch, time, channels) layout, so no convolution
transposes its input; the strided SEANet convolutions (kernel = 2 * stride) run as one matmul each.
The encoder transformer is causal over a 250-frame window with exact GELU, as in PyTorch; upstream
moshi_mlx 0.3.0 omits the mask and uses the tanh GELU, which gives different codes on whole clips.

Codes match PyTorch CPU float32 on 99.98% of frames with every first codebook equal; the rest are
float near-ties in the residual codebooks. Keep float32: half precision changes 11-55% of frames.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # one 12.5 Hz code frame
BUCKET_FRAMES = 16  # pad lengths to a multiple of this so the compiled encoder sees few shapes
RATIOS = (4, 5, 6, 8)  # SEANet strides, first to last (24 kHz -> 25 Hz)
SEANET_INDICES = ((1, 3), (4, 6), (7, 9), (10, 12))  # (residual block, downsample) in encoder.model
CONTEXT = 250  # encoder transformer attention window, in 25 Hz frames
NUM_HEADS = 8
CODEBOOK_EPS = 1e-5


def _conv(weights: dict[str, mx.array], name: str) -> tuple[mx.array, mx.array | None]:
    """PyTorch conv weight (out, in, k) -> MLX layout (out, k, in), with its bias if any."""
    return weights[f"{name}.weight"].swapaxes(1, 2), weights.get(f"{name}.bias")


def _causal_conv(
    xs: mx.array, weight: mx.array, bias: mx.array | None, stride: int = 1
) -> mx.array:
    ksize = weight.shape[1]
    if ksize == 1:
        ys = xs @ weight[:, 0, :].T
    elif stride > 1 and ksize == 2 * stride:
        # Left padding is one block of `stride` zeros, so output t sees input blocks t-1 and t:
        # one matmul against both kernel halves, then add the previous block's half.
        batch, steps, channels = xs.shape
        blocks = xs.reshape(batch, steps // stride, stride * channels)
        halves = weight.reshape(weight.shape[0], 2, stride * channels)
        both = blocks @ mx.concatenate([halves[:, 0].T, halves[:, 1].T], axis=1)
        out = weight.shape[0]
        ys = mx.pad(both[:, :-1, :out], [(0, 0), (1, 0), (0, 0)]) + both[:, :, out:]
    else:
        pad = ksize - stride
        ys = mx.conv1d(mx.pad(xs, [(0, 0), (pad, 0), (0, 0)]), weight, stride=stride)
    return ys if bias is None else ys + bias


class MimiEncoder:
    """Whole-clip Mimi encoder: 24 kHz float32 PCM -> int32 codes, `num_codebooks` x frames."""

    def __init__(self, checkpoint: str | Path, num_codebooks: int = 16):
        w = mx.load(str(checkpoint))
        self.num_codebooks = num_codebooks
        self.init = _conv(w, "encoder.model.0.conv.conv")
        self.layers = [
            (
                _conv(w, f"encoder.model.{res}.block.1.conv.conv"),
                _conv(w, f"encoder.model.{res}.block.3.conv.conv"),
                _conv(w, f"encoder.model.{down}.conv.conv"),
                stride,
            )
            for (res, down), stride in zip(SEANET_INDICES, RATIOS, strict=True)
        ]
        self.final = _conv(w, "encoder.model.14.conv.conv")
        prefix = "encoder_transformer.transformer.layers"
        count = 1 + max(int(k.split(".")[3]) for k in w if k.startswith(prefix))
        self.blocks = [
            {
                name: w[f"{prefix}.{i}.{name}"]
                for name in (
                    "norm1.weight",
                    "norm1.bias",
                    "norm2.weight",
                    "norm2.bias",
                    "self_attn.in_proj_weight",
                    "self_attn.out_proj.weight",
                    "linear1.weight",
                    "linear2.weight",
                    "layer_scale_1.scale",
                    "layer_scale_2.scale",
                )
            }
            for i in range(count)
        ]
        self.downsample = _conv(w, "downsample.conv.conv.conv")[0]
        self.cardinality = int(
            w["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"].shape[0]
        )
        self.quantizers = []
        for name, size in (("rvq_first", 1), ("rvq_rest", num_codebooks - 1)):
            q = f"quantizer.{name}"
            codebooks = []
            for i in range(size):
                c = f"{q}.vq.layers.{i}._codebook"
                emb = (
                    w[f"{c}.embedding_sum"]
                    / mx.maximum(w[f"{c}.cluster_usage"], CODEBOOK_EPS)[:, None]
                )
                codebooks.append((emb, (emb * emb).sum(axis=-1) / 2))
            self.quantizers.append((w[f"{q}.input_proj.weight"][:, :, 0], codebooks))
        mx.eval(self.__dict__)
        self._encode = mx.compile(self._forward)

    def _seanet(self, xs: mx.array) -> mx.array:
        hs = _causal_conv(xs, *self.init)
        for (w1, b1), (w2, b2), (wd, bd), stride in self.layers:
            hs = hs + _causal_conv(nn.elu(_causal_conv(nn.elu(hs), w1, b1)), w2, b2)
            hs = _causal_conv(nn.elu(hs), wd, bd, stride)
        return _causal_conv(nn.elu(hs), *self.final)

    def _transformer(self, xs: mx.array) -> mx.array:
        batch, steps, dim = xs.shape
        head = dim // NUM_HEADS
        delta = mx.arange(steps)[:, None] - mx.arange(steps)[None, :]
        mask = (delta >= 0) & (delta < CONTEXT)
        for p in self.blocks:
            hs = mx.fast.layer_norm(xs, p["norm1.weight"], p["norm1.bias"], 1e-5)
            qkv = (hs @ p["self_attn.in_proj_weight"].T).reshape(batch, steps, 3, NUM_HEADS, head)
            q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
            q = mx.fast.rope(q, head, traditional=True, base=10000.0, scale=1.0, offset=0)
            k = mx.fast.rope(k, head, traditional=True, base=10000.0, scale=1.0, offset=0)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=head**-0.5, mask=mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, steps, dim)
            xs = xs + p["layer_scale_1.scale"] * (attn @ p["self_attn.out_proj.weight"].T)
            hs = mx.fast.layer_norm(xs, p["norm2.weight"], p["norm2.bias"], 1e-5)
            ff = nn.gelu(hs @ p["linear1.weight"].T) @ p["linear2.weight"].T
            xs = xs + p["layer_scale_2.scale"] * ff
        return xs

    def _forward(self, pcm: mx.array) -> mx.array:
        hs = self._transformer(self._seanet(pcm[..., None]))
        # 25 Hz -> 12.5 Hz: causal kernel-4 stride-2 conv with edge (replicate) left padding.
        hs = mx.concatenate([hs[:, :1], hs[:, :1], hs], axis=1)
        hs = mx.conv1d(hs, self.downsample, stride=2)
        codes = []
        for proj, codebooks in self.quantizers:
            residual = hs @ proj.T
            for emb, half_norm in codebooks:
                index = (half_norm - residual @ emb.T).argmin(axis=-1)
                residual = residual - emb[index]
                codes.append(index)
        return mx.stack(codes, axis=1).astype(mx.int32)

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode one mono 24 kHz clip as a whole; returns int32 codes of ceil(len / 1920) frames."""
        if pcm.ndim != 1 or not len(pcm):
            raise ValueError(f"Mimi needs a non-empty mono clip, got shape {pcm.shape}")
        frames = math.ceil(len(pcm) / FRAME_SAMPLES)
        bucket = BUCKET_FRAMES * FRAME_SAMPLES
        padded = np.zeros(math.ceil(len(pcm) / bucket) * bucket, np.float32)
        padded[: len(pcm)] = pcm
        return np.array(self._encode(mx.array(padded[None]))[0, :, :frames])
