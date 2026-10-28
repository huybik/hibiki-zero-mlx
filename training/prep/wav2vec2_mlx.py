"""wav2vec2-base-960h CTC in MLX over length buckets, for English word timing on Apple Silicon.

The base model has no attention mask and normalizes its first convolution over the whole input,
so zero padding alone changes its output. This port pads each clip to a multiple of
BUCKET_SAMPLES and masks the padding instead: the first GroupNorm's statistics, the positional
convolution's input and the attention keys see only the clip's own frames, and the unpadded
convolutions never reach it. A padded clip therefore gives the log-probabilities of the clip
alone, up to float rounding, and the compiled forward sees at most ~18 shapes
(docs/agent-memory/mlx-inference.md#memory). Against mlx-audio 0.6.1's eager model it gives the
same cache rows on 1,400 of 1,400 rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

KERNELS = (10, 3, 3, 3, 3, 2, 2)  # feature-encoder convolutions, 16 kHz -> 50 Hz
STRIDES = (5, 2, 2, 2, 2, 2, 2)
BUCKET_SAMPLES = 20_480  # 1.28 s at 16 kHz
POS_KERNEL = 128
POS_GROUPS = 16
NUM_HEADS = 12
NUM_LAYERS = 12
EPS = 1e-5


def _frames(samples: int) -> tuple[int, int]:
    """Frames after the first convolution and after the last, for an unpadded clip."""
    counts = []
    for kernel, stride in zip(KERNELS, STRIDES, strict=True):
        samples = (samples - kernel) // stride + 1
        counts.append(samples)
    return counts[0], counts[-1]


class Wav2Vec2CTC:
    """16 kHz float32 PCM -> CTC log-probabilities [frames, vocab] from the HF checkpoint."""

    def __init__(self, checkpoint: str | Path):
        w = {k.removeprefix("wav2vec2."): v for k, v in mx.load(str(checkpoint)).items()}
        conv = "feature_extractor.conv_layers"
        # PyTorch conv weight (out, in, k) -> MLX layout (out, k, in); the convolutions have no bias.
        self.convs = [w[f"{conv}.{i}.conv.weight"].swapaxes(1, 2) for i in range(len(KERNELS))]
        self.group_norm = w[f"{conv}.0.layer_norm.weight"], w[f"{conv}.0.layer_norm.bias"]
        self.projection_norm = (
            w["feature_projection.layer_norm.weight"],
            w["feature_projection.layer_norm.bias"],
        )
        self.projection = (
            w["feature_projection.projection.weight"].T,
            w["feature_projection.projection.bias"],
        )
        pos = "encoder.pos_conv_embed.conv"
        g, v = w[f"{pos}.weight_g"], w[f"{pos}.weight_v"]  # weight norm over the kernel axis
        self.pos_conv = (
            (g * v / mx.sqrt((v * v).sum(axis=(0, 1), keepdims=True))).swapaxes(1, 2),
            w[f"{pos}.bias"],
        )
        self.encoder_norm = w["encoder.layer_norm.weight"], w["encoder.layer_norm.bias"]
        self.layers = []
        for i in range(NUM_LAYERS):
            layer = f"encoder.layers.{i}"
            attn = f"{layer}.attention"
            ff = f"{layer}.feed_forward"
            self.layers.append(
                {
                    "qkv": mx.concatenate([w[f"{attn}.{n}_proj.weight"] for n in "qkv"]).T,
                    "qkv_bias": mx.concatenate([w[f"{attn}.{n}_proj.bias"] for n in "qkv"]),
                    "out": (w[f"{attn}.out_proj.weight"].T, w[f"{attn}.out_proj.bias"]),
                    "norm": (w[f"{layer}.layer_norm.weight"], w[f"{layer}.layer_norm.bias"]),
                    "ff1": (
                        w[f"{ff}.intermediate_dense.weight"].T,
                        w[f"{ff}.intermediate_dense.bias"],
                    ),
                    "ff2": (w[f"{ff}.output_dense.weight"].T, w[f"{ff}.output_dense.bias"]),
                    "final_norm": (
                        w[f"{layer}.final_layer_norm.weight"],
                        w[f"{layer}.final_layer_norm.bias"],
                    ),
                }
            )
        self.head = w["lm_head.weight"].T, w["lm_head.bias"]
        mx.eval(self.__dict__)
        self._forward_compiled = mx.compile(self._forward)

    def _forward(self, pcm: mx.array, first_frames: mx.array, frames: mx.array) -> mx.array:
        hs = mx.conv1d(pcm[..., None], self.convs[0], stride=STRIDES[0])
        # GroupNorm with one group per channel, over the clip's own frames only.
        own = (mx.arange(hs.shape[1]) < first_frames)[None, :, None]
        count = first_frames.astype(mx.float32)
        mean = mx.where(own, hs, 0).sum(axis=1, keepdims=True) / count
        var = mx.where(own, mx.square(hs - mean), 0).sum(axis=1, keepdims=True) / count
        hs = nn.gelu((hs - mean) * mx.rsqrt(var + EPS) * self.group_norm[0] + self.group_norm[1])
        for weight, stride in zip(self.convs[1:], STRIDES[1:], strict=True):
            hs = nn.gelu(mx.conv1d(hs, weight, stride=stride))
        hs = mx.fast.layer_norm(hs, *self.projection_norm, EPS) @ self.projection[0]
        hs = hs + self.projection[1]
        # The positional convolution zero-pads the clip, so the padding frames enter it as zeros.
        valid = mx.arange(hs.shape[1]) < frames
        hs = mx.where(valid[None, :, None], hs, 0)
        pos = mx.conv1d(hs, self.pos_conv[0], padding=POS_KERNEL // 2, groups=POS_GROUPS)
        hs = hs + nn.gelu(pos[:, :-1] + self.pos_conv[1])
        hs = mx.fast.layer_norm(hs, *self.encoder_norm, EPS)
        batch, steps, dim = hs.shape
        head = dim // NUM_HEADS
        mask = valid[None, None, None, :]
        for p in self.layers:
            qkv = hs @ p["qkv"] + p["qkv_bias"]
            q, k, v = (
                part.reshape(batch, steps, NUM_HEADS, head).transpose(0, 2, 1, 3)
                for part in (
                    qkv[..., :dim] * head**-0.5,
                    qkv[..., dim : 2 * dim],
                    qkv[..., 2 * dim :],
                )
            )
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, steps, dim) @ p["out"][0] + p["out"][1]
            hs = mx.fast.layer_norm(hs + attn, *p["norm"], EPS)
            ff = nn.gelu(hs @ p["ff1"][0] + p["ff1"][1]) @ p["ff2"][0] + p["ff2"][1]
            hs = mx.fast.layer_norm(hs + ff, *p["final_norm"], EPS)
        logits = hs @ self.head[0] + self.head[1]
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def log_probs(self, pcm: np.ndarray) -> mx.array:
        """Lazy CTC log-probabilities [frames, vocab] of one mono 16 kHz clip."""
        if pcm.ndim != 1 or len(pcm) < KERNELS[0]:
            raise ValueError(f"wav2vec2 needs a mono clip of at least {KERNELS[0]} samples")
        # The feature extractor's zero-mean, unit-variance input normalization, over the clip.
        pcm = np.asarray(pcm, np.float32)
        pcm = (pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)
        first_frames, frames = _frames(len(pcm))
        padded = np.zeros(math.ceil(len(pcm) / BUCKET_SAMPLES) * BUCKET_SAMPLES, np.float32)
        padded[: len(pcm)] = pcm
        out = self._forward_compiled(
            mx.array(padded[None]), mx.array(first_frames), mx.array(frames)
        )
        return out[0, :frames]
