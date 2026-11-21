"""wav2vec2 CTC in MLX over length buckets, for word timing on Apple Silicon: either layout a
`text_timing.ALIGNERS` model has, read from its config.json.

    base   (wav2vec2-base-960h, English): bias-free convolutions, GroupNorm on the first only, the
           encoder's layer norm before its post-norm layers
    stable (omniASR-CTC-300M-v2, Vietnamese): a LayerNorm after every convolution, pre-norm layers,
           the encoder's layer norm after them

Neither may see padding: the base model normalizes its first convolution over the whole input, and
both mix frames in the positional convolution and the attention. This port pads each clip to a
multiple of BUCKET_SAMPLES and masks the padding instead: the first GroupNorm's statistics, the
positional convolution's input and the attention keys see only the clip's own frames, and the
unpadded convolutions and per-frame LayerNorms never reach it. A padded clip therefore gives the
log-probabilities of the clip alone, up to float rounding, and the compiled forward sees at most
~18 shapes (docs/agent-memory/mlx-inference.md#memory). Against mlx-audio 0.6.1's eager model the
base layout builds the same cache rows but for float near-ties at CTC: on 2 of ~3,200 rows a word
sits 1-2 frames apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from training.prep.text_timing import CTC_CONVS, ctc_frames

KERNELS, STRIDES = zip(*CTC_CONVS, strict=True)  # feature-encoder convolutions, 16 kHz -> 50 Hz
BUCKET_SAMPLES = 20_480  # 1.28 s at 16 kHz


def _frames(samples: int) -> tuple[int, int]:
    """Frames after the first convolution and after the last, for an unpadded clip."""
    return (samples - KERNELS[0]) // STRIDES[0] + 1, ctc_frames(samples)


def _conv(weight: mx.array) -> mx.array:
    """A PyTorch conv weight (out, in, k) in MLX's layout (out, k, in)."""
    return weight.swapaxes(1, 2)


class Wav2Vec2CTC:
    """16 kHz PCM -> CTC log-probabilities [frames, vocab] from an HF Wav2Vec2ForCTC snapshot
    (`model.safetensors` and `config.json` in `files`).

    Runs in float32 by default; pass `half=True` to cast all weights to float16 and feed
    float16 input, then cast log-probs back to float32 before returning."""

    def __init__(self, files: str | Path, half: bool = False):
        files = Path(files)
        cfg = json.loads((files / "config.json").read_text())
        if tuple(zip(cfg["conv_kernel"], cfg["conv_stride"], strict=True)) != CTC_CONVS:
            raise ValueError(f"{files}: its convolutions are not text_timing.CTC_CONVS")
        if cfg.get("add_adapter") or cfg.get("position_embeddings_type") not in (None, "conv"):
            raise ValueError(f"{files}: an adapter or non-convolutional positions are not ported")
        self.stable = bool(cfg["do_stable_layer_norm"])
        self.layer_norm_convs = cfg["feat_extract_norm"] == "layer"  # else GroupNorm on the first
        self.heads = int(cfg["num_attention_heads"])
        self.pos_kernel = int(cfg["num_conv_pos_embeddings"])
        self.pos_groups = int(cfg["num_conv_pos_embedding_groups"])
        self.eps = float(cfg["layer_norm_eps"])
        w = {k.removeprefix("wav2vec2."): v for k, v in mx.load(str(files / "model.safetensors")).items()}
        conv = "feature_extractor.conv_layers"
        self.convs = [
            (_conv(w[f"{conv}.{i}.conv.weight"]), w.get(f"{conv}.{i}.conv.bias"))
            for i in range(len(KERNELS))
        ]
        norms = range(len(KERNELS)) if self.layer_norm_convs else range(1)
        self.conv_norms = [(w[f"{conv}.{i}.layer_norm.weight"], w[f"{conv}.{i}.layer_norm.bias"]) for i in norms]
        self.projection_norm = (
            w["feature_projection.layer_norm.weight"],
            w["feature_projection.layer_norm.bias"],
        )
        self.projection = (
            w["feature_projection.projection.weight"].T,
            w["feature_projection.projection.bias"],
        )
        pos = "encoder.pos_conv_embed.conv"
        # Weight norm over the kernel axis, under either name transformers saves it by.
        g = w.get(f"{pos}.weight_g", w.get(f"{pos}.parametrizations.weight.original0"))
        v = w.get(f"{pos}.weight_v", w.get(f"{pos}.parametrizations.weight.original1"))
        self.pos_conv = (_conv(g * v / mx.sqrt((v * v).sum(axis=(0, 1), keepdims=True))), w[f"{pos}.bias"])
        self.encoder_norm = w["encoder.layer_norm.weight"], w["encoder.layer_norm.bias"]
        self.layers = []
        for i in range(int(cfg["num_hidden_layers"])):
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
        mx.eval(self._weights())
        self._half = half
        if half:
            from mlx.utils import tree_map

            # The base layout's convolutions have no bias: None stays None.
            half16 = tree_map(lambda x: None if x is None else x.astype(mx.float16), self._weights())
            for k, v in half16.items():
                setattr(self, k, v)
            mx.eval(self._weights())
        self._forward_compiled = mx.compile(self._forward)
        self._stream_compiled = mx.compile(lambda *args: self._stream(*args)[0])

    def _weights(self) -> dict:
        names = ("convs", "conv_norms", "projection_norm", "projection", "pos_conv", "encoder_norm", "layers", "head")
        return {name: getattr(self, name) for name in names}

    def _features(self, pcm: mx.array, first_frames: mx.array) -> mx.array:
        """The convolutional feature encoder: [1, samples] -> [1, frames, 512]."""
        hs = pcm[..., None]
        for i, ((weight, bias), stride) in enumerate(zip(self.convs, STRIDES, strict=True)):
            hs = mx.conv1d(hs, weight, stride=stride)
            if bias is not None:
                hs = hs + bias
            if self.layer_norm_convs:  # each frame normalized alone
                hs = mx.fast.layer_norm(hs, *self.conv_norms[i], self.eps)
            elif i == 0:
                # GroupNorm with one group per channel, over the clip's own frames only.
                own = (mx.arange(hs.shape[1]) < first_frames)[None, :, None]
                # Statistics in float32: first_frames can exceed float16 max (65504) for clips
                # over 20.47 s, and the sum of squares overflows float16 for clips over ~8 s.
                count = first_frames.astype(mx.float32)
                h_fp = hs.astype(mx.float32)
                mean = mx.where(own, h_fp, 0).sum(axis=1, keepdims=True) / count
                var = mx.where(own, mx.square(h_fp - mean), 0).sum(axis=1, keepdims=True) / count
                scale = mx.rsqrt(var + self.eps).astype(hs.dtype)
                hs = (hs - mean.astype(hs.dtype)) * scale * self.conv_norms[0][0] + self.conv_norms[0][1]
            hs = nn.gelu(hs)
        return hs

    def _stream(self, pcm: mx.array, first_frames: mx.array, frames: mx.array) -> tuple[mx.array, mx.array]:
        """The transformer's input [1, frames, dim]: the projected features plus the positional
        convolution; and the mask of the clip's own frames."""
        hs = self._features(pcm, first_frames)
        hs = mx.fast.layer_norm(hs, *self.projection_norm, self.eps) @ self.projection[0]
        hs = hs + self.projection[1]
        # The positional convolution zero-pads the clip, so the padding frames enter it as zeros.
        valid = mx.arange(hs.shape[1]) < frames
        hs = mx.where(valid[None, :, None], hs, 0)
        pos = mx.conv1d(hs, self.pos_conv[0], padding=self.pos_kernel // 2, groups=self.pos_groups)
        if self.pos_kernel % 2 == 0:  # an even kernel's padding gives one frame too many
            pos = pos[:, :-1]
        return hs + nn.gelu(pos + self.pos_conv[1]), valid

    def _forward(self, pcm: mx.array, first_frames: mx.array, frames: mx.array) -> mx.array:
        hs, valid = self._stream(pcm, first_frames, frames)
        if not self.stable:
            hs = mx.fast.layer_norm(hs, *self.encoder_norm, self.eps)
        batch, steps, dim = hs.shape
        head = dim // self.heads
        mask = valid[None, None, None, :]
        for p in self.layers:
            x = mx.fast.layer_norm(hs, *p["norm"], self.eps) if self.stable else hs
            qkv = x @ p["qkv"] + p["qkv_bias"]
            q, k, v = (
                part.reshape(batch, steps, self.heads, head).transpose(0, 2, 1, 3)
                for part in (
                    qkv[..., :dim] * head**-0.5,
                    qkv[..., dim : 2 * dim],
                    qkv[..., 2 * dim :],
                )
            )
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
            attn = attn.transpose(0, 2, 1, 3).reshape(batch, steps, dim) @ p["out"][0] + p["out"][1]
            if self.stable:
                hs = hs + attn
                x = mx.fast.layer_norm(hs, *p["final_norm"], self.eps)
                hs = hs + nn.gelu(x @ p["ff1"][0] + p["ff1"][1]) @ p["ff2"][0] + p["ff2"][1]
            else:
                hs = mx.fast.layer_norm(hs + attn, *p["norm"], self.eps)
                ff = nn.gelu(hs @ p["ff1"][0] + p["ff1"][1]) @ p["ff2"][0] + p["ff2"][1]
                hs = mx.fast.layer_norm(hs + ff, *p["final_norm"], self.eps)
        if self.stable:
            hs = mx.fast.layer_norm(hs, *self.encoder_norm, self.eps)
        logits = (hs @ self.head[0] + self.head[1]).astype(mx.float32)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _inputs(self, pcm: np.ndarray) -> tuple[tuple[mx.array, mx.array, mx.array], int]:
        """A clip padded to its bucket as the compiled functions take it, and its frames."""
        if pcm.ndim != 1:
            raise ValueError(f"wav2vec2 needs a mono clip, got shape {pcm.shape}")
        first_frames, frames = _frames(len(pcm))
        if frames < 1:
            raise ValueError(f"A {len(pcm)}-sample clip is too short for one wav2vec2 frame")
        padded = np.zeros(math.ceil(len(pcm) / BUCKET_SAMPLES) * BUCKET_SAMPLES, np.float32)
        padded[: len(pcm)] = pcm
        dtype = mx.float16 if self._half else mx.float32
        return (mx.array(padded[None], dtype=dtype), mx.array(first_frames), mx.array(frames)), frames

    def log_probs(self, pcm: np.ndarray) -> mx.array:
        """Lazy CTC log-probabilities [frames, vocab] of one mono 16 kHz float32 clip, normalized
        to zero mean and unit variance as the feature extractor does (`cache_rows.prepare`).
        Always returns float32, even when the model runs in float16."""
        inputs, frames = self._inputs(pcm)
        return self._forward_compiled(*inputs)[0, :frames]  # float32: _forward casts before logsumexp

    def stream(self, pcm: np.ndarray) -> tuple[mx.array, int]:
        """Lazy transformer input [1, bucket frames, dim] of one clip (as `log_probs` takes it),
        zero-padded before the positional convolution, and the clip's frames: what
        `wav2vec2_coreml` runs the transformer on."""
        inputs, frames = self._inputs(pcm)
        return self._stream_compiled(*inputs), frames
