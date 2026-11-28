"""Mimi encoder in MLX for whole-clip batch encoding on Apple Silicon.

Encode-only port of moshi 0.2.13's PyTorch Mimi (the cache reference) that loads its checkpoint
directly. Activations stay in MLX's native (batch, time, channels) layout, so no convolution
transposes its input; the strided SEANet convolutions (kernel = 2 * stride) run as one matmul each,
and the residual blocks of the first three SEANet stages as one Metal kernel each (`_residual`).
The encoder transformer is causal over a 250-frame window with exact GELU, as in PyTorch; upstream
moshi_mlx 0.3.0 omits the mask and uses the tanh GELU, which gives different codes on whole clips.

Codes match PyTorch CPU float32 on 99.96-99.98% of frames, the first codebook on all but rare
near-ties; the rest are float near-ties in the residual codebooks. Keep float32: half precision
changes 11-55% of frames.
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
# SEANet stages whose residual block runs as one Metal kernel (`_residual`), with the 8-step row
# blocks each simdgroup takes: (24 kHz, 64 channels), (6 kHz, 128), (1.2 kHz, 256). The 512-channel
# stage gained nothing and stays on MLX ops.
FUSED_ROWS = (4, 4, 2)
# Clips encoded at once, each call on the next of these GPU streams: on one clip the 25 Hz stages'
# small matmuls leave much of the GPU idle (+2-7% on Mimi alone, none in the en-vi build beside the
# CTC). Codes are those of one stream; each stream compiles its own traces of the bucketed shapes.
STREAMS = 4

# elu(x + W2 elu(conv3(elu(x)) + b1) + b2) for 8 * RB time steps a simdgroup, on 8x8 simdgroup
# matrices: the hidden activations stay in registers, the causal left padding is a bounds check.
# Fragment layout as MLX's steel kernels (BaseMMAFrag::get_coord): lane holds row fm, columns fn
# and fn + 1.
_RESIDUAL_HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
inline float2 elu2(float2 v) { return select(precise::exp(v) - 1.0f, v, v > 0.0f); }
"""
_RESIDUAL_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const int T = x_shape[1];
    const int t0 = int(thread_position_in_grid.y) * 8 * RB;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const device float* xb = x + size_t(thread_position_in_grid.z) * T * C;
    device float* yb = y + size_t(thread_position_in_grid.z) * T * C;

    simdgroup_float8x8 mid[RB][H / 8];
    _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++)
        _Pragma("clang loop unroll(full)") for (int ob = 0; ob < H / 8; ob++)
            mid[rb][ob] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < 3; k++) {
        for (int cb = 0; cb < C / 8; cb++) {
            simdgroup_float8x8 a[RB];
            _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++) {
                const int row = t0 + rb * 8 + fm - 2 + k;
                const float2 v = row >= 0 ? *(const device float2*)(xb + size_t(row) * C + cb * 8 + fn)
                                          : float2(0.0f);
                reinterpret_cast<thread float2&>(a[rb].thread_elements()) = elu2(v);
            }
            _Pragma("clang loop unroll(full)") for (int ob = 0; ob < H / 8; ob++) {
                simdgroup_float8x8 w;
                reinterpret_cast<thread float2&>(w.thread_elements()) =
                    *(const device float2*)(w1 + size_t(k * C + cb * 8 + fm) * H + ob * 8 + fn);
                _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++)
                    simdgroup_multiply_accumulate(mid[rb][ob], a[rb], w, mid[rb][ob]);
            }
        }
    }
    _Pragma("clang loop unroll(full)") for (int ob = 0; ob < H / 8; ob++) {
        const float2 bias = *(const device float2*)(b1 + ob * 8 + fn);
        _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++) {
            thread float2& e = reinterpret_cast<thread float2&>(mid[rb][ob].thread_elements());
            e = elu2(e + bias);
        }
    }
    for (int oc = 0; oc < C / 8; oc++) {
        simdgroup_float8x8 acc[RB];
        _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++)
            acc[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)") for (int kb = 0; kb < H / 8; kb++) {
            simdgroup_float8x8 w;
            reinterpret_cast<thread float2&>(w.thread_elements()) =
                *(const device float2*)(w2 + size_t(kb * 8 + fm) * C + oc * 8 + fn);
            _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++)
                simdgroup_multiply_accumulate(acc[rb], mid[rb][kb], w, acc[rb]);
        }
        const float2 bias = *(const device float2*)(b2 + oc * 8 + fn);
        _Pragma("clang loop unroll(full)") for (int rb = 0; rb < RB; rb++) {
            const size_t at = size_t(t0 + rb * 8 + fm) * C + oc * 8 + fn;
            *(device float2*)(yb + at) = elu2(reinterpret_cast<thread float2&>(acc[rb].thread_elements())
                                              + bias + *(const device float2*)(xb + at));
        }
    }
"""
_residual_kernel = mx.fast.metal_kernel(
    name="seanet_residual",
    input_names=["x", "w1", "b1", "w2", "b2"],
    output_names=["y"],
    source=_RESIDUAL_SOURCE,
    header=_RESIDUAL_HEADER,
)


def _conv(weights: dict[str, mx.array], name: str) -> tuple[mx.array, mx.array | None]:
    """PyTorch conv weight (out, in, k) -> MLX layout (out, k, in), with its bias if any."""
    return weights[f"{name}.weight"].swapaxes(1, 2), weights.get(f"{name}.bias")


def _causal_conv(xs: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    ksize = weight.shape[1]
    if ksize == 1:
        ys = xs @ weight[:, 0, :].T
    else:
        ys = mx.conv1d(mx.pad(xs, [(0, 0), (ksize - 1, 0), (0, 0)]), weight)
    return ys if bias is None else ys + bias


def _downsample(xs: mx.array, both: mx.array, bias: mx.array, stride: int) -> mx.array:
    """Causal conv of kernel 2 * stride and stride `stride`. Its left padding is one block of
    `stride` zeros, so output t sees input blocks t-1 and t: one matmul against both kernel halves
    (`both`, (stride * in, 2 * out), the first `out` columns the taps on block t-1), then the
    previous block's half added."""
    batch, steps, channels = xs.shape
    ys = xs.reshape(batch, steps // stride, stride * channels) @ both
    out = both.shape[1] // 2
    return mx.pad(ys[:, :-1, :out], [(0, 0), (1, 0), (0, 0)]) + ys[:, :, out:] + bias


def _residual(
    xs: mx.array, w1: mx.array, b1: mx.array, w2: mx.array, b2: mx.array, rows: int
) -> mx.array:
    """elu(xs + residual block(xs)) by `_residual_kernel`; w1 is the kernel-3 conv as
    (3 * channels, hidden), row k * channels + c, and w2 the kernel-1 conv as (hidden, channels)."""
    batch, steps, channels = xs.shape
    if steps % (8 * rows):
        raise ValueError(f"{steps} steps are not a multiple of {8 * rows}")
    return _residual_kernel(
        inputs=[xs, w1, b1, w2, b2],
        template=[("C", channels), ("H", w1.shape[1]), ("RB", rows)],
        grid=(32, steps // (8 * rows), batch),
        threadgroup=(32, 2, 1),
        output_shapes=[xs.shape],
        output_dtypes=[mx.float32],
    )[0]


class MimiEncoder:
    """Whole-clip Mimi encoder: 24 kHz float32 PCM -> int32 codes, `num_codebooks` x frames.
    Build and call it on one thread: its GPU streams exist only on the thread that made them
    (MLX 0.32.4), so `encode` from another thread raises."""

    def __init__(self, checkpoint: str | Path, num_codebooks: int = 16):
        w = mx.load(str(checkpoint))
        self.num_codebooks = num_codebooks
        self.init = _conv(w, "encoder.model.0.conv.conv")
        self.stages = []
        for i, ((res, down), stride) in enumerate(zip(SEANET_INDICES, RATIOS, strict=True)):
            (w1, b1), (w2, b2) = (
                _conv(w, f"encoder.model.{res}.block.{j}.conv.conv") for j in (1, 3)
            )
            if i < len(FUSED_ROWS):
                # `_residual`'s layout: (out, 3, in) -> (3 * in, out), (out, 1, in) -> (in, out).
                residual = (
                    mx.contiguous(w1.transpose(1, 2, 0).reshape(-1, w1.shape[0])),
                    b1,
                    mx.contiguous(w2[:, 0, :].T),
                    b2,
                    FUSED_ROWS[i],
                )
            else:
                residual = ((w1, b1), (w2, b2))
            wd, bd = _conv(w, f"encoder.model.{down}.conv.conv")
            halves = wd.reshape(wd.shape[0], 2, -1)
            both = mx.concatenate([halves[:, 0].T, halves[:, 1].T], axis=1)
            self.stages.append((residual, both, bd, stride))
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
        self.streams = [mx.new_stream(mx.gpu) for _ in range(STREAMS)]
        self.calls = 0

    def _seanet(self, xs: mx.array) -> mx.array:
        hs = _causal_conv(xs, *self.init)
        for i, (residual, both, bias, stride) in enumerate(self.stages):
            if i < len(FUSED_ROWS):
                hs = _residual(hs, *residual)
            else:
                (w1, b1), (w2, b2) = residual
                hs = nn.elu(hs + _causal_conv(nn.elu(_causal_conv(nn.elu(hs), w1, b1)), w2, b2))
            hs = _downsample(hs, both, bias, stride)
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

    def encode(self, pcm: np.ndarray, delay_samples: int = 0) -> mx.array:
        """Encode one mono 24 kHz clip as a whole (optionally left-padded with `delay_samples`
        zeros); returns lazy int32 codes, ceil((len + delay_samples) / 1920) frames, queued on the
        next of the encoder's streams (evaluate several calls together)."""
        if pcm.ndim != 1 or not len(pcm):
            raise ValueError(f"Mimi needs a non-empty mono clip, got shape {pcm.shape}")
        total = delay_samples + len(pcm)
        frames = math.ceil(total / FRAME_SAMPLES)
        bucket = BUCKET_FRAMES * FRAME_SAMPLES
        padded = np.zeros(math.ceil(total / bucket) * bucket, np.float32)
        padded[delay_samples:total] = pcm
        self.calls += 1
        with mx.stream(self.streams[self.calls % len(self.streams)]):
            return self._encode(mx.array(padded[None]))[0, :, :frames]
