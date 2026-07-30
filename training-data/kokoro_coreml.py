"""Kokoro iSTFTNet decoder as a Core ML mlprogram on the Metal GPU.

The decoder conv stack is ~94% of EN wall time and runs 12.7x faster through
Core ML on the M-series GPU than as torch.compile'd CPU (73 vs 927 ms for 6 s
of audio; ANE measured and rejected: the E5 compiler takes 26 min and emits a
CPU-bound program). The random sine source module, its har STFT, and the tiny
20-point iSTFT stay in torch on CPU: random/complex ops don't convert, they're
<3% of the time, and keeping them out makes the converted graph deterministic —
the fp32 split is bit-exact vs the original decoder, so all Core ML error is
fp16 rounding (audio corr 0.99999).

Shapes are flexible (RangeDim on T = asr frames), but every first-seen shape
pays ~600 ms of Metal specialization, so T is zero-padded up to a multiple of
32 and the audio trimmed back: a worker sees ~20 shapes per run, all warm after
the first minute. Padding is made exact, not approximate: AdaIN's InstanceNorm
normalizes over the time axis, so the graph is traced with a MASKED variant
that (a) reduces with 1/n-prescaled prefix masks passed as inputs (summands
and outputs stay at mean/variance scale, fp16-safe with no fp32 pinning) and
(b) re-zeros the pad tail at every output — each conv then sees exactly the
zeros the unpadded graph's implicit conv padding provides, so the valid region
reproduces the unpadded computation bit-for-fp32-bit (naive padding
decorrelates to ~0.9 corr; torch A/B of this scheme gives corr 1.000000).
The .mlpackage converts once and is cached on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HAR_PER_FRAME = 120  # har STFT frames per asr frame (decode 2x, rates 10*6, hop 5)
SAMPLES_PER_FRAME = 600
MAX_T = 2048         # 51 s of audio; Kokoro chunks are <=510 tokens, well under
T_BUCKET = 32        # pad T up to a multiple of this to bound the Metal shape set

CACHE_DIR = Path(
    os.environ.get("KOKORO_COREML_CACHE", Path.home() / ".cache" / "kokoro-coreml")
)
MODEL_PATH = CACHE_DIR / "kokoro-decoder-core-v2.mlpackage"


# Per-call masked-norm context: trace-time dict {stage length -> (mask, inv_n)}.
# None outside DecoderCore.forward, so the patched AdaIN1d falls back to the
# stock InstanceNorm for every other user of the class (e.g. the CPU path and
# the prosody predictor, which run unpadded).
_MASKS: list = [None]


def patch_masked_adain() -> None:
    import kokoro.istftnet as istftnet

    stock = istftnet.AdaIN1d.forward

    def forward(self, x, s):
        ctx = _MASKS[0]
        if ctx is None:
            return stock(self, x, s)
        mask, wmask = ctx[int(x.shape[-1])]
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # wmask = mask/n keeps every summand and both reduction outputs at
        # mean/variance scale, so the whole norm stays fp16-safe without
        # pinning ops to fp32 (a raw variance sum over the 120T stage would
        # overflow fp16's 65504).
        mean = (x * wmask).sum(dim=-1, keepdim=True)
        d = x - mean
        var = (d * d * wmask).sum(dim=-1, keepdim=True)
        # eps 1e-4, not InstanceNorm's 1e-5: 1e-5 is subnormal in fp16 and
        # Metal flushes it to zero, so a near-constant channel on a long chunk
        # (var summands below the subnormal floor -> var == 0) hits
        # rsqrt(0)=inf -> 0*inf=NaN and the global norms spread it everywhere.
        # Only near-dead channels see the changed eps; their output is ~0.
        xn = d * torch.rsqrt(var + 1e-4)
        xn = xn * self.norm.weight.view(1, -1, 1) + self.norm.bias.view(1, -1, 1)
        # Re-zero the pad tail: every conv then sees the same zeros the
        # unpadded graph's implicit conv padding provides, keeping the valid
        # region exact instead of letting boundary bleed grow layer by layer.
        return ((1 + gamma) * xn + beta) * mask

    istftnet.AdaIN1d.forward = forward

    # The upsampling decode block's transposed-conv pool spreads valid content
    # into the pad region; re-zero it so conv1's valid-edge taps read the
    # zeros the unpadded graph implies.
    def residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        ctx = _MASKS[0]
        if ctx is not None and not isinstance(self.pool, torch.nn.Identity):
            x = x * ctx[int(x.shape[-1])][0]
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    istftnet.AdainResBlk1d._residual = residual

    # Snake1D uses sin(a*x)**2; write it as a multiply (identical math, no
    # `pow` op over the audio-rate tensors).
    def resblock_forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            sn = torch.sin(a1 * xt)
            xt = xt + (1 / a1) * (sn * sn)
            xt = c1(xt)
            xt = n2(xt, s)
            sn = torch.sin(a2 * xt)
            xt = xt + (1 / a2) * (sn * sn)
            xt = c2(xt)
            x = xt + x
        return x

    istftnet.AdaINResBlock1.forward = resblock_forward


class DecoderCore(torch.nn.Module):
    """Decoder.forward minus the sine source (har is an input) and the iSTFT."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, asr, F0_curve, N, s, har, mask_t, inv_n):
        # Stage masks derived from the frame-rate mask; nearest-upsample of a
        # prefix-ones mask is exact for integer factors. Keyed by trace-time
        # length; at runtime all lengths scale together with T.
        m2 = F.interpolate(mask_t, scale_factor=2, mode="nearest")
        m3 = F.interpolate(m2, scale_factor=10, mode="nearest")
        m3b = F.interpolate(m3, scale_factor=6, mode="nearest")
        m4 = F.pad(m3b, (1, 0), value=1.0)
        _MASKS[0] = {
            int(m.shape[-1]): (m, m * w)
            for m, w in ((mask_t, inv_n[0]), (m2, inv_n[1]), (m3, inv_n[2]),
                         (m3b, inv_n[3]), (m4, inv_n[3]))
        }
        try:
            return self._forward(asr, F0_curve, N, s, har)
        finally:
            _MASKS[0] = None

    def _forward(self, asr, F0_curve, N, s, har):
        d = self.dec
        masks = _MASKS[0]
        F0 = d.F0_conv(F0_curve.unsqueeze(1))
        Nc = d.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, Nc], dim=1)
        x = d.encode(x, s)
        asr_res = d.asr_res(asr)
        res = True
        for block in d.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nc], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        g = d.generator
        # zero har's pad tail: the strided noise_convs otherwise read content
        # where the unpadded graph reads implicit zeros
        har = har * masks[int(har.shape[-1])][0]
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = g.noise_convs[i](har)
            x_source = g.noise_res[i](x_source, s)
            x_source = x_source * masks[int(x_source.shape[-1])][0]
            # zero pad tails around the transposed conv: it must see (and
            # leave) the zeros the unpadded graph's out-of-range inputs imply
            x = g.ups[i](x * masks[int(x.shape[-1])][0])
            x = x * masks[int(x.shape[-1])][0]
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(g.num_kernels):
                blk = g.resblocks[i * g.num_kernels + j]
                xs = blk(x, s) if xs is None else xs + blk(x, s)
            x = xs / g.num_kernels
        x = F.leaky_relu(x)
        x = g.conv_post(x * masks[int(x.shape[-1])][0])
        spec = torch.exp(x[:, : g.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, g.post_n_fft // 2 + 1 :, :])
        return spec, phase


def fold_weight_norm(module: torch.nn.Module) -> None:
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    for mod in module.modules():
        if is_parametrized(mod, "weight"):
            remove_parametrizations(mod, "weight")
        else:
            try:
                remove_weight_norm(mod)
            except ValueError:
                pass


def patch_rsqrt() -> None:
    """torch.rsqrt(torch.tensor(2)) traces as int32 rsqrt, which MIL rejects."""
    import kokoro.istftnet as istftnet

    def _forward(self, x, s):
        out = self._residual(x, s)
        return (out + self._shortcut(x)) * 0.7071067811865476

    istftnet.AdainResBlk1d.forward = _forward


def make_har(generator, F0_curve: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        f0 = generator.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source)
        return torch.cat([har_spec, har_phase], dim=1)


def convert(decoder):
    """Trace + convert the (weight-norm-folded) decoder; cache the mlpackage."""
    import coremltools as ct

    patch_rsqrt()
    patch_masked_adain()
    core = DecoderCore(decoder).eval()
    T, t = 240, 230  # trace with real padding so the mask ops are exercised
    torch.manual_seed(0)
    mask_t = torch.zeros(1, 1, T)
    mask_t[..., :t] = 1.0
    inv_n = torch.tensor(
        [1 / t, 1 / (2 * t), 1 / (20 * t), 1 / (HAR_PER_FRAME * t + 1)],
        dtype=torch.float32,
    )
    example = (
        torch.randn(1, 512, T) * 0.5,
        torch.rand(1, 2 * T) * 200 + 50,
        torch.rand(1, 2 * T) * 0.5,
        torch.randn(1, 128) * 0.3,
        torch.randn(1, 22, HAR_PER_FRAME * T + 1) * 0.5,
        mask_t,
        inv_n,
    )
    with torch.no_grad():
        traced = torch.jit.trace(core, example)

    rd = lambda hi, default: ct.RangeDim(lower_bound=8, upper_bound=hi, default=default)
    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="asr", shape=(1, 512, rd(MAX_T, T))),
            ct.TensorType(name="F0_curve", shape=(1, rd(2 * MAX_T, 2 * T))),
            ct.TensorType(name="N", shape=(1, rd(2 * MAX_T, 2 * T))),
            ct.TensorType(name="s", shape=(1, 128)),
            ct.TensorType(
                name="har",
                shape=(1, 22, rd(HAR_PER_FRAME * MAX_T + 1, HAR_PER_FRAME * T + 1)),
            ),
            ct.TensorType(name="mask_t", shape=(1, 1, rd(MAX_T, T))),
            ct.TensorType(name="inv_n", shape=(4,)),
        ],
        outputs=[ct.TensorType(name="spec"), ct.TensorType(name="phase")],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        convert_to="mlprogram",
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mlm.save(str(MODEL_PATH))


class CoreMLDecoder(torch.nn.Module):
    """Drop-in for KModel.decoder: (asr, F0_curve, N, s) -> audio, GPU-backed."""

    def __init__(self, decoder):
        import coremltools as ct

        super().__init__()
        self.generator = decoder.generator
        if not MODEL_PATH.exists():
            fold_weight_norm(decoder)
            convert(decoder)
        self.mlm = ct.models.MLModel(
            str(MODEL_PATH), compute_units=ct.ComputeUnit.CPU_AND_GPU
        )
        self._cpu_mlm = None

    def forward(self, asr, F0_curve, N, s):
        t = asr.shape[-1]
        pad = -t % T_BUCKET
        # har from the UNPADDED F0 (then zero-padded): its tail STFT frames and
        # its RNG draws exactly match the stock CPU decoder's
        har = make_har(self.generator, F0_curve)
        if pad:
            asr = F.pad(asr, (0, pad))
            F0_curve = F.pad(F0_curve, (0, 2 * pad))
            N = F.pad(N, (0, 2 * pad))
            har = F.pad(har, (0, HAR_PER_FRAME * pad))
        mask_t = np.zeros((1, 1, t + pad), dtype=np.float32)
        mask_t[..., :t] = 1.0
        inv_n = np.array(
            [1 / t, 1 / (2 * t), 1 / (20 * t), 1 / (HAR_PER_FRAME * t + 1)],
            dtype=np.float32,
        )
        feed = {
            "asr": asr.numpy(),
            "F0_curve": F0_curve.numpy(),
            "N": N.numpy(),
            "s": s.numpy(),
            "har": har.numpy(),
            "mask_t": mask_t,
            "inv_n": inv_n,
        }
        audio = self._audio(self.mlm.predict(feed), t)
        if not self._plausible(audio):
            # Metal buffer-reuse bug: specific long-chunk shape/value sequences
            # (repro: T 1184->800->480->1088 with real values) poison the
            # process-wide GPU context for that feed, deterministically — even
            # a fresh MLModel instance fails, but CPU execution is immune. The
            # corruption shows as NaN OR as finite near-silence (rms ~0.011 vs
            # >=0.036 on every healthy chunk, n=200), hence the rms gate.
            # Rare (extreme-length clips), so eat the ~2.5 s CPU predict.
            audio = self._audio(self._cpu_rescue(feed), t)
            if not torch.isfinite(audio).all():
                raise RuntimeError("CoreML Kokoro decoder returned NaN on GPU and CPU")
        return audio

    def _audio(self, out, t):
        with torch.no_grad():
            audio = self.generator.stft.inverse(
                torch.from_numpy(out["spec"]), torch.from_numpy(out["phase"])
            )
        return audio[..., : t * SAMPLES_PER_FRAME]

    @staticmethod
    def _plausible(audio) -> bool:
        return bool(torch.isfinite(audio).all()) and float(audio.pow(2).mean().sqrt()) >= 0.02

    def _cpu_rescue(self, feed):
        import coremltools as ct

        if self._cpu_mlm is None:
            self._cpu_mlm = ct.models.MLModel(
                str(MODEL_PATH), compute_units=ct.ComputeUnit.CPU_ONLY
            )
        return self._cpu_mlm.predict(feed)
