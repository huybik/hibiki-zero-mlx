"""Kaldi's log mel filterbank in PyTorch, on any device: the features of WeSpeaker's speaker scores
(`finetune/voice_scores`, Hamming windows) and of VieNeu's speaker encoder (`tranche/voice`, povey windows, as its
SDK's kaldi-native-fbank). Neither env has torchaudio or the other's runtime."""
from __future__ import annotations

import torch


def mel_banks(bins: int, fft: int, rate: int, low: float = 20.0) -> torch.Tensor:
    """Kaldi's triangular mel filters (torchaudio.compliance.kaldi.get_mel_banks), `low` to Nyquist over an rfft's
    fft // 2 + 1 bins (the last, Nyquist, unused): [bins, fft // 2 + 1]."""
    mel = lambda f: 1127.0 * torch.log1p(torch.as_tensor(f, dtype=torch.float64) / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(rate / 2)
    delta = (hi - lo) / (bins + 1)
    left = lo + torch.arange(bins, dtype=torch.float64)[:, None] * delta
    freqs = mel(rate / fft * torch.arange(fft // 2, dtype=torch.float64))[None]
    banks = torch.clamp(torch.minimum((freqs - left) / delta, (left + 2 * delta - freqs) / delta), min=0)
    return torch.nn.functional.pad(banks, (0, 1)).float()


def fbank(wav: torch.Tensor, rate: int, window: str, bins: int = 80) -> torch.Tensor:
    """Equal-length clips [B, N] -> [B, frames, bins]: 25 ms frames every 10 ms (edges snipped), DC removed,
    pre-emphasis 0.97, a `window` ("povey" or "hamming") window, the power spectrum, the mel filters, the log floored
    at float32's epsilon, less its mean over the clip. No dither."""
    n, hop = rate // 40, rate // 100
    fft = 1 << (n - 1).bit_length()
    f = wav.float().unfold(-1, n, hop)
    f = f - f.mean(-1, keepdim=True)
    f = torch.cat([f[..., :1] * (1 - 0.97), f[..., 1:] - 0.97 * f[..., :-1]], -1)
    if window == "povey":
        w = torch.hann_window(n, periodic=False, device=f.device).pow(0.85)
    elif window == "hamming":
        w = torch.hamming_window(n, periodic=False, device=f.device)
    else:
        raise ValueError(f"window {window!r}")
    power = torch.fft.rfft(f * w, n=fft).abs().pow(2)
    feats = (power @ mel_banks(bins, fft, rate).to(f.device).T).clamp(min=torch.finfo(torch.float32).eps).log()
    return feats - feats.mean(-2, keepdim=True)
