"""How much a translation's speech sounds like its speaker, and how clean it is: free-running
evaluation's voice scores (`common.generate_batch`, `inference/tools/gate.py`).

- `speaker_sim`: cosine between WeSpeaker ResNet34-LM (VoxCeleb) embeddings of the source file as
  recorded (not through the level stage, which lifts the pauses' noise) and the translation, each
  cut to its stretches of speech (`_speech`), so silence does not dilute either.
- `hnr_db`: the translation's harmonics-to-noise ratio over its loud 40 ms frames, voiced or not.
  A rough or breathy voice, or speech turning to noise, scores low; a steady buzz is periodic
  and does not.

Both are "" where the translation has under a second of speech, `speaker_sim` also where the
source has. WeSpeaker's own runtimes (its
package, ONNX Runtime) are not in the trainer's environment, so its ResNet34 and Kaldi filterbank
are written here in PyTorch and load its published checkpoint. WavLM's speaker head, which
transformers has, scored a voice the user heard as unlike theirs as high as the teacher's.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import soxr
import sphn
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F

SPEAKER_MODEL = ("Wespeaker/wespeaker-voxceleb-resnet34-LM", "f0c48c298fd835726c27956a5d617bad7115627e")
RATE = 16_000
COLUMNS = ("speaker_sim", "hnr_db")
MIN_SPEECH_S = 1.0


class _Block(nn.Module):
    """WeSpeaker's BasicBlock (wespeaker/models/resnet.py)."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class SpeakerEncoder(nn.Module):
    """WeSpeaker's ResNet34 with its checkpoint's names: blocks [3, 4, 6, 3] of 32-256 channels,
    mean and standard deviation over time (TSTP), one 256-d embedding layer (`seg_1`). The
    checkpoint's `projection`, a training head, is not loaded."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        cin = 32
        for index, (blocks, cout, stride) in enumerate(((3, 32, 1), (4, 64, 2), (6, 128, 2), (3, 256, 2)), 1):
            layers = []
            for block in range(blocks):
                layers.append(_Block(cin, cout, stride if block == 0 else 1))
                cin = cout
            setattr(self, f"layer{index}", nn.Sequential(*layers))
        self.seg_1 = nn.Linear(256 * 10 * 2, 256)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, T, 80] filterbanks -> [B, 256] embeddings."""
        x = F.relu(self.bn1(self.conv1(feats.transpose(1, 2).unsqueeze(1))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x)))).flatten(1, 2)
        return self.seg_1(torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + 1e-7)], dim=-1))


def _mel_banks(bins: int = 80, fft: int = 512, low: float = 20.0) -> torch.Tensor:
    """Kaldi's triangular mel filters (torchaudio.compliance.kaldi.get_mel_banks), [bins, fft / 2 + 1]."""
    mel = lambda f: 1127.0 * torch.log1p(torch.as_tensor(f, dtype=torch.float64) / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(RATE / 2)
    delta = (hi - lo) / (bins + 1)
    left = lo + torch.arange(bins, dtype=torch.float64)[:, None] * delta
    freqs = mel(RATE / fft * torch.arange(fft // 2, dtype=torch.float64))[None]
    banks = torch.clamp(torch.minimum((freqs - left) / delta, (left + 2 * delta - freqs) / delta), min=0)
    return F.pad(banks, (0, 1)).float()


def fbank(wav: torch.Tensor) -> torch.Tensor:
    """WeSpeaker's features: Kaldi's 80-bin log mel filterbank (torchaudio.compliance.kaldi.fbank
    with Hamming windows of 25 ms every 10 ms, no dither) of int16-scaled 16 kHz audio [N],
    less its mean over the clip; [T, 80]."""
    frames = wav.unfold(0, 400, 160)
    frames = frames - frames.mean(1, keepdim=True)
    frames = frames - 0.97 * torch.cat([frames[:, :1], frames[:, :-1]], 1)
    frames = frames * torch.hamming_window(400, periodic=False, device=wav.device)
    power = torch.fft.rfft(F.pad(frames, (0, 112))).abs().pow(2)
    feats = torch.clamp(power @ _mel_banks().to(wav.device).T, min=torch.finfo(torch.float32).eps).log()
    return feats - feats.mean(0, keepdim=True)


def checkpoint() -> str:
    """The speaker model's checkpoint, from the Hub cache (`h100.sh preflight` fetches it before
    a run, so a run never meets the Hub at its first validation)."""
    return hf_hub_download(SPEAKER_MODEL[0], "avg_model", revision=SPEAKER_MODEL[1])


@functools.cache
def _encoder(device: str) -> SpeakerEncoder:
    state = torch.load(checkpoint(), map_location="cpu", weights_only=True)
    model = SpeakerEncoder()
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("projection.")})
    return model.eval().to(device)


def _speech(x: np.ndarray, frame: int = 400, gap: int = 12) -> np.ndarray:
    """The 25 ms frames within 30 dB of the loudest, and the gaps under `gap` frames (0.3 s)
    between them, joined: whole stretches of speech, since every join of loose frames moves the
    embedding (two borderline frames flipped by 16-bit rounding moved a clip's similarity to
    itself to 0.96). Relative to the clip's own loudest, so a quiet recording (FLEURS-en peaks
    near -48 dBFS) keeps the same speech as a loud one. Nothing where no frame reaches -70 dBFS:
    above Mimi's silence (about -74), below FLEURS-en's quietest recording (-61)."""
    if not np.isfinite(x).all():
        raise ValueError("audio with non-finite samples")
    if len(x) < frame:
        return x[:0]
    frames = x[: len(x) // frame * frame].reshape(-1, frame)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    if db.max() < -70:
        return x[:0]
    keep = db > db.max() - 30
    loud = np.flatnonzero(keep)
    for start, end in zip(loud, loud[1:]):
        if end - start <= gap:
            keep[start:end] = True
    return frames[keep].reshape(-1)


def _embedding(x: np.ndarray, device: str) -> torch.Tensor:
    wav = torch.from_numpy(x * (1 << 15)).to(device)
    with torch.no_grad():
        return F.normalize(_encoder(device)(fbank(wav)[None])[0], dim=0)


@functools.lru_cache(maxsize=None)
def _source_embedding(path: str, device: str) -> torch.Tensor | None:
    """A source file's embedding, None under a second of speech; cached, since every validation
    scores the same rows."""
    data, rate = sphn.read(path)
    speech = _speech(soxr.resample(np.asarray(data, np.float32).mean(0), rate, RATE))
    return _embedding(speech, device) if len(speech) >= MIN_SPEECH_S * RATE else None


def hnr_db(x: np.ndarray) -> float:
    """Harmonics-to-noise ratio of 16 kHz audio (a second or more): 40 ms Hann windows every
    10 ms within 26 dB of the loudest, each with its normalized autocorrelation's peak r between
    the lags of 400 and 60 Hz (clipped to 0 .. 0.999), as 10 log10(m / (1 - m)) of their mean m."""
    n = 640
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::160].astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frames = frames[rms >= rms.max() * 0.05]
    window = np.hanning(n)
    frames = (frames - frames.mean(1, keepdims=True)) * window
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, 2 * n)) ** 2)[:, :n]
    wac = np.fft.irfft(np.abs(np.fft.rfft(window, 2 * n)) ** 2)[:n]
    ac = ac / np.maximum(ac[:, :1], 1e-12) / (wac / wac[0])
    lo, hi = RATE // 400, RATE // 60
    m = float(np.clip(ac[:, lo:hi].max(axis=1), 0, 0.999).mean())
    return 10 * math.log10(max(m, 1e-6) / (1 - m))


def score(source: str | Path, output: np.ndarray, sample_rate: int, device: str | torch.device) -> dict:
    """`speaker_sim` and `hnr_db` of a translation's mono float audio at `sample_rate` against its
    source file."""
    out = soxr.resample(np.asarray(output, np.float32), sample_rate, RATE)
    speech = _speech(out)
    if len(speech) < MIN_SPEECH_S * RATE:
        return {"speaker_sim": "", "hnr_db": ""}
    source_embedding = _source_embedding(str(source), str(device))
    return {
        "speaker_sim": "" if source_embedding is None
        else round(float(source_embedding @ _embedding(speech, str(device))), 4),
        "hnr_db": round(hnr_db(out), 2),
    }
