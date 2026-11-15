"""Simulated recording conditions for the Vietnamese source: room reverb, background noise, a
narrowed band and a level change, applied to a 24 kHz clip without changing its length.

    clip ─► reverb (p 0.5) ─► + noise at 5–30 dB SNR (p 0.8) ─► band ─► gain, clipped to ±1
            room impulse        background 0.7, looped        full 24 kHz 0.40 · 16 kHz rate 0.35
            responses           foreground 0.3, one event     · 8 kHz rate 0.15 · phone 0.10

The impulse responses and noises are OpenSLR 28 (RIRS_NOISES, Apache 2.0): real and simulated
small and medium room responses, and point-source and diffuse noises, all 16 kHz, resampled to
24 kHz with soxr as the cache builder resamples. Every draw is seeded by the row id, so a clip gets
the same conditions on every machine and rerun. Runs on any torch device.

`room_tracks` draws the room a stream holds where nobody speaks (GRPO's quiet stretches): a
background noise with foreground events, through the input level stage as a phone hears it.
Every QUIET_HELD_OUT-th noise file of each kind is left out of it, for testing.
"""

from __future__ import annotations

import hashlib
import math
import random
import tempfile
import urllib.request
import wave
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr
import torch

from inference.hibiki_mlx.level import Leveller

SAMPLE_RATE = 24_000
BANK_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
BANK_SHA256 = "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
BANK_RATE = 16_000
SIMULATED_ROOMS = ("smallroom", "mediumroom")
SIMULATED_PER_ROOM = 1_000  # of 20,000 each: plenty of rooms, a fast load
RIR_LEAD_S = 0.0025  # kept before the direct path's peak
RIR_MAX_S = 1.0
REVERB_P = 0.5
NOISE_P = 0.8
# Steady background (the point-source noises marked background and the rooms' diffuse noises,
# 129 files) more often than foreground events (806: doors, keys, animals, ...).
NOISE_KINDS = {"background": 0.7, "foreground": 0.3}
SNR_DB = (5.0, 30.0)
GAIN_DB = (-15.0, 0.0)  # TTS peaks at a median 0.73: louder clips most rows
# name: (probability, high-pass Hz, low-pass Hz); the 16 and 8 kHz bands are FLEURS's and
# MultiMed-ST's sampling rates, the phone band the narrowband telephone channel.
BANDS = {
    "full": (0.40, None, None),
    "16k": (0.35, None, 8_000.0),
    "8k": (0.15, None, 4_000.0),
    "phone": (0.10, 300.0, 3_400.0),
}
TRANSITION = 0.1  # raised-cosine edge, as a fraction of the cutoff
# A room where nobody speaks, before the level stage raises it: a background peaking from a quiet
# room to a loud one (the user's room peaked at -54 dBFS; the stage lifts a floor under -40 dBFS
# to -40), with an event (a door, a cough) starting every 15 s on average: an excerpt of a
# foreground file (they run up to 8 min, median 11 s) of at most QUIET_EVENT_MAX_S.
QUIET_PEAK_DB = (-65.0, -35.0)
QUIET_EVENT_PEAK_DB = (-45.0, -25.0)
QUIET_EVENTS_PER_S = 1 / 15
QUIET_EVENT_MAX_S = 3.0
QUIET_EVENT_FADE_S = 0.01  # an excerpt's cut edges ramp in and out, so they add no click
QUIET_HELD_OUT = 5  # every fifth file of each kind of noise, kept out of room_tracks for testing


@dataclass(frozen=True)
class Conditions:
    """The drawn conditions of one clip, stored with the row (`dataclasses.asdict`)."""

    rir: str | None
    noise: str | None
    # Where the noise starts: in a background loop, or of the room a foreground event leaves in
    # the clip (in the event, when it outlasts the clip); a fraction.
    noise_offset: float
    snr_db: float | None
    band: str
    gain_db: float


def fetch_bank(root: Path) -> Path:
    """Download and unpack RIRS_NOISES under `root` once; return its directory."""
    bank = root / "RIRS_NOISES"
    if bank.is_dir():
        return bank
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "rirs_noises.zip"
    if not archive.is_file():
        tmp = archive.with_name(archive.name + ".tmp")
        urllib.request.urlretrieve(BANK_URL, tmp)
        tmp.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != BANK_SHA256:
        raise RuntimeError(f"{archive} has SHA-256 {digest.hexdigest()}, expected {BANK_SHA256}")
    # Unpacked aside and renamed into place: a cut-short unpack must not pass for the bank.
    with tempfile.TemporaryDirectory(dir=root, prefix=".unpack-") as tmp:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(tmp)
        (Path(tmp) / bank.name).rename(bank)
    return bank


def _read(path: Path) -> np.ndarray:
    """Channel 0 of a 16 kHz 16-bit bank file at 24 kHz. The standard library reads every file;
    sphn refuses the multichannel ones' WAVE_FORMAT_EXTENSIBLE headers."""
    with wave.open(str(path)) as handle:
        if handle.getframerate() != BANK_RATE or handle.getsampwidth() != 2:
            raise RuntimeError(f"{path}: {handle.getframerate()} Hz, {handle.getsampwidth()} bytes")
        channels = handle.getnchannels()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    first = pcm.reshape(-1, channels)[:, 0].astype(np.float32) / 32768
    return soxr.resample(first, BANK_RATE, SAMPLE_RATE).astype(np.float32)


def noise_files(bank: Path) -> dict[str, list[Path]]:
    """The bank's background noises (the point-source files marked background, the rooms' diffuse
    noises) and foreground events, each sorted."""
    return {
        "background": _listed(bank, "pointsource_noises", "noise", "--bg-fg-type background")
        + _listed(bank, "real_rirs_isotropic_noises", "noise"),
        "foreground": _listed(bank, "pointsource_noises", "noise", "--bg-fg-type foreground"),
    }


def quiet_noises(bank: Path, held_out: bool = False) -> dict[str, list[Path]]:
    """The noise files `room_tracks` draws from, or with `held_out` the ones it leaves for testing
    (every QUIET_HELD_OUT-th of each kind)."""
    return {
        kind: [path for index, path in enumerate(paths) if (index % QUIET_HELD_OUT == 0) == held_out]
        for kind, paths in noise_files(bank).items()
    }


def room_tracks(bank: Path, count: int, samples: int, seed: int) -> Iterator[np.ndarray]:
    """`count` tracks of `samples` of a room where nobody speaks, at 24 kHz, as the runtimes hand
    them to Mimi: a background noise looped from a drawn offset and peaking in QUIET_PEAK_DB, and
    events starting at QUIET_EVENTS_PER_S on average, each an excerpt of at most
    QUIET_EVENT_MAX_S of a foreground file peaking in QUIET_EVENT_PEAK_DB; then the input level
    stage, run from the track's start as on a new stream. Drawn from `seed` and the files
    `quiet_noises` keeps."""
    noises = quiet_noises(bank)
    rng = random.Random(seed)
    read: dict[Path, np.ndarray] = {}

    def file(path: Path) -> np.ndarray:
        if path not in read:
            read[path] = _read(path)
        return read[path]

    def at_peak(wav: np.ndarray, peak_db: float) -> np.ndarray:
        return wav * (10 ** (peak_db / 20) / max(float(np.abs(wav).max()), 1e-6))

    longest = round(QUIET_EVENT_MAX_S * SAMPLE_RATE)
    fade = round(QUIET_EVENT_FADE_S * SAMPLE_RATE)
    for _ in range(count):
        background = file(rng.choice(noises["background"]))
        start = rng.randrange(len(background))
        track = at_peak(np.resize(np.roll(background, -start), samples), rng.uniform(*QUIET_PEAK_DB))
        track = track.astype(np.float32)
        at = rng.expovariate(QUIET_EVENTS_PER_S) * SAMPLE_RATE
        while at < samples:
            source = file(rng.choice(noises["foreground"]))
            offset = rng.randrange(max(1, len(source) - longest + 1))
            event = source[offset : offset + longest].copy()
            ramp = np.linspace(0.0, 1.0, min(fade, len(event) // 2), dtype=np.float32)
            if len(ramp):
                event[: len(ramp)] *= ramp
                event[len(event) - len(ramp) :] *= ramp[::-1]
            begin = int(at)
            end = min(samples, begin + len(event))
            track[begin:end] += at_peak(event, rng.uniform(*QUIET_EVENT_PEAK_DB))[: end - begin]
            at += rng.expovariate(QUIET_EVENTS_PER_S) * SAMPLE_RATE
        yield Leveller()(np.clip(track, -1.0, 1.0))


def _listed(bank: Path, folder: str, kind: str, match: str = "") -> list[Path]:
    """The files a bank folder's `rir_list` or `noise_list` names on lines holding `match`
    (Kaldi's lists: the path is the last field, relative to the bank's parent), sorted."""
    lines = (bank / folder / f"{kind}_list").read_text().splitlines()
    return sorted(
        bank.parent / line.split()[-1] for line in lines if line.strip() and match in line
    )


class Bank:
    """The room impulse responses and noises, at 24 kHz on `device`."""

    def __init__(self, bank: Path, device: torch.device | str):
        rirs = _listed(bank, "real_rirs_isotropic_noises", "rir")
        for room in SIMULATED_ROOMS:
            pool = sorted((bank / "simulated_rirs" / room).glob("Room*/*.wav"))
            rirs += sorted(random.Random(room).sample(pool, SIMULATED_PER_ROOM))
        noises = noise_files(bank)
        self.rir_names = [str(path.relative_to(bank)) for path in rirs]
        self.noise_names = {
            kind: [str(p.relative_to(bank)) for p in paths] for kind, paths in noises.items()
        }
        self.rirs = {
            name: torch.from_numpy(_rir(_read(p))).to(device)
            for name, p in zip(self.rir_names, rirs)
        }
        self.noises = {
            str(p.relative_to(bank)): torch.from_numpy(_read(p)).to(device)
            for paths in noises.values()
            for p in paths
        }
        self.foreground = set(self.noise_names["foreground"])
        if not self.rirs or not all(self.noise_names.values()):
            raise RuntimeError(f"{bank} lacks impulse responses or a kind of noise")

    def draw(self, key: str) -> Conditions:
        """The conditions for the clip `key` (a row id with its seed)."""
        rng = random.Random(key)
        rir = rng.choice(self.rir_names) if rng.random() < REVERB_P else None
        noise = None
        if rng.random() < NOISE_P:
            kind = rng.choices(list(NOISE_KINDS), weights=list(NOISE_KINDS.values()))[0]
            noise = rng.choice(self.noise_names[kind])
        offset = rng.random()
        snr_db = rng.uniform(*SNR_DB) if noise else None
        band = rng.choices(list(BANDS), weights=[p for p, _, _ in BANDS.values()])[0]
        return Conditions(
            rir=rir,
            noise=noise,
            noise_offset=offset,
            snr_db=snr_db,
            band=band,
            gain_db=rng.uniform(*GAIN_DB),
        )

    def apply(self, clip: torch.Tensor, conditions: Conditions) -> torch.Tensor:
        """`clip` (1-D, 24 kHz, on the bank's device) under `conditions`, the same length."""
        out = clip
        level = _rms(clip)
        if conditions.rir is not None:
            wet = _convolve(out, self.rirs[conditions.rir])
            out = wet * (level / _rms(wet))
        if conditions.noise is not None:
            noise = self.noises[conditions.noise]
            if conditions.noise in self.foreground:
                # One event (a door, a cough), its SNR measured over the event itself.
                track, event = _event(noise, conditions.noise_offset, out.shape[-1])
            else:
                start = int(conditions.noise_offset * noise.shape[-1])
                track = event = _loop(noise, start, out.shape[-1])
            out = out + track * (level / (_rms(event) * 10 ** (conditions.snr_db / 20)))
        _, high_pass, low_pass = BANDS[conditions.band]
        if high_pass or low_pass:
            out = _band(out, high_pass, low_pass)
        return (out * 10 ** (conditions.gain_db / 20)).clamp(-1.0, 1.0)


def _rir(wav: np.ndarray) -> np.ndarray:
    """An impulse response from just before its direct path, at most RIR_MAX_S, unit energy."""
    start = max(0, int(np.argmax(np.abs(wav))) - round(RIR_LEAD_S * SAMPLE_RATE))
    rir = wav[start : start + round(RIR_MAX_S * SAMPLE_RATE)]
    return rir / np.sqrt(np.sum(rir**2))


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt().clamp_min(1e-8)


def _fft_size(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal linear convolution, cut to x's length (the tail past the clip's end is dropped)."""
    size = _fft_size(x.shape[-1] + h.shape[-1] - 1)
    y = torch.fft.irfft(torch.fft.rfft(x, size) * torch.fft.rfft(h, size), size)
    return y[: x.shape[-1]]


def _event(noise: torch.Tensor, offset: float, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A silent track of `length` holding `noise` once at `offset` of the room left, or the
    window of `noise` at `offset` when it is longer; and the event's own samples."""
    if noise.shape[-1] >= length:
        start = int(offset * (noise.shape[-1] - length))
        window = noise[start : start + length]
        return window, window
    start = int(offset * (length - noise.shape[-1]))
    track = torch.zeros(length, dtype=noise.dtype, device=noise.device)
    track[start : start + noise.shape[-1]] = noise
    return track, noise


def _loop(noise: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """`length` samples of `noise` from `start`, wrapping around."""
    repeats = math.ceil((start + length) / noise.shape[-1])
    return noise.repeat(repeats)[start : start + length]


def _band(x: torch.Tensor, high_pass: float | None, low_pass: float | None) -> torch.Tensor:
    """Zero-phase band limit with raised-cosine edges, on a zero-padded FFT."""
    size = _fft_size(x.shape[-1] + SAMPLE_RATE // 10)
    freqs = torch.fft.rfftfreq(size, 1 / SAMPLE_RATE, device=x.device)
    gain = torch.ones_like(freqs)
    if low_pass:
        gain = gain * _edge(freqs, low_pass * (1 - TRANSITION), low_pass)
    if high_pass:
        gain = gain * (1 - _edge(freqs, high_pass * (1 - TRANSITION), high_pass))
    return torch.fft.irfft(torch.fft.rfft(x, size) * gain, size)[: x.shape[-1]]


def _edge(freqs: torch.Tensor, start: float, stop: float) -> torch.Tensor:
    """1 below `start`, 0 above `stop`, a raised cosine between."""
    t = ((freqs - start) / (stop - start)).clamp(0.0, 1.0)
    return 0.5 * (1 + torch.cos(math.pi * t))
