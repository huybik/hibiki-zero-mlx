"""The input level stage: raises quiet source audio toward the level Hibiki trained on, before
Mimi encodes it. Mimi codes quiet speech differently from the same speech loud (a clip peaking at
−46 dBFS keeps 4% of the semantic codes it has at −6 dBFS), so quiet input was translated worse,
or not at all (docs/agent-memory/sft-evaluation.md#input-level).

Each 80 ms frame takes one gain, set from two trackers of the frame peaks: a signal envelope that
rises at once to a louder frame and falls 6 dB a second, and a noise floor that falls at once to
a quieter frame and rises 1.5 dB a second:

    envelope = max(peak, previous envelope × RELEASE)
    floor    = max(NOISE_CEILING / MAX_GAIN, min(peak, previous floor × RISE))
    gain     = min(TARGET_PEAK / envelope, NOISE_CEILING / floor), at least 1

Quiet speech is raised toward TARGET_PEAK, by at most MAX_GAIN, but never so far that its noise
floor passes NOISE_CEILING: the gain cannot tell quiet speech from a quiet room, so pauses and a
stream's start, where the envelope falls toward the floor, end up near the ceiling rather than at
the speech target, and a recording whose floor is above the ceiling is left as it is. Frames
whose envelope is above TARGET_PEAK pass unchanged, and a raised frame never peaks above it, so
nothing clips. It is causal and frame by frame, so a stream gives the same samples in any
whole-frame chunks. The MLX runtime and the PyTorch evaluation use this module; the Swift
runtime mirrors it (`Audio.Leveller`).
"""

from __future__ import annotations

import math

import numpy as np

FRAME = 1920  # samples at 24 kHz: one Mimi frame, 80 ms
FRAME_RATE = 12.5
TARGET_PEAK = 10 ** (-20 / 20)
MAX_GAIN = 10 ** (30 / 20)  # speech peaking at −50 dBFS reaches the target
NOISE_CEILING = 10 ** (-40 / 20)  # quieter than the trained noisy sources' noise
RELEASE = 10 ** (-6 / 20 / FRAME_RATE)  # per frame: the envelope falls 6 dB a second
RISE = 10 ** (1.5 / 20 / FRAME_RATE)  # per frame: the floor rises 1.5 dB a second


class Leveller:
    """One stream's level stage: call it on the stream's samples (1-D float, 24 kHz) in order, in
    chunks of whole frames (the last chunk may end mid-frame); a new stream takes a new one."""

    def __init__(self) -> None:
        self.envelope = 0.0
        self.noise_floor = math.inf  # the first frame sets it

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.ndim != 1 or pcm.dtype.kind != "f":
            raise ValueError(f"the level stage takes 1-D float samples, not {pcm.dtype} {pcm.shape}")
        out = np.empty_like(pcm)
        for start in range(0, len(pcm), FRAME):
            frame = pcm[start : start + FRAME]
            peak = float(np.abs(frame).max())
            if not math.isfinite(peak):
                raise ValueError(f"non-finite source sample in samples {start}-{start + len(frame)}")
            self.envelope = max(peak, self.envelope * RELEASE)
            self.noise_floor = max(NOISE_CEILING / MAX_GAIN, min(peak, self.noise_floor * RISE))
            gain = NOISE_CEILING / self.noise_floor
            if self.envelope > 0:
                gain = min(gain, TARGET_PEAK / self.envelope)
            out[start : start + FRAME] = frame * np.float32(gain) if gain > 1.0 else frame
        return out
