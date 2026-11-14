"""The voice gate: the model hears the source only around speech, and digital silence elsewhere, so
a room's noise alone (a fan, typing, the hum the level stage raises toward speech level) never
reaches it to be translated into invented words. main.py's file and mic modes use it, after the
level stage; the gate (tools/gate.py) does not, so it scores as training's evaluation does.

Silero VAD (v6, MIT, the ONNX model) scores each 32 ms of the source at 16 kHz. An 80 ms frame is
speech when any of its scores reaches THRESHOLD. The gate passes a frame when speech is in it, in
the HOLD frames before it (pauses within a sentence) or in the LOOKAHEAD frames after it (the
start of a word), so the model hears the source LOOKAHEAD frames late.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from silero_vad import load_silero_vad

FRAME = 1920  # samples at 24 kHz: one Mimi frame, 80 ms
THRESHOLD = 0.5  # Silero's default speech probability
LOOKAHEAD = 2  # frames: 160 ms
HOLD = 10  # frames: 0.8 s


class _To16k:
    """24 kHz to 16 kHz as a stream: up by 2, a low-pass below 7.6 kHz at 48 kHz, down by 3."""

    def __init__(self, taps: int = 96) -> None:
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(n * 2 * 7600 / 48000) * np.hamming(taps)
        self.h = (2 * h / h.sum()).astype(np.float32)  # 2: the gain the zero-stuffing loses
        self.history = np.zeros(taps - 1, np.float32)
        self.phase = 0

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        up = np.zeros(2 * pcm.shape[-1], np.float32)
        up[::2] = pcm
        buffer = np.concatenate([self.history, up])
        self.history = buffer[-self.history.shape[-1] :]
        out = np.convolve(buffer, self.h, mode="valid")[self.phase :: 3]
        self.phase = (self.phase - up.shape[-1]) % 3
        return out


class VoiceGate:
    """One stream's gate: call it with each 80 ms frame (FRAME samples at 24 kHz) in order; it
    returns the frame LOOKAHEAD calls back, zeroed unless speech is near it (None for the first
    LOOKAHEAD calls). `flush` returns the frames still held once the stream ends."""

    def __init__(self) -> None:
        self.vad = load_silero_vad(onnx=True)
        self.to16k = _To16k()
        self.pending = np.zeros(0, np.float32)  # 16 kHz samples short of a 512-sample window
        self.held: deque[np.ndarray] = deque()
        self.frames = 0  # frames taken
        self.last_speech = -HOLD - 1  # the index of the latest frame with speech

    def _speech(self, frame: np.ndarray) -> bool:
        self.pending = np.concatenate([self.pending, self.to16k(frame)])
        found = False
        while self.pending.shape[-1] >= 512:
            window, self.pending = self.pending[:512], self.pending[512:]
            found |= float(self.vad(torch.from_numpy(window), 16000)) >= THRESHOLD
        return found

    def _release(self) -> np.ndarray:
        index = self.frames - len(self.held)
        frame = self.held.popleft()
        return frame if self.last_speech >= index - HOLD else np.zeros_like(frame)

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        if frame.shape != (FRAME,):
            raise ValueError(f"the voice gate takes {FRAME}-sample frames, not {frame.shape}")
        if self._speech(frame):
            self.last_speech = self.frames
        self.held.append(frame)
        self.frames += 1
        return self._release() if len(self.held) > LOOKAHEAD else None

    def flush(self) -> list[np.ndarray]:
        return [self._release() for _ in range(len(self.held))]
