"""Rooms where nobody speaks: a share of streams hold a room's noise in their source instead of
digital silence (GRPO's, and SFT's with `train.py --room-share`), its lead, gaps and the stretch
after its EOS alike, as a phone's microphone hears a room between sentences
([noisy-source](../../docs/agent-memory/noisy-source.md#rooms)). Each is encoded from a fresh Mimi
state (`encode_track`, which `streams.silence_track` encodes the silence with).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.audio import augment
from training.paths import require_dir


def encode_track(mimi: Any, pcm: torch.Tensor, codebooks: int | None = None) -> torch.Tensor:
    """Mimi codes of `pcm` (1-D, Mimi's sample rate, whole frames) encoded from a fresh state, as
    inference encodes a stream, [codebooks, frames] int32 on the CPU: `codebooks` of them, or as
    many as `mimi` is set to."""
    frame = int(round(mimi.sample_rate / mimi.frame_rate))
    if pcm.ndim != 1 or len(pcm) % frame:
        raise ValueError(f"Mimi encodes whole {frame}-sample frames of one channel, not {tuple(pcm.shape)}")
    active = mimi.num_codebooks
    if codebooks is not None:
        mimi.set_num_codebooks(codebooks)
    device = next(mimi.parameters()).device
    try:
        with torch.no_grad():
            codes = mimi.encode(pcm.to(device=device, dtype=torch.float32)[None, None])[0]
    finally:
        mimi.set_num_codebooks(active)  # the caller's Mimi keeps its own count
    if codes.shape[-1] != len(pcm) // frame:
        raise RuntimeError(f"Mimi encoded {codes.shape[-1]} frames, expected {len(pcm) // frame}")
    return codes.to(device="cpu", dtype=torch.int32)


ROOM_BATCH_CUDA = 8


def room_codes(mimi: Any, bank: Any, count: int, frames: int, codebooks: int, seed: int) -> torch.Tensor:
    """Mimi codes of `count` rooms where nobody speaks (`augment.room_tracks` from the noise bank
    `bank`), each `frames` long from a fresh state: [count, codebooks, frames] int32 on the CPU.
    Batched on CUDA (all room tracks share the same `samples` length, zero padding waste)."""
    frame = int(round(mimi.sample_rate / mimi.frame_rate))
    samples = frame * frames
    tracks = augment.room_tracks(bank, count, samples, seed)
    device = next(mimi.parameters()).device
    if device.type != "cuda":
        return torch.stack([encode_track(mimi, torch.from_numpy(track), codebooks) for track in tracks])
    active = mimi.num_codebooks
    mimi.set_num_codebooks(codebooks)
    out: list[torch.Tensor] = []
    buf: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for track in tracks:
                buf.append(torch.from_numpy(track))
                if len(buf) == ROOM_BATCH_CUDA:
                    batch = torch.stack(buf, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
                    out.append(mimi.encode(batch).to(device="cpu", dtype=torch.int32))
                    buf.clear()
            if buf:
                batch = torch.stack(buf, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
                out.append(mimi.encode(batch).to(device="cpu", dtype=torch.int32))
    finally:
        mimi.set_num_codebooks(active)
    result = torch.cat(out, dim=0)
    if result.shape[-1] != frames:
        raise RuntimeError(f"Mimi encoded {result.shape[-1]} room frames, expected {frames}")
    return result


def add_room_args(parser: Any, share: float) -> None:
    """The rooms' options, --room-share defaulting to `share`."""
    parser.add_argument(
        "--room-share", type=float, default=share,
        help="The share of streams whose source holds a room's noise where nobody speaks; 0 keeps digital silence.",
    )
    parser.add_argument("--bank-dir", type=Path, default=Path("data/rirs_noises"), help="The noise bank's parent (augment.fetch_bank's root).")


def run_rooms(args: Any, mimi: Any, silence: torch.Tensor, count: int) -> torch.Tensor | None:
    """A run's `count` rooms where nobody speaks (`room_codes`, drawn with --seed from the bank
    under --bank-dir), shaped as its `silence` track, encoded by `mimi` (the run's, on its GPU);
    None at --room-share 0."""
    if args.room_share <= 0:
        return None
    bank = require_dir(args.bank_dir / "RIRS_NOISES", "noise bank (augment.fetch_bank; grpo_launch.sh fetches it)")
    return room_codes(mimi, bank, count, silence.shape[1], silence.shape[0], args.seed)
