# PyTorch evaluation helpers used by training/finetune/eval.py.
import math
from pathlib import Path

import sphn
import torch
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors


def audio_read(
    fpath: Path, to_sample_rate: int | None = None, mono: bool = False
) -> tuple[torch.Tensor, int]:
    """Read audio fpath and resample at to_sample_rate/transform to mono audio if specified."""
    wav, sr = sphn.read(fpath)
    if to_sample_rate is not None and sr != to_sample_rate:
        wav = sphn.resample(wav, sr, to_sample_rate)
        sr = to_sample_rate
    wav_tensor: torch.Tensor = torch.tensor(wav)
    if wav_tensor.ndim == 1:
        wav_tensor.unsqueeze(0)
    elif wav_tensor.ndim == 2:
        if wav_tensor.shape[0] > 2:
            raise ValueError(
                f"Audio {fpath} has too many channels, got {wav_tensor.shape[0]} but expected 1 or 2."
            )
        elif wav_tensor.shape[0] == 2 and mono:
            print(f"Audio {fpath} is stereo, averaging both channels to get a mono audio.")
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
    elif wav_tensor.ndim >= 3:
        raise ValueError(
            f"Audio {fpath} was loaded into a tensor of unsupported shape {wav_tensor.ndim}"
        )
    return wav_tensor, sr


def stack_and_pad_audio(wavs: list[torch.Tensor], max_len: int | None = None) -> torch.Tensor:
    """Stack the given audios on the first dimenion (created), padding them with 0 if needed."""
    actual_max_len = max(wav.shape[-1] for wav in wavs)
    if max_len is None:
        max_len = actual_max_len
    else:
        assert actual_max_len <= max_len, (actual_max_len, max_len)
    other_dims = wavs[0].shape[:-1]
    out = torch.zeros(len(wavs), *other_dims, max_len, dtype=wavs[0].dtype, device=wavs[0].device)
    for k, wav in enumerate(wavs):
        out[k, ..., : wav.shape[-1]] = wav
    return out


def get_lmgen(
    lm: LMModel,
    checkpoint_info: loaders.CheckpointInfo,
    batch_size: int,
    cfg_coef: int = 1.0,
    **sampling: float,
) -> LMGen:
    """An LMGen with the checkpoint's generation config, `sampling` (temp, top_k, ...) over it."""
    condition_tensors = get_condition_tensors(
        checkpoint_info.model_type, lm, batch_size=batch_size, cfg_coef=cfg_coef
    )
    lm_gen = LMGen(
        lm,
        cfg_coef=cfg_coef,
        condition_tensors=condition_tensors,
        **{**checkpoint_info.lm_gen_config, **sampling},
    )
    return lm_gen


def add_input_eos(
    codes: torch.Tensor, mimi: MimiModel, audio_durations: list[float]
) -> torch.Tensor:
    other_audio_eos_idx: torch.Tensor = torch.tensor(
        [
            min(math.ceil(duration * mimi.frame_rate), codes.shape[-1] - 1)
            for duration in audio_durations
        ]
    )[:, None, None].to(codes.device)  # B, 1, 1
    codes_like_indexes: torch.Tensor = torch.arange(0, codes.shape[-1])[None, None].to(
        codes.device
    )  # 1, 1, T
    codes_with_input_eos: torch.Tensor = torch.where(
        codes_like_indexes == other_audio_eos_idx,
        torch.full([1], mimi.cardinality, device=codes.device),
        codes,
    )  # B, K, T
    return codes_with_input_eos


# On non-CUDA (MPS/CPU), Mimi runs in fp32 at the full sample rate (~0.8 GiB per 40 s row).
# On CUDA, `FastMimiEncoder` uses FP16 fused SEANet (~4x less VRAM), allowing larger chunks.
MIMI_CHUNK_ROWS = 8
MIMI_CHUNK_ROWS_CUDA = 16
MIMI_DECODE_CHUNK_CUDA = 8


def encode_inputs(
    batch_wavs: torch.Tensor, mimi: MimiModel, lm_gen: LMGen, audio_durations: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    device = lm_gen.lm_model.device
    chunk_rows = MIMI_CHUNK_ROWS_CUDA if device.type == "cuda" else MIMI_CHUNK_ROWS
    with torch.no_grad():
        codes: torch.Tensor = torch.cat(
            [
                mimi.encode(chunk.to(device))
                for chunk in batch_wavs.split(chunk_rows)
            ]
        )
        codes_with_input_eos = add_input_eos(codes, mimi, audio_durations)
        warmup_wav: torch.Tensor = torch.zeros(
            codes.shape[0],
            1,
            frame_size * lm_gen.max_delay,
            dtype=torch.float32,
            device=codes.device,
        )
        warmup_codes: torch.Tensor = mimi.encode(warmup_wav)
    return codes_with_input_eos, warmup_codes


def decode_wavs(
    batch_codes: torch.Tensor, end_frames: list[int], mimi: MimiModel
) -> list[torch.Tensor]:
    """Each row's audio codes up to its end frame. Mimi's decoder is causal, so right-padding
    leaves earlier samples untouched: on CUDA decode in length-sorted batches of
    `MIMI_DECODE_CHUNK_CUDA` for 3-5x faster voice-score decoding; on MPS/CPU decode one row at
    a time to keep peak memory small."""
    frame_size = mimi.sample_rate / mimi.frame_rate
    if not batch_codes.is_cuda:
        with torch.no_grad():
            return [
                mimi.decode(codes[None, :, :end])[0, :, : int(end * frame_size)].cpu()
                if end else torch.zeros(1, 0)
                for codes, end in zip(batch_codes, end_frames, strict=True)
            ]
    out: list[torch.Tensor] = [torch.zeros(1, 0) for _ in end_frames]
    active = sorted((i for i, end in enumerate(end_frames) if end > 0), key=end_frames.__getitem__)
    with torch.no_grad():
        for start in range(0, len(active), MIMI_DECODE_CHUNK_CUDA):
            group = active[start : start + MIMI_DECODE_CHUNK_CUDA]
            max_end = end_frames[group[-1]]
            sub = batch_codes[group, :, :max_end]
            wavs = mimi.decode(sub).cpu()
            for row, idx in enumerate(group):
                out[idx] = wavs[row, :, : int(end_frames[idx] * frame_size)].clone()
    return out
