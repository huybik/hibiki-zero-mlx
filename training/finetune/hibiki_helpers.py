# PyTorch evaluation helpers used by training/finetune/eval.py.
import math
from pathlib import Path

import sphn
import torch
from moshi.conditioners import ConditionAttributes
from moshi.models import LMGen, LMModel, MimiModel, loaders

# The two-way model's condition: a lookup table summed into every frame's input, one entry a
# direction (`training.finetune.direction` adds it), so the runtimes say which way to translate.
DIRECTION = "direction"


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


def direction_conditions(lm: LMModel, directions: list[str]) -> dict | None:
    """The condition tensors for rows translated in `directions` (<source>-<target>, one a row):
    a conditioned model's direction entries, nothing for a model without conditioners (the
    upstream model, the two-way checkpoints before the condition). Computed in the model's graph,
    so training reaches the table through them."""
    provider = lm.condition_provider
    if provider is None or not provider.conditioners:
        return None if lm.fuser is None else {}
    if set(provider.conditioners) != {DIRECTION}:
        raise ValueError(f"unsupported conditioners {sorted(provider.conditioners)}")
    attributes = [ConditionAttributes(text={DIRECTION: direction}, tensor={}) for direction in directions]
    return provider.prepare_and_provide(attributes)


def get_lmgen(
    lm: LMModel, checkpoint_info: loaders.CheckpointInfo, directions: list[str], **sampling: float
) -> LMGen:
    """An LMGen for a batch of rows translated in `directions`, with the checkpoint's generation
    config, `sampling` (temp, top_k, ...) over it."""
    with torch.no_grad():
        conditions = direction_conditions(lm, directions)
    return LMGen(lm, condition_tensors=conditions, **{**checkpoint_info.lm_gen_config, **sampling})


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


# Mimi runs in fp32 at the full sample rate, so its activations grow with rows times length:
# encoding one 40 s row peaks at 0.8 GiB, and a whole batch of 64 rows would not fit beside the
# model on an 80 GB card. Each row is coded independently, so chunks give the same codes.
MIMI_CHUNK_ROWS = 8


def encode_inputs(
    batch_wavs: torch.Tensor, mimi: MimiModel, lm_gen: LMGen, audio_durations: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    with torch.no_grad():
        codes: torch.Tensor = torch.cat(
            [
                mimi.encode(chunk.to(lm_gen.lm_model.device))
                for chunk in batch_wavs.split(MIMI_CHUNK_ROWS)
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
    """Decode each row's audio codes and cut it at its end frame."""
    with torch.no_grad():
        output_wavs = torch.cat(
            [mimi.decode(chunk).cpu() for chunk in batch_codes.split(MIMI_CHUNK_ROWS)]
        )
    frame_size = mimi.sample_rate / mimi.frame_rate
    return [
        wav[:, : int(end * frame_size)] for wav, end in zip(output_wavs, end_frames, strict=True)
    ]
