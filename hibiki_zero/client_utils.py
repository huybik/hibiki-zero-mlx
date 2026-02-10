# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import torch
import sphn
from IPython.display import Audio, display


def get_color_code(color_name: str) -> tuple:
    if color_name == "blue":
        return (92, 158, 255)
    elif color_name == "yellow":
        return (255, 255, 85)
    elif color_name == "red":
        return (255, 85, 85)
    elif color_name == "orange":
        return (255, 171, 64)
    elif color_name == "green":
        return (57, 242, 174)


def colorize_rgb(text: str, rgb: tuple) -> str:
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"


def make_colored_log(
    level: str, msg: str, colored_parts: list[tuple[str, str]] | None = None
) -> str:
    if level == "info":
        prefix = colorize_rgb("[Info]", get_color_code("blue"))
    elif level == "warning":
        prefix = colorize_rgb("[Warn]", get_color_code("yellow"))
    elif level == "error":
        prefix = colorize_rgb("[Err ]", get_color_code("red"))
    else:
        raise ValueError(f"Unknown level {level}")
    if colored_parts is not None:
        msg = msg.format(
            *[colorize_rgb(text, get_color_code(color_code)) for text, color_code in colored_parts]
        )
    return prefix + " " + msg


def log(level: str, msg: str, colored_parts: list[tuple[str, str]] | None = None) -> None:
    """Log something with a given level."""
    print(make_colored_log(level, msg, colored_parts))


def audio_read(fpath: Path, to_sample_rate: int | None = None, mono: bool = False) -> tuple[torch.Tensor, int]:
    """ Read audio fpath and resample at to_sample_rate/transform to mono audio if specified. """
    wav, sr = sphn.read(fpath)
    if to_sample_rate is not None and sr != to_sample_rate:
        wav = sphn.resample(wav, sr, to_sample_rate)
        sr = to_sample_rate
    wav_tensor: torch.Tensor = torch.tensor(wav)
    if wav_tensor.ndim == 1:
        wav_tensor.unsqueeze(0)
    elif wav_tensor.ndim == 2:
        if wav_tensor.shape[0] > 2:
            raise ValueError(f"Audio {fpath} has too many channels, got {wav_tensor.shape[0]} but expected 1 or 2.")
        elif wav_tensor.shape[0] == 2 and mono:
            print(f"Audio {fpath} is stereo, averaging both channels to get a mono audio.")
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)            
    elif wav_tensor.ndim >= 3:
        raise ValueError(f"Audio {fpath} was loaded into a tensor of unsupported shape {wav_tensor.ndim}")
    return wav_tensor, sr


def stack_and_pad_audio(wavs: list[torch.Tensor], max_len: int | None = None) -> torch.Tensor:
    """ Stack the given audios on the first dimenion (created), padding them with 0 if needed. """
    actual_max_len = max(wav.shape[-1] for wav in wavs)
    if max_len is None:
        max_len = actual_max_len
    else:
        assert max_len >= actual_max_len, (max_len, actual_max_len)
    other_dims = wavs[0].shape[:-1]
    out = torch.zeros(len(wavs), *other_dims, max_len, dtype=wavs[0].dtype, device=wavs[0].device)
    for k, wav in enumerate(wavs):
        out[k, ..., : wav.shape[-1]] = wav
    return out


def display_audio(wav: torch.Tensor, sample_rate: int):
    """ Display an audio reader to be used in an interactive notebook. """
    display(Audio(wav.numpy(), rate=sample_rate))
