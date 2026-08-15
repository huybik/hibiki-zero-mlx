# PyTorch evaluation helpers used by finetune/eval.py.
import math
from pathlib import Path

import sentencepiece
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
    lm: LMModel, checkpoint_info: loaders.CheckpointInfo, batch_size: int, cfg_coef: int = 1.0
) -> LMGen:
    condition_tensors = get_condition_tensors(
        checkpoint_info.model_type, lm, batch_size=batch_size, cfg_coef=cfg_coef
    )
    lm_gen = LMGen(
        lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **checkpoint_info.lm_gen_config
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


def encode_inputs(
    batch_wavs: torch.Tensor, mimi: MimiModel, lm_gen: LMGen, audio_durations: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    with torch.no_grad():
        codes: torch.Tensor = mimi.encode(batch_wavs.to(lm_gen.lm_model.device))
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


def decode_outputs(
    batch_codes: torch.Tensor,
    batch_text_tokens: torch.Tensor,
    mimi: MimiModel,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
) -> list[tuple[torch.Tensor, str]]:
    with torch.no_grad():
        output_wavs: torch.Tensor = mimi.decode(batch_codes).cpu()

    outputs: list[tuple[torch.Tensor, str]] = []
    for output_idx, wav in enumerate(output_wavs):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        if text_tokenizer.eos_id() in text_tokens:
            eos_idx: int = text_tokens.index(text_tokenizer.eos_id())
        else:
            print(
                "warning: the model didn't generate output EOS token for "
                f"entry {output_idx}, truncating audio after the last word generated."
            )
            eos_idx: int = len(text_tokens) - 1
            while eos_idx > 0 and text_tokens[eos_idx] == text_tokenizer.pad_id():
                eos_idx -= 1
        text_tokens = [t for t in text_tokens[:eos_idx] if t > text_tokenizer.pad_id()]
        text: str = text_tokenizer.decode(text_tokens)
        wav = wav[:, : int(eos_idx * mimi.sample_rate / mimi.frame_rate)]
        outputs.append((wav, text))

    return outputs
