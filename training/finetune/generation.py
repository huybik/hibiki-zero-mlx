"""Free-running (sampled) translation, as `eval.py` evaluates and the trainers validate: a batch
of source clips encoded by Mimi and decoded by moshi's `LMGen`, its text, End Offset, LAAL, speech
and voice scores, the outputs written, and the trainers' validation on a bf16 copy of the policy.
"""
from __future__ import annotations

import csv
import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sphn
import torch
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.modules.transformer import set_attention_context
from moshi.run_inference import get_condition_tensors

from training.finetune import voice_scores
from training.finetune.eval_rows import (
    EVAL_SET,
    REFERENCE,
    SOURCE,
    read_eval_sets,
    read_source,
    validate_eval_rows,
)
from training.finetune.metrics import laal, score_sets
from training.finetune.model import bf16_copy, seed_all
from training.paths import repo_display_path, require_file, resolve_repo_path


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


def decode_text_batch(batch_text_tokens: Any, text_tokenizer: Any) -> list[dict[str, Any]]:
    eos_id = int(text_tokenizer.eos_id())
    pad_id = int(text_tokenizer.pad_id())
    decoded: list[dict[str, Any]] = []
    for output_idx in range(batch_text_tokens.shape[0]):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        eos_found = eos_id in text_tokens
        # The output ends at the EOS; without one it runs to the end of generation.
        eos_idx = text_tokens.index(eos_id) if eos_found else len(text_tokens)
        words = [index for index, token in enumerate(text_tokens[:eos_idx]) if token > pad_id]
        content_tokens = [text_tokens[index] for index in words]
        decoded.append(
            {
                "text": text_tokenizer.decode(content_tokens),
                "eos_found": eos_found,
                "generated_text_tokens": len(content_tokens),
                "last_word_frame": words[-1] if words else None,
                "end_frame": eos_idx,
            }
        )
    return decoded


def word_frames(text_tokens: list[int], text_tokenizer: Any) -> list[int]:
    """The frame each word of a text stream is complete at: its last piece's. The tokens above PAD
    are the words' pieces, and a piece that opens with "▁" starts a word, as decoding puts a space
    before it."""
    pad_id = int(text_tokenizer.pad_id())
    frames: list[int] = []
    for frame, token in enumerate(text_tokens):
        if token <= pad_id:
            continue
        if text_tokenizer.id_to_piece(token).startswith("▁") or not frames:
            frames.append(frame)
        else:
            frames[-1] = frame
    return frames


def output_paths(out_dir: Path, index: int, source_path: Path, tag: str | None) -> dict[str, Path]:
    suffix = "" if tag is None else f"_{tag}"
    stem = f"{index:04d}_{source_path.stem}{suffix}"
    return {
        "mono_wav": out_dir / f"{stem}_mono.wav",
        "stereo_wav": out_dir / f"{stem}_stereo.wav",
        "text": out_dir / f"{stem}.txt",
    }


def save_batch_outputs(
    rows: list[dict[str, str]],
    files: list[Path],
    input_wavs: list[Any],
    outputs: list[dict[str, Any]],
    sample_rate: int,
    out_dir: Path,
    cfg: GenerationConfig,
    start_index: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_index, (row, source_path, in_wav, output) in enumerate(
        zip(rows, files, input_wavs, outputs)
    ):
        global_index = start_index + local_index
        out_wav = output.get("wav")
        out_text = str(output["text"])
        paths = output_paths(out_dir, global_index, source_path, cfg.tag)
        mono_wav = ""
        stereo_wav = ""
        if out_wav is not None and cfg.write_audio:
            stereo_audio = stack_and_pad_audio([in_wav, out_wav]).squeeze()
            sphn.write_wav(paths["mono_wav"], out_wav.numpy(), sample_rate)
            sphn.write_wav(paths["stereo_wav"], stereo_audio.numpy(), sample_rate)
            mono_wav = repo_display_path(paths["mono_wav"])
            stereo_wav = repo_display_path(paths["stereo_wav"])
        paths["text"].write_text(out_text, encoding="utf-8")
        record = {
            "id": row[cfg.id_column],
            EVAL_SET: row[EVAL_SET],
            SOURCE: row[SOURCE],
            REFERENCE: row[REFERENCE],
            "prediction_text": out_text,
            "eos_found": bool(output["eos_found"]),
            "generated_text_tokens": int(output["generated_text_tokens"]),
            "end_offset_s": output["end_offset_s"],
            "mono_wav": mono_wav,
            "stereo_wav": stereo_wav,
            "text_file": repo_display_path(paths["text"]),
        }
        if cfg.laal:
            record["laal_s"] = output["laal_s"]
        record.update({key: output[key] for key in voice_scores.COLUMNS if key in output})
        records.append(record)
    return records


@dataclass(frozen=True)
class GenerationConfig:
    """How free-running evaluation reads its rows, how long it generates and how it samples.
    The defaults are eval.py's, and train.py selects checkpoints with them."""

    id_column: str = "id"
    gen_duration: float = 0.0  # seconds; 0 generates the batch's longest source plus tail_s
    # The translation runs seconds behind the source: the text EOS came up to 14 s after a FLEURS
    # row ended, and 8 s cut a third of them.
    tail_s: float = 20.0
    stop_on_eos: bool = True
    text_only: bool = False  # skip decoding, scoring and writing the generated audio
    write_audio: bool = True  # write each row's generated wavs (the trainers only score them)
    tag: str | None = None  # appended to every output file's name
    laal: bool = False  # also record each row's text LAAL (eval.py --laal; the trainers never do)
    # The paper samples every token at temperature 0.8, top-k 250 (Sec. 4.5), as the checkpoint's
    # lm_gen_config does. A text temperature of 0.4 sharpens each frame's choice toward PAD, so the
    # model falls behind, stalls and seldom ends.
    audio_temp: float = 0.8
    text_temp: float = 0.8
    top_k: int = 250
    top_k_text: int = 250


# How the trainers validate: eval.py's defaults, the speech scored but not written (checkpoints are
# selected on the text; the voice scores are reported).
TRAINER_GENERATION = GenerationConfig(write_audio=False)


def load_eval_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    """A trainer's evaluation rows: the set names in the order given (each direction's first
    selects the checkpoint, `eval_rows.selecting_sets`) and every set's rows, checked before the run starts rather than at its first
    validation: the columns, unique ids, a duration and the audio file of each."""
    cfg = TRAINER_GENERATION
    sets = read_eval_sets(paths)
    rows = [row for set_rows in sets.values() for row in set_rows]
    validate_eval_rows(rows, cfg.id_column, duration_column="duration_s")
    for row in rows:
        require_file(row[SOURCE], "evaluation audio")
    return list(sets), rows


def generate_batch(
    rows: list[dict[str, str]],
    batch_start: int,
    cfg: GenerationConfig,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
) -> list[dict[str, str]]:
    """Sampled autoregressive decode of a batch of source clips."""
    files = [resolve_repo_path(row[SOURCE]) for row in rows]
    input_wavs = [read_source(path, mimi.sample_rate) for path in files]
    audio_durations = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    gen_duration = cfg.gen_duration if cfg.gen_duration else max(audio_durations) + cfg.tail_s
    if max(audio_durations) > gen_duration:
        raise RuntimeError(f"Source audio is longer than generation duration: {max(audio_durations)}")

    batch_wavs = stack_and_pad_audio(input_wavs, max_len=int(gen_duration * mimi.sample_rate))
    lm_gen = get_lmgen(
        lm,
        checkpoint_info,
        batch_size=batch_wavs.shape[0],
        temp=cfg.audio_temp,
        temp_text=cfg.text_temp,
        top_k=cfg.top_k,
        top_k_text=cfg.top_k_text,
    )
    codes, warmup_codes = encode_inputs(batch_wavs, mimi, lm_gen, audio_durations)

    output_text_tokens: list[Any] = []
    output_audio_tokens: list[Any] = []
    finished = torch.zeros(batch_wavs.shape[0], dtype=torch.bool, device=codes.device)
    eos_id = int(text_tokenizer.eos_id())
    # moshi sizes the main transformer's KV cache to its whole context (3,000 frames), while a batch
    # takes only its own steps. A cache of those steps gives the same attention (unwritten slots are
    # masked out) and, at batch 64 on the Mac, is 15 GiB smaller.
    set_attention_context(lm.transformer, min(warmup_codes.shape[-1] + codes.shape[-1], lm.context))
    start_time = time.time()
    try:
        with torch.no_grad(), lm_gen.streaming(batch_wavs.shape[0]):
            for step in range(warmup_codes.shape[-1]):
                _ = lm_gen.step(warmup_codes[:, :, step : step + 1])
            for step in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, step : step + 1])
                if tokens is None:
                    continue
                output_text_tokens.append(tokens[:, 0, :])
                output_audio_tokens.append(tokens[:, 1:, :])
                finished |= tokens[:, 0, 0] == eos_id
                if cfg.stop_on_eos and bool(finished.all().detach().cpu()):
                    break
    finally:
        set_attention_context(lm.transformer, lm.context)
    elapsed = time.time() - start_time
    print(
        f"Generated batch {batch_start}-{batch_start + len(rows) - 1} "
        f"in {elapsed:.1f}s ({gen_duration / elapsed:.2f}x RT)"
    )

    if not output_text_tokens:
        raise RuntimeError("LM generation produced no output tokens; increase --gen-duration")
    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    text_outputs = decode_text_batch(batch_text_tokens, text_tokenizer)
    # How far the last generated word's text token lands after the source audio ends: the
    # paper's End Offset, measured to the end of the audio instead of its last word. Output
    # index i is the token sampled at step i, which is where the saved audio places it; after
    # max_delay warm-up frames the model chose it having heard (i - max_delay + 1) frames of
    # source. The offset counts it at index i for the 3B's max_delay of 2; a parallel-heads
    # student's larger one (up to `heads.DELAY_CAP`) delays its speech, not its text, so it is taken off.
    extra_delay = lm_gen.max_delay - 2
    for row, t, duration in zip(rows, text_outputs, audio_durations, strict=True):
        frame = t["last_word_frame"]
        t["end_offset_s"] = "" if frame is None else round((frame - extra_delay) / mimi.frame_rate - duration, 2)
        if not t["eos_found"]:
            print(f"warning: no output EOS for {row[cfg.id_column]}; keeping all generated text.")
    if cfg.laal:  # every word timed as the end offset times the last
        for row, t, tokens, duration in zip(
            rows, text_outputs, batch_text_tokens.tolist(), audio_durations, strict=True
        ):
            frames = word_frames(tokens[: t["end_frame"]], text_tokenizer)
            times = [(frame - extra_delay) / mimi.frame_rate for frame in frames]
            words = len(row[REFERENCE].split())
            t["laal_s"] = round(laal(times, duration, words), 2) if times else ""
    if cfg.text_only:
        wavs = [None] * len(text_outputs)
        voices = [{}] * len(text_outputs)
    else:
        end_frames = [t["end_frame"] for t in text_outputs]
        wavs = decode_wavs(torch.concat(output_audio_tokens, dim=-1), end_frames, mimi)
        voices = [
            voice_scores.score(path, wav[0].numpy(), mimi.sample_rate, lm.device)
            for path, wav in zip(files, wavs, strict=True)
        ]
    outputs = [
        {**t, "wav": wav, **voice} for t, wav, voice in zip(text_outputs, wavs, voices, strict=True)
    ]
    return save_batch_outputs(
        rows, files, input_wavs, outputs, mimi.sample_rate, out_dir, cfg, batch_start
    )


def translate_rows(
    rows: list[dict[str, str]],
    cfg: GenerationConfig,
    batch_size: int,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
    evaluation_seed: int | None = None,
) -> list[dict[str, str]]:
    """Free-running translation of rows with `duration_s`, sampled as `cfg` says, in length order
    in batches of `batch_size`; batch i is reseeded with `evaluation_seed` + i. A row's sample
    therefore depends on the rows it shares a batch with: scores compare under one row list, seed
    and batch size."""
    # Batches of similar length pad less: on FLEURS dev, 6,000 generation steps instead of 8,300.
    rows = sorted(rows, key=lambda row: float(row["duration_s"]))
    records: list[dict[str, str]] = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        if evaluation_seed is not None:
            seed_all(evaluation_seed + batch_index)
        records.extend(
            generate_batch(
                rows[start : start + batch_size], start, cfg, mimi, lm, text_tokenizer,
                checkpoint_info, out_dir,
            )
        )
        if lm.device.type == "mps":
            # PyTorch's MPS cache keeps a finished batch's blocks without reusing them: it grew
            # 5 GiB a batch of 16 and swapped a 48 GB Mac while live tensors stayed at 6.3 GiB.
            torch.mps.empty_cache()
    return records


def write_predictions(path: Path, records: list[dict[str, str]]) -> None:
    latency = ["end_offset_s", "laal_s"] if records and "laal_s" in records[0] else ["end_offset_s"]
    voice = list(voice_scores.COLUMNS) if records and "speaker_sim" in records[0] else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", EVAL_SET, "source_audio", "reference_text", "prediction_text", "eos_found",
                "generated_text_tokens", *latency, *voice, "mono_wav", "stereo_wav", "text_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def validate_free_running(
    lm: Any,
    checkpoint_info: Any,
    mimi: Any,
    text_tokenizer: Any,
    rows: list[dict[str, str]],
    set_names: list[str],
    batch_size: int,
    out_dir: Path,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """A trainer's free-running validation of its policy `lm` (SFT's and GRPO's): translate
    `rows` on a bf16 copy with eval.py's sampling, write `out_dir`/predictions.csv and return each
    set's metrics and health gates (`score_sets`). The copy is freed before returning, so the
    next training step has its memory back."""
    generator = bf16_copy(checkpoint_info, lm, lm.device)
    try:
        records = translate_rows(
            rows, TRAINER_GENERATION, batch_size, mimi, generator, text_tokenizer,
            checkpoint_info, out_dir, seed,
        )
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
    write_predictions(out_dir / "predictions.csv", records)
    return score_sets(records, set_names)
