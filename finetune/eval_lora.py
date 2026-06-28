#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_PAIRS_DIR,
    DEFAULT_RUN_DIR,
    DEFAULT_TOKENIZER,
    read_pair_file,
    repo_display_path,
    require_file,
    resolve_repo_path,
)
from hibiki_zero.client_utils import audio_read, stack_and_pad_audio  # noqa: E402
from hibiki_zero.inference import decode_outputs, encode_inputs, get_lmgen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tiny vi->en validation outputs with a transformer-only LoRA adapter."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_DIR / "validation.jsonl")
    parser.add_argument("--adapter", type=Path, required=True, help="Adapter .safetensors file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR / "eval")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model/LoRA dtype.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Max validation pairs to generate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gen-duration",
        type=float,
        default=0.0,
        help="Generation seconds. 0 means max source duration plus --tail-s.",
    )
    parser.add_argument("--tail-s", type=float, default=8.0, help="Silence tail when gen-duration=0.")
    parser.add_argument("--audio-temp", type=float, default=0.8)
    parser.add_argument("--text-temp", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-k-text", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="lora")
    return parser.parse_args()


def require_runtime_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import sphn
        import torch
        from moshi.models import loaders
        from moshi.modules.lora import replace_all_linear_with_lora
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit(f"Missing eval dependency: {exc.name}") from exc
    return torch, sphn, loaders, replace_all_linear_with_lora, safe_open, load_file


def check_device(torch: Any, device_name: str) -> Any:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but torch.backends.mps is not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_all(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def adapter_metadata(safe_open: Any, adapter_path: Path) -> tuple[int, float]:
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    missing = [key for key in ("target", "lora_rank", "lora_scaling") if key not in metadata]
    if missing:
        raise RuntimeError(f"Adapter is missing metadata: {', '.join(missing)}")
    if metadata["target"] not in ("LMModel.transformer", "LMModel.transformer+text_linear"):
        raise RuntimeError(f"Unsupported adapter target: {metadata['target']}")
    return int(metadata["lora_rank"]), float(metadata["lora_scaling"])


def load_main_lora(
    lm: Any,
    adapter_path: Path,
    torch: Any,
    safe_open: Any,
    load_file: Any,
    replace_all_linear_with_lora: Any,
    device: Any,
    dtype: Any,
) -> None:
    rank, scaling = adapter_metadata(safe_open, adapter_path)
    replace_all_linear_with_lora(lm.transformer, rank, scaling, device=device, dtype=dtype)
    state = load_file(str(adapter_path), device=str(device))
    allowed_prefixes = ("transformer.", "text_linear.")
    bad_keys = [key for key in state if not key.startswith(allowed_prefixes)]
    if bad_keys:
        raise RuntimeError(f"Adapter has unsupported tensors: {bad_keys[:5]}")
    for key, value in state.items():
        if value.dtype.is_floating_point:
            state[key] = value.to(dtype=dtype)
    result = lm.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:5]}")
    print(f"Loaded {len(state)} adapter tensors from {repo_display_path(adapter_path)}")


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
    outputs: list[tuple[Any, str]],
    sample_rate: int,
    out_dir: Path,
    tag: str | None,
    start_index: int,
    sphn: Any,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_index, (row, source_path, in_wav, output) in enumerate(
        zip(rows, files, input_wavs, outputs)
    ):
        global_index = start_index + local_index
        out_wav, out_text = output
        paths = output_paths(out_dir, global_index, source_path, tag)
        stereo_audio = stack_and_pad_audio([in_wav, out_wav]).squeeze()
        sphn.write_wav(paths["mono_wav"], out_wav.numpy(), sample_rate)
        sphn.write_wav(paths["stereo_wav"], stereo_audio.numpy(), sample_rate)
        paths["text"].write_text(out_text, encoding="utf-8")
        records.append(
            {
                "id": row["id"],
                "vi_audio": row["vi_audio"],
                "reference_text": row["text_en"],
                "prediction_text": out_text,
                "mono_wav": repo_display_path(paths["mono_wav"]),
                "stereo_wav": repo_display_path(paths["stereo_wav"]),
                "text_file": repo_display_path(paths["text"]),
            }
        )
    return records


def generate_batch(
    rows: list[dict[str, str]],
    batch_start: int,
    args: argparse.Namespace,
    torch: Any,
    sphn: Any,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
) -> list[dict[str, str]]:
    files = [resolve_repo_path(row["vi_audio"]) for row in rows]
    input_wavs = [audio_read(path, to_sample_rate=mimi.sample_rate, mono=True)[0] for path in files]
    audio_durations = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    gen_duration = args.gen_duration if args.gen_duration else max(audio_durations) + args.tail_s
    if max(audio_durations) > gen_duration:
        raise RuntimeError(f"Source audio is longer than generation duration: {max(audio_durations)}")

    batch_wavs = stack_and_pad_audio(input_wavs, max_len=int(gen_duration * mimi.sample_rate))
    lm_gen = get_lmgen(lm, checkpoint_info, batch_size=batch_wavs.shape[0])
    codes, warmup_codes = encode_inputs(batch_wavs, mimi, lm_gen, audio_durations)

    output_text_tokens: list[Any] = []
    output_audio_tokens: list[Any] = []
    start_time = time.time()
    with torch.no_grad(), lm_gen.streaming(batch_wavs.shape[0]):
        for step in range(warmup_codes.shape[-1]):
            _ = lm_gen.step(warmup_codes[:, :, step : step + 1])
        for step in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, step : step + 1])
            if tokens is None:
                continue
            output_text_tokens.append(tokens[:, 0, :])
            output_audio_tokens.append(tokens[:, 1:, :])
    elapsed = time.time() - start_time
    print(
        f"Generated batch {batch_start}-{batch_start + len(rows) - 1} "
        f"in {elapsed:.1f}s ({gen_duration / elapsed:.2f}x RT)"
    )

    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    batch_codes = torch.concat(output_audio_tokens, dim=-1)
    outputs = decode_outputs(batch_codes, batch_text_tokens, mimi, text_tokenizer)
    return save_batch_outputs(
        rows,
        files,
        input_wavs,
        outputs,
        mimi.sample_rate,
        out_dir,
        args.tag,
        batch_start,
        sphn,
    )


def write_predictions(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "vi_audio",
                "reference_text",
                "prediction_text",
                "mono_wav",
                "stereo_wav",
                "text_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch, sphn, loaders, replace_all_linear_with_lora, safe_open, load_file = require_runtime_deps()
    device = check_device(torch, args.device)
    dtype = dtype_from_name(torch, args.dtype)
    seed_all(torch, args.seed)

    rows = read_pair_file(args.pairs)[: args.limit]
    if not rows:
        raise RuntimeError(f"No rows selected from {args.pairs}")
    adapter = require_file(args.adapter, "LoRA adapter")
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
    checkpoint_info.lm_gen_config["temp"] = args.audio_temp
    checkpoint_info.lm_gen_config["temp_text"] = args.text_temp
    checkpoint_info.lm_gen_config["top_k"] = args.top_k
    checkpoint_info.lm_gen_config["top_k_text"] = args.top_k_text

    print(f"Loading Mimi on {device} from {repo_display_path(args.mimi_weight)}")
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
    load_main_lora(
        lm,
        adapter,
        torch,
        safe_open,
        load_file,
        replace_all_linear_with_lora,
        device,
        dtype,
    )
    lm.eval()

    records: list[dict[str, str]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        records.extend(
            generate_batch(
                batch_rows,
                start,
                args,
                torch,
                sphn,
                mimi,
                lm,
                text_tokenizer,
                checkpoint_info,
                out_dir,
            )
        )

    predictions_path = out_dir / "predictions.csv"
    write_predictions(predictions_path, records)
    print(f"Wrote {len(records)} predictions -> {repo_display_path(predictions_path)}")


if __name__ == "__main__":
    main()
