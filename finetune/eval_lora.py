#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import re
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
    repo_display_path,
    require_file,
    resolve_repo_path,
)
from hibiki_zero.client_utils import audio_read, stack_and_pad_audio  # noqa: E402
from hibiki_zero.inference import decode_outputs, encode_inputs, get_lmgen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validation outputs with the base model or a LoRA adapter."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_DIR / "validation.jsonl")
    parser.add_argument("--adapter", type=Path, help="Optional adapter .safetensors file.")
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
        "--ids",
        default="",
        help="Comma-separated ids to generate in that order; bypasses --limit.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Text file with one id per line to generate in that order; bypasses --limit.",
    )
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
    parser.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a batch once every row has emitted text EOS.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Decode/write text sidecars and predictions.csv only; skip generated wav decode/write.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional metrics JSON path. Defaults to <out-dir>/metrics.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="lora")
    parser.add_argument("--source-column", default="vi_audio")
    parser.add_argument("--reference-column", default="text_en")
    parser.add_argument("--id-column", default="id")
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


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    path = require_file(path, "eval rows")
    if path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [
                {key: value or "" for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    if path.suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append({key: str(value) for key, value in row.items()})
        return rows
    raise ValueError(f"Eval rows must be .jsonl or .csv: {path}")


def validate_eval_rows(
    rows: list[dict[str, str]], source_column: str, reference_column: str, id_column: str
) -> None:
    if not rows:
        return
    missing = [
        column
        for column in (source_column, reference_column, id_column)
        if column not in rows[0]
    ]
    if missing:
        raise ValueError(f"Eval rows are missing columns: {', '.join(missing)}")


def selected_ids(args: argparse.Namespace) -> list[str]:
    ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    if args.ids_file is not None:
        path = require_file(args.ids_file, "id file")
        ids.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids in --ids/--ids-file")
    return ids


def select_eval_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    ids = selected_ids(args)
    if not ids:
        return rows[: args.limit]
    by_id = {row[args.id_column]: row for row in rows}
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Requested ids are missing from {args.pairs}: {missing[:10]}")
    return [by_id[row_id] for row_id in ids]


def adapter_metadata(safe_open: Any, adapter_path: Path) -> tuple[int, float, set[str]]:
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    missing = [key for key in ("target", "lora_rank", "lora_scaling") if key not in metadata]
    if missing:
        raise RuntimeError(f"Adapter is missing metadata: {', '.join(missing)}")
    targets = set(metadata["target"].split("+"))
    supported = {"LMModel.transformer", "text_linear", "audio_heads"}
    if "LMModel.transformer" not in targets or targets - supported:
        raise RuntimeError(f"Unsupported adapter target: {metadata['target']}")
    return int(metadata["lora_rank"]), float(metadata["lora_scaling"]), targets


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
    rank, scaling, targets = adapter_metadata(safe_open, adapter_path)
    replace_all_linear_with_lora(lm.transformer, rank, scaling, device=device, dtype=dtype)
    if "audio_heads" in targets:
        replace_all_linear_with_lora(lm.depformer_in, rank, scaling, device=device, dtype=dtype)
        replace_all_linear_with_lora(lm.linears, rank, scaling, device=device, dtype=dtype)
    state = load_file(str(adapter_path), device=str(device))
    allowed_prefixes = ["transformer."]
    if "text_linear" in targets:
        allowed_prefixes.append("text_linear.")
    if "audio_heads" in targets:
        allowed_prefixes.extend(("depformer_in.", "linears."))
    bad_keys = [key for key in state if not key.startswith(tuple(allowed_prefixes))]
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


def decode_text_batch(
    batch_text_tokens: Any,
    text_tokenizer: Any,
    warn: bool,
) -> list[dict[str, Any]]:
    eos_id = int(text_tokenizer.eos_id())
    pad_id = int(text_tokenizer.pad_id())
    decoded: list[dict[str, Any]] = []
    for output_idx in range(batch_text_tokens.shape[0]):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        if eos_id in text_tokens:
            eos_idx = text_tokens.index(eos_id)
            eos_found = True
        else:
            if warn:
                print(
                    "warning: model did not generate output EOS token for "
                    f"entry {output_idx}; truncating text after the last non-pad token."
                )
            eos_idx = len(text_tokens) - 1
            while eos_idx > 0 and text_tokens[eos_idx] == pad_id:
                eos_idx -= 1
            eos_found = False
        content_tokens = [token for token in text_tokens[:eos_idx] if token > pad_id]
        decoded.append(
            {
                "text": text_tokenizer.decode(content_tokens),
                "eos_found": eos_found,
                "generated_text_tokens": len(content_tokens),
            }
        )
    return decoded


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_words = normalize(reference).split()
        hyp_words = normalize(hypothesis).split()
        edits += edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
    return edits / total_words if total_words else 0.0


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    references = [str(record["reference_text"]).strip() for record in records]
    hypotheses = [str(record["prediction_text"]).strip() for record in records]
    nonempty_pairs = [
        (reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
        if hypothesis
    ]
    metrics: dict[str, Any] = {
        "num_predictions": len(records),
        "nonempty_predictions": sum(1 for hypothesis in hypotheses if hypothesis),
        "empty_predictions": sum(1 for hypothesis in hypotheses if not hypothesis),
        "exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if reference == hypothesis
        ),
        "normalized_exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if normalize(reference) == normalize(hypothesis)
        ),
        "eos_found": sum(1 for record in records if record.get("eos_found") is True),
        "eos_missing": sum(1 for record in records if record.get("eos_found") is False),
        "wer": word_error_rate(references, hypotheses),
    }
    try:
        import sacrebleu
    except ImportError:
        metrics["sacrebleu"] = "missing"
        return metrics

    metrics["bleu"] = sacrebleu.corpus_bleu(hypotheses, [references]).score
    metrics["chrf"] = sacrebleu.corpus_chrf(hypotheses, [references]).score
    if nonempty_pairs:
        nonempty_refs, nonempty_hyps = zip(*nonempty_pairs, strict=True)
        metrics["nonempty_bleu"] = sacrebleu.corpus_bleu(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
        metrics["nonempty_chrf"] = sacrebleu.corpus_chrf(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
    else:
        metrics["nonempty_bleu"] = 0.0
        metrics["nonempty_chrf"] = 0.0
    return metrics


def save_batch_outputs(
    rows: list[dict[str, str]],
    files: list[Path],
    input_wavs: list[Any],
    outputs: list[dict[str, Any]],
    sample_rate: int,
    out_dir: Path,
    tag: str | None,
    start_index: int,
    sphn: Any,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_index, (row, source_path, in_wav, output) in enumerate(
        zip(rows, files, input_wavs, outputs)
    ):
        global_index = start_index + local_index
        out_wav = output.get("wav")
        out_text = str(output["text"])
        paths = output_paths(out_dir, global_index, source_path, tag)
        mono_wav = ""
        stereo_wav = ""
        if out_wav is not None:
            stereo_audio = stack_and_pad_audio([in_wav, out_wav]).squeeze()
            sphn.write_wav(paths["mono_wav"], out_wav.numpy(), sample_rate)
            sphn.write_wav(paths["stereo_wav"], stereo_audio.numpy(), sample_rate)
            mono_wav = repo_display_path(paths["mono_wav"])
            stereo_wav = repo_display_path(paths["stereo_wav"])
        paths["text"].write_text(out_text, encoding="utf-8")
        records.append(
            {
                "id": row[args.id_column],
                "source_audio": row[args.source_column],
                "reference_text": row[args.reference_column],
                "prediction_text": out_text,
                "eos_found": bool(output["eos_found"]),
                "generated_text_tokens": int(output["generated_text_tokens"]),
                "mono_wav": mono_wav,
                "stereo_wav": stereo_wav,
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
    files = [resolve_repo_path(row[args.source_column]) for row in rows]
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
    finished = torch.zeros(batch_wavs.shape[0], dtype=torch.bool, device=codes.device)
    eos_id = int(text_tokenizer.eos_id())
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
            finished |= tokens[:, 0, 0] == eos_id
            if args.stop_on_eos and bool(finished.all().detach().cpu()):
                break
    elapsed = time.time() - start_time
    print(
        f"Generated batch {batch_start}-{batch_start + len(rows) - 1} "
        f"in {elapsed:.1f}s ({gen_duration / elapsed:.2f}x RT)"
    )

    if not output_text_tokens:
        raise RuntimeError("LM generation produced no output tokens; increase --gen-duration")
    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    text_outputs = decode_text_batch(batch_text_tokens, text_tokenizer, warn=args.text_only)
    if args.text_only:
        outputs = [
            {
                "wav": None,
                "text": text_output["text"],
                "eos_found": text_output["eos_found"],
                "generated_text_tokens": text_output["generated_text_tokens"],
            }
            for text_output in text_outputs
        ]
    else:
        batch_codes = torch.concat(output_audio_tokens, dim=-1)
        decoded_outputs = decode_outputs(batch_codes, batch_text_tokens, mimi, text_tokenizer)
        outputs = [
            {
                "wav": wav,
                "text": text,
                "eos_found": text_output["eos_found"],
                "generated_text_tokens": text_output["generated_text_tokens"],
            }
            for (wav, text), text_output in zip(decoded_outputs, text_outputs, strict=True)
        ]
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
        args,
    )


def write_predictions(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_audio",
                "reference_text",
                "prediction_text",
                "eos_found",
                "generated_text_tokens",
                "mono_wav",
                "stereo_wav",
                "text_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.limit <= 0 and not (args.ids or args.ids_file):
        raise ValueError("--limit must be positive unless --ids/--ids-file is set")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch, sphn, loaders, replace_all_linear_with_lora, safe_open, load_file = require_runtime_deps()
    device = check_device(torch, args.device)
    dtype = dtype_from_name(torch, args.dtype)
    seed_all(torch, args.seed)

    rows = select_eval_rows(read_eval_rows(args.pairs), args)
    validate_eval_rows(rows, args.source_column, args.reference_column, args.id_column)
    if not rows:
        raise RuntimeError(f"No rows selected from {args.pairs}")
    adapter = require_file(args.adapter, "LoRA adapter") if args.adapter else None
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
    if adapter is not None:
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
    else:
        print("Using base model without adapter.")
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
    metrics = score_records(records)
    metrics_path = resolve_repo_path(args.metrics_json) if args.metrics_json else out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = (
        f"nonempty={metrics['nonempty_predictions']}/{metrics['num_predictions']} "
        f"eos={metrics['eos_found']}/{metrics['num_predictions']} "
        f"exact={metrics['exact_matches']}/{metrics['num_predictions']} "
        f"wer={100 * metrics['wer']:.2f}%"
    )
    if "bleu" in metrics:
        summary += f" bleu={metrics['bleu']:.2f} chrf={metrics['chrf']:.2f}"
    print(f"Metrics: {summary}")
    print(f"Wrote metrics -> {repo_display_path(metrics_path)}")


if __name__ == "__main__":
    main()
