#!/usr/bin/env python
"""Generate free-running validation outputs with a base or full-SFT model.

Rows run as training validates them: every --pairs file in one pass, in length-sorted batches
(64 by default) reseeded per batch from --seed. Given the trainer's --eval-pairs in its order
and its --seed, the batches are the trainer's; each set's score depends on the rows it shares
batches with, so runs compare under the same --pairs list.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from training.finetune.eval_rows import (
    ids_from_args,
    read_eval_sets,
    select_eval_rows,
    validate_eval_rows,
)
from training.finetune.generation import GenerationConfig, translate_rows, write_predictions
from training.finetune.metrics import score_sets, score_summary
from training.finetune.model import (
    check_device,
    dtype_from_name,
    load_checkpoint_info,
    load_lm,
    run_tokenizer,
    seed_all,
)
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    repo_display_path,
    require_file,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    defaults = GenerationConfig()
    parser = argparse.ArgumentParser(
        description="Generate validation outputs with a base or full-SFT model."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        nargs="+",
        required=True,
        help="Pairs files (.jsonl) with id and duration_s columns and vi_audio and text_en "
        "(Vietnamese to English) or en_audio and text_vi (English to Vietnamese); each is scored as "
        "its own set.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional full-model .safetensors checkpoint."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, help="Default: the one the checkpoint (else the model weight) is on (its text_tokenizer metadata), in weights/.")
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16", help="Model dtype."
    )
    parser.add_argument("--limit", type=int, help="Max rows of each pairs file; all by default.")
    parser.add_argument("--batch-size", type=int, default=64, help="Rows per batch, as training validates.")
    parser.add_argument("--ids", default="", help="Comma-separated ids; bypasses --limit.")
    parser.add_argument("--ids-file", type=Path, help="Text file of ids, one per line; bypasses --limit.")
    parser.add_argument(
        "--gen-duration",
        type=float,
        default=defaults.gen_duration,
        help="Generation seconds. 0 = max source + --tail-s.",
    )
    parser.add_argument(
        "--tail-s", type=float, default=defaults.tail_s, help="Silence tail when gen-duration=0."
    )
    parser.add_argument("--audio-temp", type=float, default=defaults.audio_temp)
    parser.add_argument("--text-temp", type=float, default=defaults.text_temp)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--top-k-text", type=int, default=defaults.top_k_text)
    parser.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=defaults.stop_on_eos,
        help="Stop a batch once every row has emitted text EOS.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Write text sidecars and predictions.csv only; skip decoding, scoring and writing the "
        "generated speech.",
    )
    parser.add_argument(
        "--write-audio",
        action=argparse.BooleanOptionalAction,
        default=defaults.write_audio,
        help="Write each row's generated wavs (the voice scores need only the decode).",
    )
    parser.add_argument(
        "--laal",
        action="store_true",
        help="Also record each row's text LAAL (laal_s) and report each set's mean_laal_s, as the "
        "Hibiki-Zero paper computes it.",
    )
    parser.add_argument("--metrics-json", type=Path, help="Metrics JSON path; default <out-dir>/metrics.json.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="sft")
    parser.add_argument("--id-column", default=defaults.id_column)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = check_device(args.device)
    dtype = dtype_from_name(args.dtype)
    seed_all(args.seed)

    ids = ids_from_args(args.ids, args.ids_file)
    sets = read_eval_sets(args.pairs)
    rows = [row for set_rows in sets.values() for row in set_rows]
    # Every row, before selection: --ids must not pick one of two rows that share an id.
    validate_eval_rows(rows, args.id_column, duration_column="duration_s")
    if ids:
        rows = select_eval_rows(rows, ids, args.id_column)
    else:
        rows = [row for set_rows in sets.values() for row in set_rows[: args.limit]]
    checkpoint = require_file(args.checkpoint, "checkpoint") if args.checkpoint else None
    args.config_path = require_file(args.config_path, "config")
    if checkpoint is None:  # a checkpoint replaces the base weights, which are never read
        args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = run_tokenizer(args.tokenizer, checkpoint or args.model_weight)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = dataclasses.fields(GenerationConfig)
    generation = GenerationConfig(
        **{field.name: getattr(args, field.name) for field in fields}
    )
    checkpoint_info = load_checkpoint_info(args)

    print(f"Loading Mimi on {device} from {repo_display_path(args.mimi_weight)}")
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = load_lm(checkpoint_info, checkpoint, device, dtype)
    lm.eval()

    records = translate_rows(
        rows,
        generation,
        args.batch_size,
        mimi,
        lm,
        text_tokenizer,
        checkpoint_info,
        out_dir,
        args.seed,
    )
    metrics = score_sets(records, list(sets))

    predictions_path = out_dir / "predictions.csv"
    write_predictions(predictions_path, records)
    print(f"Wrote {len(records)} predictions -> {repo_display_path(predictions_path)}")
    metrics_path = resolve_repo_path(args.metrics_json) if args.metrics_json else out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, set_metrics in metrics.items():
        print(f"Metrics {name}: {score_summary(set_metrics)}")
    print(f"Wrote metrics -> {repo_display_path(metrics_path)}")


if __name__ == "__main__":
    main()
