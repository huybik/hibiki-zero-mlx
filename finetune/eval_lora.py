#!/usr/bin/env python
"""Generate greedy validation outputs with the base model or a LoRA adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune import common  # noqa: E402
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
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16", help="Model/LoRA dtype."
    )
    parser.add_argument("--limit", type=int, default=1, help="Max validation pairs to generate.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ids", default="", help="Comma-separated ids in that order; bypasses --limit.")
    parser.add_argument("--ids-file", type=Path, help="Text file of ids, one per line; bypasses --limit.")
    parser.add_argument(
        "--gen-duration", type=float, default=0.0, help="Generation seconds. 0 = max source + --tail-s."
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
        help="Write text sidecars and predictions.csv only; skip generated wav decode/write.",
    )
    parser.add_argument("--metrics-json", type=Path, help="Metrics JSON path; default <out-dir>/metrics.json.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="lora")
    parser.add_argument("--source-column", default="vi_audio")
    parser.add_argument("--reference-column", default="text_en")
    parser.add_argument("--id-column", default="id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0 and not (args.ids or args.ids_file):
        raise ValueError("--limit must be positive unless --ids/--ids-file is set")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = common.check_device(args.device)
    dtype = common.dtype_from_name(args.dtype)
    common.seed_all(args.seed)

    ids = common.ids_from_args(args.ids, args.ids_file)
    rows = common.select_eval_rows(common.read_eval_rows(args.pairs), ids, args.id_column, args.limit)
    common.validate_eval_rows(rows, args.source_column, args.reference_column, args.id_column)
    if not rows:
        raise RuntimeError(f"No rows selected from {args.pairs}")
    adapter = require_file(args.adapter, "LoRA adapter") if args.adapter else None
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = common.load_checkpoint_info(args)
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
        common.load_main_lora(lm, adapter, device, dtype)
    else:
        print("Using base model without adapter.")
    lm.eval()

    records, metrics = common.run_greedy_eval(
        rows, args, args.batch_size, mimi, lm, text_tokenizer, checkpoint_info, out_dir
    )

    predictions_path = out_dir / "predictions.csv"
    common.write_predictions(predictions_path, records)
    print(f"Wrote {len(records)} predictions -> {repo_display_path(predictions_path)}")
    metrics_path = resolve_repo_path(args.metrics_json) if args.metrics_json else out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = (
        f"nonempty={metrics['nonempty_predictions']}/{metrics['num_predictions']} "
        f"eos={metrics['eos_found']}/{metrics['num_predictions']} "
        f"exact={metrics['exact_matches']}/{metrics['num_predictions']} "
        f"wer={100 * metrics['wer']:.2f}% overlong={metrics['overlong_predictions']} "
        f"repeat4={metrics['repeated_4gram_predictions']}"
    )
    if "bleu" in metrics:
        summary += f" bleu={metrics['bleu']:.2f} chrf={metrics['chrf']:.2f}"
    print(f"Metrics: {summary}")
    print(f"Wrote metrics -> {repo_display_path(metrics_path)}")


if __name__ == "__main__":
    main()
