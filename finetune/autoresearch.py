#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/bin/python")
RESULTS_TSV = Path("finetune/autoresearch/results.tsv")
RUN_ROOT = Path("finetune/runs/autoresearch")
BASE_ADAPTER = Path(
    "finetune/runs/vn_lora_full_rank32_texthead_audioheads_textw5/"
    "adapter_step000362.safetensors"
)
ANCHOR_IDS = "213,211,245"

RESULT_FIELDS = [
    "timestamp",
    "commit",
    "branch",
    "trial",
    "status",
    "primary_metric",
    "val_bleu",
    "val_chrf",
    "val_nonempty_rate",
    "val_eos_rate",
    "val_exact",
    "seen_exact",
    "seen_eos_rate",
    "seen_audio_loss",
    "seen_text_loss",
    "validation_audio_loss",
    "validation_text_loss",
    "max_steps",
    "lr",
    "text_loss_weight",
    "replay_weight",
    "replay_ids",
    "run_dir",
    "description",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
    return str(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def current_commit() -> str:
    return git_output("rev-parse", "--short", "HEAD")


def current_branch() -> str:
    return git_output("rev-parse", "--abbrev-ref", "HEAD")


def read_json(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = repo_path(path)
    return read_json(resolved) if resolved.is_file() else {}


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if repo_path(path).is_file():
            return path
    return None


def rate(metrics: dict[str, Any], count_key: str) -> str:
    total = float(metrics.get("num_predictions") or 0)
    if total == 0:
        return ""
    return f"{float(metrics.get(count_key, 0)) / total:.6f}"


def metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, "")
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def command(args: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(args), flush=True)
    if dry_run:
        return
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def append_result(path: Path, row: dict[str, str]) -> None:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    exists = resolved.is_file()
    with resolved.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    print(f"appended result -> {rel(resolved)}")


def build_result_row(
    args: argparse.Namespace,
    run_dir: Path,
    seen_loss_path: Path | None,
    validation_loss_path: Path | None,
    seen_metrics_path: Path | None,
    val_metrics_path: Path | None,
) -> dict[str, str]:
    seen_loss = optional_json(seen_loss_path)
    validation_loss = optional_json(validation_loss_path)
    seen_metrics = optional_json(seen_metrics_path)
    val_metrics = optional_json(val_metrics_path)
    primary = metric(val_metrics, "chrf")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": current_commit(),
        "branch": current_branch(),
        "trial": args.trial,
        "status": args.status,
        "primary_metric": primary,
        "val_bleu": metric(val_metrics, "bleu"),
        "val_chrf": metric(val_metrics, "chrf"),
        "val_nonempty_rate": rate(val_metrics, "nonempty_predictions"),
        "val_eos_rate": rate(val_metrics, "eos_found"),
        "val_exact": metric(val_metrics, "exact_matches"),
        "seen_exact": metric(seen_metrics, "exact_matches"),
        "seen_eos_rate": rate(seen_metrics, "eos_found"),
        "seen_audio_loss": metric(seen_loss, "audio_loss"),
        "seen_text_loss": metric(seen_loss, "text_loss"),
        "validation_audio_loss": metric(validation_loss, "audio_loss"),
        "validation_text_loss": metric(validation_loss, "text_loss"),
        "max_steps": str(args.max_steps),
        "lr": str(args.lr),
        "text_loss_weight": str(args.text_loss_weight),
        "replay_weight": str(args.replay_weight),
        "replay_ids": args.replay_ids,
        "run_dir": rel(run_dir),
        "description": args.description,
    }


def train_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        "finetune/train_lora.py",
        "--cache-dir",
        "finetune/cache/train",
        "--out-dir",
        rel(run_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--epochs",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--init-adapter",
        rel(args.init_adapter),
        "--lora-rank",
        "32",
        "--lora-scaling",
        "2",
        "--train-text-head",
        "--train-audio-heads",
        "--lr",
        str(args.lr),
        "--audio-loss-weight",
        str(args.audio_loss_weight),
        "--text-loss-weight",
        str(args.text_loss_weight),
        "--log-every",
        str(args.log_every),
        "--save-every",
        "0",
        "--mps-empty-cache-every",
        "10",
    ]
    if args.replay_ids:
        cmd.extend(
            [
                "--replay-ids",
                args.replay_ids,
                "--replay-weight",
                str(args.replay_weight),
                "--replay-seed",
                str(args.replay_seed),
            ]
        )
    return cmd


def validate_command(
    args: argparse.Namespace,
    cache_dir: str,
    adapter: Path,
    out_json: Path,
    batch_size: int,
) -> list[str]:
    return [
        str(args.python),
        "finetune/validate_lora.py",
        "--cache-dir",
        cache_dir,
        "--adapter",
        rel(adapter),
        "--out-json",
        rel(out_json),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(batch_size),
        "--text-loss-weight",
        str(args.text_loss_weight),
    ]


def eval_command(
    args: argparse.Namespace,
    pairs: str,
    adapter: Path,
    out_dir: Path,
    metrics_json: Path,
    batch_size: int,
    tag: str,
    ids: str = "",
    limit: int = 0,
) -> list[str]:
    cmd = [
        str(args.python),
        "finetune/eval_lora.py",
        "--pairs",
        pairs,
        "--adapter",
        rel(adapter),
        "--out-dir",
        rel(out_dir),
        "--metrics-json",
        rel(metrics_json),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(batch_size),
        "--text-only",
        "--gen-duration",
        "0",
        "--tail-s",
        "8",
        "--text-temp",
        str(args.text_temp),
        "--audio-temp",
        "0.8",
        "--seed",
        str(args.seed),
        "--tag",
        tag,
    ]
    if ids:
        cmd.extend(["--ids", ids])
    else:
        cmd.extend(["--limit", str(limit)])
    return cmd


def copy_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def run_trial(args: argparse.Namespace) -> None:
    if args.adapter is not None and not args.skip_train:
        raise ValueError("--adapter is only valid with --skip-train")
    run_dir = RUN_ROOT / args.trial
    adapter = args.adapter or run_dir / f"adapter_step{args.max_steps:06d}.safetensors"
    seen_loss = run_dir / "seen_first3_loss.json"
    validation_loss = run_dir / "validation_loss.json"
    seen_metrics = run_dir / "eval_seen3_text_metrics.json"
    val_metrics = run_dir / f"eval_val{args.eval_limit}_text_metrics.json"

    if not args.skip_train:
        command(train_command(args, run_dir), args.dry_run)
    command(
        validate_command(args, "finetune/cache/seen_first3", adapter, seen_loss, 3),
        args.dry_run,
    )
    command(
        validate_command(args, "finetune/cache/validation", adapter, validation_loss, 8),
        args.dry_run,
    )
    command(
        eval_command(
            args,
            "finetune/pairs/train.jsonl",
            adapter,
            run_dir / "eval_seen3_text",
            seen_metrics,
            3,
            args.trial,
            ids=ANCHOR_IDS,
        ),
        args.dry_run,
    )
    command(
        eval_command(
            args,
            "finetune/pairs/validation.jsonl",
            adapter,
            run_dir / f"eval_val{args.eval_limit}_text",
            val_metrics,
            args.eval_batch_size,
            args.trial,
            limit=args.eval_limit,
        ),
        args.dry_run,
    )
    if not args.dry_run:
        row = build_result_row(
            args,
            run_dir,
            seen_loss,
            validation_loss,
            seen_metrics,
            val_metrics,
        )
        append_result(args.results_tsv, row)


def run_staged_trial(args: argparse.Namespace) -> None:
    run_dir = RUN_ROOT / args.trial
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage1_adapter = stage1_dir / f"adapter_step{args.stage1_steps:06d}.safetensors"
    adapter = stage2_dir / f"adapter_step{args.stage2_steps:06d}.safetensors"
    seen_loss = run_dir / "seen_first3_loss.json"
    validation_loss = run_dir / "validation_loss.json"
    seen_metrics = run_dir / "eval_seen3_text_metrics.json"
    val_metrics = run_dir / f"eval_val{args.eval_limit}_text_metrics.json"

    stage1_args = copy_args(
        args,
        max_steps=args.stage1_steps,
        init_adapter=args.init_adapter,
        replay_weight=args.stage1_replay_weight,
        text_loss_weight=args.stage1_text_loss_weight,
    )
    stage2_args = copy_args(
        args,
        max_steps=args.stage2_steps,
        init_adapter=stage1_adapter,
        replay_weight=args.stage2_replay_weight,
        text_loss_weight=args.stage2_text_loss_weight,
    )
    eval_args = copy_args(args, text_loss_weight=args.stage2_text_loss_weight)

    command(train_command(stage1_args, stage1_dir), args.dry_run)
    command(train_command(stage2_args, stage2_dir), args.dry_run)
    command(
        validate_command(eval_args, "finetune/cache/seen_first3", adapter, seen_loss, 3),
        args.dry_run,
    )
    command(
        validate_command(eval_args, "finetune/cache/validation", adapter, validation_loss, 8),
        args.dry_run,
    )
    command(
        eval_command(
            args,
            "finetune/pairs/train.jsonl",
            adapter,
            run_dir / "eval_seen3_text",
            seen_metrics,
            3,
            args.trial,
            ids=ANCHOR_IDS,
        ),
        args.dry_run,
    )
    command(
        eval_command(
            args,
            "finetune/pairs/validation.jsonl",
            adapter,
            run_dir / f"eval_val{args.eval_limit}_text",
            val_metrics,
            args.eval_batch_size,
            args.trial,
            limit=args.eval_limit,
        ),
        args.dry_run,
    )
    if not args.dry_run:
        row_args = copy_args(
            args,
            max_steps=args.stage1_steps + args.stage2_steps,
            text_loss_weight=(
                f"{args.stage1_text_loss_weight:g}->{args.stage2_text_loss_weight:g}"
            ),
            replay_weight=f"{args.stage1_replay_weight:g}->{args.stage2_replay_weight:g}",
        )
        row = build_result_row(
            row_args,
            run_dir,
            seen_loss,
            validation_loss,
            seen_metrics,
            val_metrics,
        )
        append_result(args.results_tsv, row)


def record_existing(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    seen_loss = args.seen_loss or first_existing(
        [
            run_dir / "seen_first3_loss.json",
            run_dir / "seen_first3_loss_step000120.json",
            run_dir / "seen_first3_loss_step000362.json",
        ]
    )
    validation_loss = args.validation_loss or first_existing(
        [
            run_dir / "validation_loss.json",
            run_dir / "validation_loss_step000120.json",
            run_dir / "validation_loss_step000362.json",
        ]
    )
    seen_metrics = args.seen_metrics or first_existing(
        [
            run_dir / "eval_seen3_text_metrics.json",
            run_dir / "eval_seen_first3_textonly" / "metrics.json",
        ]
    )
    val_metrics = args.val_metrics or first_existing(
        [
            run_dir / "eval_val16_text_metrics.json",
            run_dir / "eval_validation" / "metrics.json",
        ]
    )
    row = build_result_row(
        args,
        run_dir,
        seen_loss,
        validation_loss,
        seen_metrics,
        val_metrics,
    )
    append_result(args.results_tsv, row)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--status", default="candidate")
    parser.add_argument("--results-tsv", type=Path, default=RESULTS_TSV)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=5.0)
    parser.add_argument("--replay-ids", default=ANCHOR_IDS)
    parser.add_argument("--replay-weight", type=float, default=300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed vi->en LoRA trial runner for small AutoResearch loops."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-trial")
    add_common_args(run_parser)
    run_parser.add_argument("--python", type=Path, default=PYTHON)
    run_parser.add_argument("--init-adapter", type=Path, default=BASE_ADAPTER)
    run_parser.add_argument(
        "--adapter",
        type=Path,
        help="Existing adapter to evaluate when --skip-train is set.",
    )
    run_parser.add_argument("--device", default="mps")
    run_parser.add_argument("--dtype", default="bfloat16")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--grad-accum-steps", type=int, default=2)
    run_parser.add_argument("--log-every", type=int, default=10)
    run_parser.add_argument("--replay-seed", type=int, default=0)
    run_parser.add_argument("--eval-limit", type=int, default=16)
    run_parser.add_argument("--eval-batch-size", type=int, default=4)
    run_parser.add_argument("--text-temp", type=float, default=0.0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--skip-train", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    staged_parser = subparsers.add_parser("run-staged-trial")
    add_common_args(staged_parser)
    staged_parser.add_argument("--python", type=Path, default=PYTHON)
    staged_parser.add_argument("--init-adapter", type=Path, default=BASE_ADAPTER)
    staged_parser.add_argument("--device", default="mps")
    staged_parser.add_argument("--dtype", default="bfloat16")
    staged_parser.add_argument("--batch-size", type=int, default=1)
    staged_parser.add_argument("--grad-accum-steps", type=int, default=2)
    staged_parser.add_argument("--log-every", type=int, default=10)
    staged_parser.add_argument("--replay-seed", type=int, default=0)
    staged_parser.add_argument("--eval-limit", type=int, default=16)
    staged_parser.add_argument("--eval-batch-size", type=int, default=4)
    staged_parser.add_argument("--text-temp", type=float, default=0.0)
    staged_parser.add_argument("--seed", type=int, default=42)
    staged_parser.add_argument("--stage1-steps", type=int, default=40)
    staged_parser.add_argument("--stage1-replay-weight", type=float, default=300.0)
    staged_parser.add_argument("--stage1-text-loss-weight", type=float, default=5.0)
    staged_parser.add_argument("--stage2-steps", type=int, default=40)
    staged_parser.add_argument("--stage2-replay-weight", type=float, default=100.0)
    staged_parser.add_argument("--stage2-text-loss-weight", type=float, default=5.0)
    staged_parser.add_argument("--dry-run", action="store_true")

    record_parser = subparsers.add_parser("record-existing")
    add_common_args(record_parser)
    record_parser.add_argument("--run-dir", type=Path, required=True)
    record_parser.add_argument("--seen-loss", type=Path)
    record_parser.add_argument("--validation-loss", type=Path)
    record_parser.add_argument("--seen-metrics", type=Path)
    record_parser.add_argument("--val-metrics", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-trial":
        run_trial(args)
    elif args.command == "run-staged-trial":
        run_staged_trial(args)
    elif args.command == "record-existing":
        record_existing(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
