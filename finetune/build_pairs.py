#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_PAIRS_DIR,
    VALID_SPLITS,
    read_fleurs_manifest,
    repo_display_path,
    require_dir,
    resolve_repo_path,
    write_pair_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic vi->en pair files from FLEURS manifests."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="FLEURS vi/en dataset root containing split/manifest.csv files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PAIRS_DIR,
        help="Output directory for pair files.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output file format.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows per split, 0 means all.")
    parser.add_argument(
        "--max-source-hours",
        type=float,
        default=0.0,
        help="Cap Vietnamese source hours per split, 0 means all.",
    )
    parser.add_argument(
        "--min-source-duration-s",
        type=float,
        default=0.0,
        help="Drop rows shorter than this Vietnamese duration.",
    )
    parser.add_argument(
        "--max-source-duration-s",
        type=float,
        default=0.0,
        help="Drop rows longer than this Vietnamese duration, 0 disables.",
    )
    parser.add_argument(
        "--val-subsets",
        type=int,
        nargs="*",
        default=[16, 128],
        help="Also write val{N}.jsonl held-out gate files (first N validation rows). Empty to disable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing pair files.")
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    total_source_s = 0.0
    max_source_s = args.max_source_hours * 3600.0 if args.max_source_hours else 0.0

    for row in rows:
        duration_s = float(row["vi_duration_s"])
        if duration_s < args.min_source_duration_s:
            continue
        if args.max_source_duration_s and duration_s > args.max_source_duration_s:
            continue
        if max_source_s and total_source_s + duration_s > max_source_s:
            break
        selected.append(row)
        total_source_s += duration_s
        if args.max_rows and len(selected) >= args.max_rows:
            break
    return selected


def main() -> None:
    args = parse_args()
    dataset_dir = require_dir(args.dataset_dir, "FLEURS dataset directory")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        manifest = dataset_dir / split / "manifest.csv"
        rows = read_fleurs_manifest(manifest, split)
        rows = select_rows(rows, args)

        out_path = out_dir / f"{split}.{args.format}"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing pair file: {out_path}")
        write_pair_file(rows, out_path, args.format)

        source_hours = sum(float(row["vi_duration_s"]) for row in rows) / 3600.0
        print(
            f"{split}: wrote {len(rows)} rows, {source_hours:.2f} source hours -> "
            f"{repo_display_path(out_path)}"
        )

        # Deterministic held-out gate subsets (first N rows), same selection as val16.
        if split == "validation" and args.val_subsets:
            for size in args.val_subsets:
                if size <= 0:
                    continue
                subset_path = out_dir / f"val{size}.{args.format}"
                if subset_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite existing subset: {subset_path}")
                write_pair_file(rows[:size], subset_path, args.format)
                print(f"  val{size}: wrote {min(size, len(rows))} rows -> {repo_display_path(subset_path)}")


if __name__ == "__main__":
    main()

