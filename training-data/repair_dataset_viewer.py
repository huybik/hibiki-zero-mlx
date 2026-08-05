"""Reserialize Hub Parquet files so the Hugging Face Dataset Viewer can read them.

The Viewer has a 300 MB scan limit. Older 500-row audio shards have one row
group of roughly 500 MB, so even fetching a single row fails. This script
replaces each file in place with 100-row groups and Parquet page indexes while
preserving its schema and rows.

Usage:
  python repair_dataset_viewer.py --pilot  # rewrite and validate one shard locally
  python repair_dataset_viewer.py --apply  # resumable in-place Hub repair
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from paths import DATASETS_DIR


REPO = "anquachdev/PhoMT-en-vi-speech"
DATA_DIR = "data"
ROWS_PER_ROW_GROUP = 100
SHARDS_PER_COMMIT = 5
STATE_FILE = "dataset-viewer-repair-state.json"
STAGING = DATASETS_DIR / "dataset_viewer_repair"


def list_parquet_shards(api: HfApi) -> list[str]:
    return sorted(
        path
        for path in api.list_repo_files(REPO, repo_type="dataset")
        if path.startswith(f"{DATA_DIR}/") and path.endswith(".parquet")
    )


def new_state() -> dict:
    return {
        "version": 1,
        "repo": REPO,
        "row_group_rows": ROWS_PER_ROW_GROUP,
        "done": [],
    }


def load_state() -> dict:
    try:
        path = hf_hub_download(REPO, STATE_FILE, repo_type="dataset", force_download=True)
    except EntryNotFoundError:
        return new_state()

    with Path(path).open(encoding="utf-8") as file:
        state = json.load(file)
    expected = new_state()
    for key in ("version", "repo", "row_group_rows"):
        if state.get(key) != expected[key]:
            raise ValueError(f"Unexpected repair state {key}: {state.get(key)!r}")
    state.setdefault("done", [])
    return state


def verify_rewrite(path: Path, expected_rows: int, expected_schema) -> None:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    if metadata.num_rows != expected_rows or parquet.schema_arrow != expected_schema:
        raise ValueError(f"Schema or row count changed while rewriting {path.name}")
    if any(
        metadata.row_group(index).num_rows > ROWS_PER_ROW_GROUP
        for index in range(metadata.num_row_groups)
    ):
        raise ValueError(f"Oversized row group in {path.name}")

    first_column = metadata.row_group(0).column(0)
    if not first_column.has_column_index or not first_column.has_offset_index:
        raise ValueError(f"Missing Parquet page index in {path.name}")


def rewrite_shard(path_in_repo: str, staging: Path) -> dict:
    started_at = time.perf_counter()
    source = Path(hf_hub_download(REPO, path_in_repo, repo_type="dataset"))
    table = pq.read_table(source)
    staging.mkdir(parents=True, exist_ok=True)
    destination = staging / Path(path_in_repo).name

    with tempfile.NamedTemporaryFile(dir=staging, suffix=".parquet", delete=False) as file:
        temporary = Path(file.name)
    try:
        pq.write_table(
            table,
            temporary,
            compression="NONE",
            row_group_size=ROWS_PER_ROW_GROUP,
            write_page_index=True,
        )
        verify_rewrite(temporary, table.num_rows, table.schema)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "path": path_in_repo,
        "staged": str(destination),
        "rows": table.num_rows,
        "row_groups": pq.ParquetFile(destination).metadata.num_row_groups,
        "old_size": source.stat().st_size,
        "new_size": destination.stat().st_size,
        "seconds": time.perf_counter() - started_at,
    }


def commit_batch(api: HfApi, batch: list[dict], state: dict) -> None:
    proposed_state = {**state, "done": [*state["done"], *(item["path"] for item in batch)]}
    operations = [
        *(CommitOperationAdd(path_in_repo=item["path"], path_or_fileobj=item["staged"]) for item in batch),
        CommitOperationAdd(
            path_in_repo=STATE_FILE,
            path_or_fileobj=json.dumps(proposed_state, indent=2, sort_keys=True).encode(),
        ),
    ]
    api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Repair Dataset Viewer layout ({len(batch)} Parquet shards)",
        commit_description=(
            "Rewrite audio Parquet files with 100-row groups and page indexes "
            "to stay below the Dataset Viewer scan limit."
        ),
    )
    state.update(proposed_state)
    for item in batch:
        Path(item["staged"]).unlink(missing_ok=True)


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="rewrite and validate one shard locally")
    mode.add_argument("--apply", action="store_true", help="rewrite every shard and commit each batch")
    args = parser.parse_args()

    api = HfApi()
    shards = list_parquet_shards(api)
    state = load_state()
    done = set(state["done"])
    todo = [path for path in shards if path not in done]
    print(f"{len(shards)} Parquet shards, {len(done)} repaired, {len(todo)} remaining")
    if not todo:
        return

    if args.pilot:
        result = rewrite_shard(todo[0], STAGING)
        print(
            f"PASS {result['path']}: {result['rows']} rows, {result['row_groups']} row groups, "
            f"{format_size(result['old_size'])} -> {format_size(result['new_size'])}"
        )
        Path(result["staged"]).unlink(missing_ok=True)
        return

    started_at = time.perf_counter()
    batch = []
    for index, path in enumerate(todo, start=1):
        result = rewrite_shard(path, STAGING)
        batch.append(result)
        print(
            f"{index}/{len(todo)} {path}: {result['rows']} rows, {result['row_groups']} row groups, "
            f"{result['seconds']:.0f}s",
            flush=True,
        )
        if len(batch) == SHARDS_PER_COMMIT or index == len(todo):
            commit_batch(api, batch, state)
            elapsed_hours = (time.perf_counter() - started_at) / 3600
            print(f"Committed {len(state['done'])}/{len(shards)} shards in {elapsed_hours:.2f}h")
            batch = []


if __name__ == "__main__":
    main()
