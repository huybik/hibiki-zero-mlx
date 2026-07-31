from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

# hf_xet reads this at upload time. It is harmless when hf_xet is not installed.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# Claim this before importing pipeline, whose QUIET_HF_DOWNLOADS would set it to 1.
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

from datasets import Audio, Dataset, DatasetInfo, load_dataset
from datasets.info import DatasetInfosDict
from datasets.splits import SplitDict, SplitInfo
from datasets.table import embed_table_storage
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import CommitOperationAdd, DatasetCard, DatasetCardData, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from paths import DATASETS_DIR

import pipeline


# =========================
# Config
# =========================

DATASET_REPO = "anquachdev/PhoMT-en-vi-speech"
PUSH_TO_HUB = True
PRIVATE = False
CHECK_REMOTE_UPLOAD_KEYS = False

CONFIG_NAME = "default"
SPLIT = "train"
DATA_DIR = "data"
DATASET_CARD_FILE = "README.md"
UPLOAD_STATE_FILE = "upload-state.json"
UPLOAD_KEY_COLUMNS = ("en", "vi")

# A sampled pair is about 800 KB, so 500 rows produces a roughly 400 MB shard.
# datasets >= 5 picks parquet compression itself (audio columns uncompressed for
# Xet dedup); passing compression kwargs conflicts with its ParquetWriter call.
ROWS_PER_SHARD = 500
# The Hub rate-limits repository commits to 128/hour; batching shards per commit
# keeps a fast upload well under it (~30 commits/hour at 5 x ~400 MB shards).
SHARDS_PER_COMMIT = 5
DURATION_WORKERS = 8
DURATION_PROGRESS_INTERVAL = 2_000

# Env overrides let a mid-generation upload read frozen manifest snapshots so
# the validated row set and the uploaded row set are identical.
VI_MANIFEST = Path(
    os.environ.get("UPLOAD_VI_MANIFEST", DATASETS_DIR / "vieNeu" / "outputs" / "vi" / "manifest.csv")
)
EN_MANIFEST = Path(
    os.environ.get("UPLOAD_EN_MANIFEST", DATASETS_DIR / "english" / "outputs" / "en" / "manifest.csv")
)
LOCAL_SAVE_DIR = DATASETS_DIR / "phomt-en-vi-speech"

# Keep pairs with broadly similar lengths. The current generated data has slow
# English outliers, mostly af_nicole at Kokoro speed 1.0.
MIN_DURATION_RATIO = 0.4
MAX_DURATION_RATIO = 1.8


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


def load_manifests() -> tuple[dict[int, dict], dict[int, dict], list[int]]:
    vi_rows = read_manifest(VI_MANIFEST)
    en_rows = read_manifest(EN_MANIFEST)
    shared_indexes = sorted(set(vi_rows) & set(en_rows))
    if not shared_indexes:
        raise ValueError("No shared indexes found in the Vietnamese and English manifests.")
    return vi_rows, en_rows, shared_indexes


def print_next_start_index_hint(shared_indexes: list[int]) -> None:
    print("Next batch config hint:")
    print(f"  Set START_INDEX = {max(shared_indexes) + 1} in training-data/pipeline.py")
    print("  Keep N_SAMPLES as the batch size you want to generate next.")


def get_audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("soundfile is required to check audio durations.") from error

    info = sf.info(path)
    return info.frames / info.samplerate


def get_manifest_duration(row: dict, audio_path: Path) -> float:
    duration = row.get("duration_s")
    return float(duration) if duration not in (None, "") else get_audio_duration_seconds(audio_path)


def get_pair_durations(candidate: tuple) -> tuple[float, float]:
    _index, en_row, vi_row, audio_en, audio_vi = candidate
    return (
        get_manifest_duration(en_row, audio_en),
        get_manifest_duration(vi_row, audio_vi),
    )


def iter_pair_durations(candidates: list[tuple]):
    if DURATION_WORKERS <= 1:
        yield from map(get_pair_durations, candidates)
        return

    with ThreadPoolExecutor(max_workers=DURATION_WORKERS) as executor:
        yield from executor.map(get_pair_durations, candidates)


def get_upload_key(row: dict) -> tuple[str, ...]:
    return tuple(row[column] for column in UPLOAD_KEY_COLUMNS)


def pair_voices_current(index: int, vi_row: dict, en_row: dict) -> bool:
    """Rows generated before a voice-assignment change are pending regeneration;
    uploading them would freeze the old voice on the Hub."""
    target_gender = pipeline.pick_target_gender(index, pipeline.SEED)
    vi_voice = pipeline.pick_row_voice("vi", index, pipeline.SEED, pipeline.VI_VOICE, target_gender)
    en_voice = pipeline.pick_row_voice("en", index, pipeline.SEED, pipeline.EN_VOICE, target_gender)
    return (
        vi_row["voice"] == vi_voice
        and en_row["voice"] == en_voice
        and pipeline.speed_matches(
            en_row.get("speed"), pipeline.get_voice_speed(pipeline.EN_TTS, en_voice)
        )
    )


def build_rows(
    vi_rows: dict[int, dict],
    en_rows: dict[int, dict],
    shared_indexes: list[int],
    *,
    skip_keys: set[tuple[str, ...]] | None = None,
    skip_source_indexes: set[int] | None = None,
    assumed_uploaded_before_index: int | None = None,
) -> list[dict]:
    skip_keys = skip_keys or set()
    skip_source_indexes = skip_source_indexes or set()
    candidates = []
    skipped_state = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_stale_voice = 0

    for index in shared_indexes:
        if index in skip_source_indexes or (
            assumed_uploaded_before_index is not None
            and index < assumed_uploaded_before_index
        ):
            skipped_state += 1
            continue

        vi_row = vi_rows[index]
        en_row = en_rows[index]
        if (en_row["text"], vi_row["text"]) in skip_keys:
            skipped_existing += 1
            continue

        if not pair_voices_current(index, vi_row, en_row):
            skipped_stale_voice += 1
            continue

        audio_en = Path(en_row["audio_path"])
        audio_vi = Path(vi_row["audio_path"])
        if not audio_en.exists() or not audio_vi.exists():
            skipped_missing += 1
            continue

        candidates.append((index, en_row, vi_row, audio_en, audio_vi))

    if skipped_state:
        print(f"Skipped {skipped_state} source indexes recorded in upload state.")
    if skipped_existing:
        print(f"Skipped {skipped_existing} already-uploaded text pairs.")
    if skipped_stale_voice:
        print(
            f"Skipped {skipped_stale_voice} pairs whose voices predate the current "
            "assignment (pending regeneration)."
        )
    if skipped_missing:
        print(f"Skipped {skipped_missing} pairs with missing audio.")

    if not candidates:
        return []

    cached_durations = sum(
        en_row.get("duration_s") not in (None, "")
        and vi_row.get("duration_s") not in (None, "")
        for _index, en_row, vi_row, _audio_en, _audio_vi in candidates
    )
    print(
        f"Checking {len(candidates)} duration pairs with {DURATION_WORKERS} workers "
        f"({cached_durations} already cached in manifests)..."
    )

    rows = []
    skipped_duration = 0
    started_at = time.perf_counter()
    for position, (candidate, durations) in enumerate(
        zip(candidates, iter_pair_durations(candidates)), start=1
    ):
        index, en_row, vi_row, audio_en, audio_vi = candidate
        en_duration_s, vi_duration_s = durations
        duration_ratio = en_duration_s / vi_duration_s if vi_duration_s else 0.0

        if not MIN_DURATION_RATIO <= duration_ratio <= MAX_DURATION_RATIO:
            skipped_duration += 1
        else:
            rows.append(
                {
                    "_source_index": index,
                    "en": en_row["text"],
                    "vi": vi_row["text"],
                    "audio_en": str(audio_en),
                    "audio_vi": str(audio_vi),
                    "duration_en_s": en_duration_s,
                    "duration_vi_s": vi_duration_s,
                    "duration_ratio_en_vi": duration_ratio,
                }
            )

        if position % DURATION_PROGRESS_INTERVAL == 0 or position == len(candidates):
            elapsed = time.perf_counter() - started_at
            print(
                f"  checked {position}/{len(candidates)} pairs "
                f"({position / max(elapsed, 0.001):.0f} pairs/s)",
                flush=True,
            )

    if skipped_duration:
        print(
            f"Skipped {skipped_duration} duration-mismatched pairs "
            f"(allowed EN/VI ratio: {MIN_DURATION_RATIO:g}-{MAX_DURATION_RATIO:g})."
        )
    if not rows and not (skipped_state or skipped_existing):
        raise ValueError("No eligible paired rows remain after validation.")
    return rows


def build_dataset(rows: list[dict]) -> Dataset:
    public_rows = [
        {column: value for column, value in row.items() if not column.startswith("_")}
        for row in rows
    ]
    dataset = Dataset.from_list(public_rows)
    for column in ("audio_en", "audio_vi"):
        dataset = dataset.cast_column(column, Audio())
    return dataset


def embed_external_files(dataset: Dataset) -> Dataset:
    dataset_format = dataset.format
    embedded = dataset.with_format("arrow").map(
        embed_table_storage,
        batched=True,
        batch_size=ROWS_PER_SHARD,
        keep_in_memory=True,
    )
    return embedded.with_format(**dataset_format)


def list_existing_split_files(api: HfApi) -> list[str]:
    try:
        repo_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    except RepositoryNotFoundError:
        return []

    split_prefix = f"{DATA_DIR}/{SPLIT}-"
    return sorted(
        path
        for path in repo_files
        if path.startswith(split_prefix) and path.endswith(".parquet")
    )


def scan_remote_upload_keys() -> tuple[set[tuple[str, ...]], int]:
    print("Scanning existing Hub rows for resume keys...")
    existing_dataset = load_dataset(
        DATASET_REPO,
        split=SPLIT,
        columns=list(UPLOAD_KEY_COLUMNS),
        streaming=True,
    )
    existing_keys = set()
    existing_num_rows = 0
    for row in existing_dataset:
        existing_keys.add(get_upload_key(row))
        existing_num_rows += 1
        if existing_num_rows % 5_000 == 0:
            print(f"  scanned {existing_num_rows} existing rows...", flush=True)
    return existing_keys, existing_num_rows


def load_dataset_card(api: HfApi) -> DatasetCard | None:
    try:
        card_path = api.hf_hub_download(
            DATASET_REPO,
            DATASET_CARD_FILE,
            repo_type="dataset",
            force_download=True,
        )
    except EntryNotFoundError:
        return None
    return DatasetCard.load(card_path)


def get_existing_num_rows(dataset_card: DatasetCard | None) -> int:
    if dataset_card is None:
        return 0
    dataset_info = DatasetInfosDict.from_dataset_card_data(dataset_card.data).get(CONFIG_NAME)
    if dataset_info is None or dataset_info.splits is None:
        return 0
    return dataset_info.splits.get(SPLIT, SplitInfo()).num_examples or 0


def new_upload_state() -> dict:
    return {
        "version": 1,
        "dataset_repo": DATASET_REPO,
        "assumed_uploaded_before_index": None,
        "uploaded_source_ranges": [],
        "shards": [],
    }


def load_upload_state(api: HfApi) -> tuple[dict, bool]:
    try:
        state_path = api.hf_hub_download(
            DATASET_REPO,
            UPLOAD_STATE_FILE,
            repo_type="dataset",
            force_download=True,
        )
    except EntryNotFoundError:
        return new_upload_state(), False

    with Path(state_path).open("r", encoding="utf-8") as file:
        state = json.load(file)
    if state.get("version") != 1 or state.get("dataset_repo") != DATASET_REPO:
        raise ValueError(f"Unsupported or mismatched Hub upload state: {state}")
    state.setdefault("uploaded_source_ranges", [])
    state.setdefault("shards", [])
    state.setdefault("assumed_uploaded_before_index", None)
    return state, True


def ranges_to_indexes(ranges: list[list[int]]) -> set[int]:
    return {
        index
        for start_index, end_index in ranges
        for index in range(start_index, end_index + 1)
    }


def indexes_to_ranges(indexes: set[int]) -> list[list[int]]:
    if not indexes:
        return []

    ordered = sorted(indexes)
    ranges = []
    start_index = previous_index = ordered[0]
    for index in ordered[1:]:
        if index != previous_index + 1:
            ranges.append([start_index, previous_index])
            start_index = index
        previous_index = index
    ranges.append([start_index, previous_index])
    return ranges


def get_dataset_info_for_card(
    dataset_card_data: DatasetCardData,
    dataset: Dataset,
    existing_num_rows: int,
    uploaded_size: int,
    dataset_nbytes: int,
) -> DatasetInfo:
    dataset_info = DatasetInfosDict.from_dataset_card_data(dataset_card_data).get(CONFIG_NAME)
    dataset_name = DATASET_REPO.split("/")[-1]

    if dataset_info is None:
        dataset_info = dataset.info.copy()
        dataset_info.splits = SplitDict()
        previous_download_size = previous_dataset_size = previous_split_bytes = 0
    else:
        dataset_info = dataset_info.copy()
        previous_download_size = dataset_info.download_size or 0
        previous_dataset_size = dataset_info.dataset_size or 0
        previous_split = (
            dataset_info.splits.get(SPLIT, SplitInfo())
            if dataset_info.splits is not None
            else SplitInfo()
        )
        previous_split_bytes = previous_split.num_bytes or 0

    if dataset_info.features is not None and dataset_info.features != dataset.info.features:
        raise ValueError(
            "New dataset features do not match the existing Hub dataset "
            f"features: {dataset.info.features} != {dataset_info.features}"
        )

    dataset_info.features = dataset.info.features
    dataset_info.config_name = CONFIG_NAME
    dataset_info.download_checksums = None
    dataset_info.download_size = previous_download_size + uploaded_size
    dataset_info.dataset_size = previous_dataset_size + dataset_nbytes
    dataset_info.size_in_bytes = dataset_info.download_size + dataset_info.dataset_size
    dataset_info.splits = dataset_info.splits or SplitDict()
    dataset_info.splits[SPLIT] = SplitInfo(
        SPLIT,
        num_bytes=previous_split_bytes + dataset_nbytes,
        num_examples=existing_num_rows + len(dataset),
        dataset_name=dataset_name,
    )
    return dataset_info


def build_dataset_card_operation(
    dataset_card: DatasetCard | None,
    dataset: Dataset,
    existing_num_rows: int,
    uploaded_size: int,
    dataset_nbytes: int,
) -> tuple[DatasetCard, CommitOperationAdd]:
    card_data = dataset_card.data if dataset_card is not None else DatasetCardData()
    dataset_info = get_dataset_info_for_card(
        card_data,
        dataset,
        existing_num_rows,
        uploaded_size,
        dataset_nbytes,
    )
    DatasetInfosDict({CONFIG_NAME: dataset_info}).to_dataset_card_data(card_data)
    MetadataConfigs(
        {CONFIG_NAME: {"data_files": [{"split": SPLIT, "path": f"{DATA_DIR}/{SPLIT}-*"}]}}
    ).to_dataset_card_data(card_data)

    if dataset_card is None:
        dataset_card = DatasetCard(f"---\n{card_data}\n---\n")
    operation = CommitOperationAdd(
        path_in_repo=DATASET_CARD_FILE,
        path_or_fileobj=str(dataset_card).encode("utf-8"),
    )
    return dataset_card, operation


def build_upload_state_operation(state: dict) -> CommitOperationAdd:
    return CommitOperationAdd(
        path_in_repo=UPLOAD_STATE_FILE,
        path_or_fileobj=json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
    )


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000:.1f} MB"


def build_shard(
    rows: list[dict],
    shard_number: int,
    total_shards: int,
    run_id: str,
    temp_dir_path: Path,
) -> tuple[Dataset, Path, str, int, int, set[int]]:
    shard_rows = rows[shard_number * ROWS_PER_SHARD : (shard_number + 1) * ROWS_PER_SHARD]
    source_indexes = {int(row["_source_index"]) for row in shard_rows}
    filename = (
        f"{SPLIT}-append-{run_id}-{shard_number:05d}-of-{total_shards:05d}.parquet"
    )
    local_path = temp_dir_path / filename
    path_in_repo = f"{DATA_DIR}/{filename}"

    print(
        f"Shard {shard_number + 1}/{total_shards}: building {len(shard_rows)} rows "
        f"(indexes {min(source_indexes)}-{max(source_indexes)})...",
        flush=True,
    )
    dataset = build_dataset(shard_rows)
    embedded_dataset = embed_external_files(dataset)
    dataset_nbytes = embedded_dataset._estimate_nbytes()
    embedded_dataset.to_parquet(local_path)
    uploaded_size = local_path.stat().st_size
    return dataset, local_path, path_in_repo, dataset_nbytes, uploaded_size, source_indexes


def append_rows_to_hub(
    rows: list[dict],
    api: HfApi,
    dataset_card: DatasetCard | None,
    existing_num_rows: int,
    upload_state: dict,
) -> None:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    total_shards = math.ceil(len(rows) / ROWS_PER_SHARD)
    uploaded_indexes = ranges_to_indexes(upload_state["uploaded_source_ranges"])
    overall_started_at = time.perf_counter()

    # One builder thread packs shard N+1 while the main thread uploads shard N;
    # the network transfer is the bottleneck, so packing rides along for free.
    with tempfile.TemporaryDirectory(prefix="phomt_hf_append_") as temp_dir, ThreadPoolExecutor(
        max_workers=1
    ) as builder:
        temp_dir_path = Path(temp_dir)
        next_shard = builder.submit(build_shard, rows, 0, total_shards, run_id, temp_dir_path)
        pending = []
        group_started_at = time.perf_counter()
        for shard_number in range(total_shards):
            pending.append(next_shard.result())
            if shard_number + 1 < total_shards:
                next_shard = builder.submit(
                    build_shard, rows, shard_number + 1, total_shards, run_id, temp_dir_path
                )
            if len(pending) < SHARDS_PER_COMMIT and shard_number + 1 < total_shards:
                continue

            uploaded_at = datetime.now(timezone.utc).isoformat()
            group_indexes = set().union(*(built[5] for built in pending))
            group_rows = sum(len(built[0]) for built in pending)
            proposed_state = dict(upload_state)
            proposed_state["uploaded_source_ranges"] = indexes_to_ranges(
                uploaded_indexes | group_indexes
            )
            proposed_state["shards"] = [
                *upload_state["shards"],
                *(
                    {
                        "path": path_in_repo,
                        "rows": len(dataset),
                        "source_index_min": min(source_indexes),
                        "source_index_max": max(source_indexes),
                        "uploaded_at": uploaded_at,
                    }
                    for dataset, _, path_in_repo, _, _, source_indexes in pending
                ),
            ]
            proposed_state["updated_at"] = uploaded_at

            for dataset, _, _, dataset_nbytes, uploaded_size, _ in pending:
                dataset_card, card_operation = build_dataset_card_operation(
                    dataset_card,
                    dataset,
                    existing_num_rows,
                    uploaded_size,
                    dataset_nbytes,
                )
                existing_num_rows += len(dataset)
            operations = [
                *(
                    CommitOperationAdd(path_in_repo=built[2], path_or_fileobj=str(built[1]))
                    for built in pending
                ),
                card_operation,
                build_upload_state_operation(proposed_state),
            ]

            group_size = sum(built[4] for built in pending)
            print(
                f"Shards {shard_number + 2 - len(pending)}-{shard_number + 1}/{total_shards}: "
                f"uploading {format_size(group_size)} in one commit...",
                flush=True,
            )
            api.create_commit(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                operations=operations,
                commit_message=(
                    f"Append {SPLIT} shards {shard_number + 2 - len(pending)}-"
                    f"{shard_number + 1}/{total_shards} ({group_rows} rows)"
                ),
                commit_description=(
                    "Atomic resumable upload: append Parquet shards and refresh "
                    "dataset metadata plus source-index upload state."
                ),
            )

            upload_state = proposed_state
            uploaded_indexes.update(group_indexes)
            for _, local_path, _, _, _, _ in pending:
                local_path.unlink(missing_ok=True)
            elapsed = time.perf_counter() - group_started_at
            print(
                f"Shards {shard_number + 2 - len(pending)}-{shard_number + 1}/{total_shards}: "
                f"committed in {elapsed / 60:.1f} minutes. Temporary shards removed.",
                flush=True,
            )
            pending = []
            group_started_at = time.perf_counter()

    elapsed = time.perf_counter() - overall_started_at
    print(
        f"Appended {len(rows)} rows in {total_shards} shard(s) to "
        f"https://huggingface.co/datasets/{DATASET_REPO} "
        f"in {elapsed / 3600:.2f} hours."
    )


def print_transfer_backend() -> None:
    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        print("Warning: HF_HUB_DISABLE_XET=1 disables the faster Xet upload backend.")
    elif find_spec("hf_xet") is None:
        print("Performance tip: run `uv add hf-xet` before this large upload.")
    else:
        print("hf_xet high-performance uploads enabled.")


def upload_to_hub() -> None:
    print_transfer_backend()
    vi_rows, en_rows, shared_indexes = load_manifests()
    print(
        f"Local batch: {len(shared_indexes)} shared indexes "
        f"({shared_indexes[0]}-{shared_indexes[-1]})."
    )

    api = HfApi()
    api.create_repo(
        DATASET_REPO,
        repo_type="dataset",
        private=PRIVATE,
        exist_ok=True,
    )
    existing_files = list_existing_split_files(api)
    dataset_card = load_dataset_card(api)
    existing_num_rows = get_existing_num_rows(dataset_card)
    upload_state, state_exists = load_upload_state(api)

    if existing_files:
        print(
            f"Found {len(existing_files)} existing {SPLIT} Parquet shard(s) "
            f"and {existing_num_rows} recorded rows."
        )
    if not state_exists and existing_files:
        upload_state["assumed_uploaded_before_index"] = shared_indexes[0]
        print(
            "No Hub upload state exists yet. Bootstrapping this batch as new and "
            f"assuming source indexes below {shared_indexes[0]} are already handled."
        )

    uploaded_source_indexes = ranges_to_indexes(upload_state["uploaded_source_ranges"])
    existing_keys = set()
    if CHECK_REMOTE_UPLOAD_KEYS:
        existing_keys, scanned_num_rows = scan_remote_upload_keys()
        if existing_num_rows and scanned_num_rows != existing_num_rows:
            raise ValueError(
                f"Hub metadata reports {existing_num_rows} rows but remote scan found "
                f"{scanned_num_rows}."
            )

    rows = build_rows(
        vi_rows,
        en_rows,
        shared_indexes,
        skip_keys=existing_keys,
        skip_source_indexes=uploaded_source_indexes,
        assumed_uploaded_before_index=upload_state["assumed_uploaded_before_index"],
    )
    if rows:
        total_shards = math.ceil(len(rows) / ROWS_PER_SHARD)
        print(f"Ready to append {len(rows)} rows in {total_shards} shards.")
        append_rows_to_hub(rows, api, dataset_card, existing_num_rows, upload_state)
    else:
        print("No new rows to upload. This batch is already complete.")

    print_next_start_index_hint(shared_indexes)


def save_locally() -> None:
    vi_rows, en_rows, shared_indexes = load_manifests()
    rows = build_rows(vi_rows, en_rows, shared_indexes)
    dataset = build_dataset(rows)
    print(dataset)
    dataset.save_to_disk(str(LOCAL_SAVE_DIR))
    print(f"Saved local dataset to: {LOCAL_SAVE_DIR}")
    print_next_start_index_hint(shared_indexes)
    print("Set PUSH_TO_HUB = True to upload.")


def main() -> None:
    upload_to_hub() if PUSH_TO_HUB else save_locally()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        raise
