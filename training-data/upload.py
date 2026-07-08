from __future__ import annotations

import csv
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from datasets import Audio, Dataset, DatasetInfo, load_dataset
from datasets.info import DatasetInfosDict
from datasets.splits import SplitDict, SplitInfo
from datasets.table import embed_table_storage
from datasets.utils.metadata import MetadataConfigs
from datasets.utils.py_utils import convert_file_size_to_int
from huggingface_hub import CommitOperationAdd, DatasetCard, DatasetCardData, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


# =========================
# Config
# =========================

DATASET_REPO = "anquachdev/PhoMT-en-vi-speech"
PUSH_TO_HUB = True
PRIVATE = False
RESUME_UPLOAD = True
CONFIG_NAME = "default"
SPLIT = "train"
DATA_DIR = "data"
APPEND_MAX_SHARD_SIZE = "500MB"
UPLOAD_KEY_COLUMNS = ("en", "vi")
DATASET_CARD_FILE = "README.md"

DATASETS_DIR = Path(r"D:\Code\datasets")
VI_MANIFEST = DATASETS_DIR / "vieNeu" / "outputs" / "vi" / "manifest.csv"
EN_MANIFEST = DATASETS_DIR / "english" / "outputs" / "en" / "manifest.csv"
LOCAL_SAVE_DIR = DATASETS_DIR / "phomt-en-vi-speech"
CACHE_DIR = DATASETS_DIR / ".hf_cache"

# Keep pairs with broadly similar lengths. The current generated data has slow
# English outliers, mostly af_nicole at Kokoro speed 1.0.
MIN_DURATION_RATIO = 0.5
MAX_DURATION_RATIO = 1.6


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


def get_next_start_index() -> int:
    shared_indexes = set(read_manifest(VI_MANIFEST)) & set(read_manifest(EN_MANIFEST))

    if not shared_indexes:
        raise ValueError(
            "No shared indexes found in the Vietnamese and English manifests."
        )

    return max(shared_indexes) + 1


def print_next_start_index_hint() -> None:
    print("Next batch config hint:")
    print(f"  Set START_INDEX = {get_next_start_index()} in training-data/pipeline.py")
    print("  Keep N_SAMPLES as the batch size you want to generate next.")


def get_audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("soundfile is required to check audio durations.") from error

    info = sf.info(path)
    return info.frames / info.samplerate


def get_upload_key(row: dict) -> tuple[str, ...]:
    return tuple(row[column] for column in UPLOAD_KEY_COLUMNS)


def build_rows(skip_keys: set[tuple[str, ...]] | None = None) -> list[dict]:
    vi_rows = read_manifest(VI_MANIFEST)
    en_rows = read_manifest(EN_MANIFEST)
    shared_indexes = sorted(set(vi_rows) & set(en_rows))
    skip_keys = skip_keys or set()

    rows = []
    skipped_existing = 0
    skipped_missing = 0
    skipped_duration = 0
    for index in shared_indexes:
        vi_row = vi_rows[index]
        en_row = en_rows[index]
        if (en_row["text"], vi_row["text"]) in skip_keys:
            skipped_existing += 1
            continue

        audio_vi = Path(vi_row["audio_path"])
        audio_en = Path(en_row["audio_path"])

        if not audio_vi.exists() or not audio_en.exists():
            skipped_missing += 1
            continue

        en_duration_s = get_audio_duration_seconds(audio_en)
        vi_duration_s = get_audio_duration_seconds(audio_vi)
        duration_ratio = en_duration_s / vi_duration_s if vi_duration_s else 0.0
        if not MIN_DURATION_RATIO <= duration_ratio <= MAX_DURATION_RATIO:
            skipped_duration += 1
            continue

        rows.append(
            {
                "en": en_row["text"],
                "vi": vi_row["text"],
                "audio_en": str(audio_en),
                "audio_vi": str(audio_vi),
                "duration_en_s": en_duration_s,
                "duration_vi_s": vi_duration_s,
                "duration_ratio_en_vi": duration_ratio,
            }
        )

    if skipped_existing:
        print(f"Skipped {skipped_existing} already-uploaded pairs.")

    if skipped_missing or skipped_duration:
        print(
            f"Skipped {skipped_missing} missing-audio pairs and "
            f"{skipped_duration} duration-mismatched pairs "
            f"(allowed EN/VI ratio: {MIN_DURATION_RATIO:g}-{MAX_DURATION_RATIO:g})."
        )

    if not rows and not skipped_existing:
        raise ValueError(
            "No paired rows found. Generate both English and Vietnamese "
            "audio for the same indexes first."
        )

    return rows


def build_dataset(rows: list[dict]) -> Dataset:
    dataset = Dataset.from_list(rows)
    for decode in (False, True):
        for column in ("audio_en", "audio_vi"):
            dataset = dataset.cast_column(column, Audio(decode=decode))
    return dataset


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


def load_existing_upload_state() -> tuple[set[tuple[str, ...]], int]:
    existing_dataset = load_dataset(
        DATASET_REPO,
        split=SPLIT,
        cache_dir=str(CACHE_DIR),
        columns=list(UPLOAD_KEY_COLUMNS),
    )
    return {get_upload_key(row) for row in existing_dataset}, len(existing_dataset)


def embed_external_files(dataset: Dataset) -> Dataset:
    dataset_format = dataset.format
    embedded = dataset.with_format("arrow").map(
        embed_table_storage,
        batched=True,
        batch_size=1000,
        keep_in_memory=True,
    )
    return embedded.with_format(**dataset_format)


def build_append_operations(
    dataset: Dataset, temp_dir: Path
) -> tuple[list[CommitOperationAdd], int, int]:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    dataset_nbytes = dataset._estimate_nbytes()
    max_shard_size = convert_file_size_to_int(APPEND_MAX_SHARD_SIZE)
    num_shards = max(1, int(dataset_nbytes / max_shard_size) + 1)
    uploaded_size = 0
    operations = []

    for shard_index in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
        shard = embed_external_files(shard)

        filename = f"{SPLIT}-append-{run_id}-{shard_index:05d}-of-{num_shards:05d}.parquet"
        local_path = temp_dir / filename
        path_in_repo = f"{DATA_DIR}/{filename}"

        shard.to_parquet(local_path)
        uploaded_size += local_path.stat().st_size
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path))
        )

    return operations, uploaded_size, dataset_nbytes


def load_dataset_card(api: HfApi) -> DatasetCard | None:
    try:
        card_path = api.hf_hub_download(DATASET_REPO, DATASET_CARD_FILE, repo_type="dataset")
    except EntryNotFoundError:
        return None

    return DatasetCard.load(card_path)


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
    api: HfApi,
    dataset: Dataset,
    existing_num_rows: int,
    uploaded_size: int,
    dataset_nbytes: int,
) -> CommitOperationAdd:
    dataset_card = load_dataset_card(api)
    dataset_card_data = dataset_card.data if dataset_card is not None else DatasetCardData()

    dataset_info = get_dataset_info_for_card(
        dataset_card_data=dataset_card_data,
        dataset=dataset,
        existing_num_rows=existing_num_rows,
        uploaded_size=uploaded_size,
        dataset_nbytes=dataset_nbytes,
    )
    DatasetInfosDict({CONFIG_NAME: dataset_info}).to_dataset_card_data(dataset_card_data)
    MetadataConfigs(
        {CONFIG_NAME: {"data_files": [{"split": SPLIT, "path": f"{DATA_DIR}/{SPLIT}-*"}]}}
    ).to_dataset_card_data(dataset_card_data)

    if dataset_card is None:
        dataset_card = DatasetCard(f"---\n{dataset_card_data}\n---\n")

    return CommitOperationAdd(
        path_in_repo=DATASET_CARD_FILE,
        path_or_fileobj=str(dataset_card).encode("utf-8"),
    )


def append_dataset_to_hub(dataset: Dataset, api: HfApi, existing_num_rows: int) -> None:
    api.create_repo(
        DATASET_REPO,
        repo_type="dataset",
        private=PRIVATE,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="phomt_hf_append_") as temp_dir:
        shard_operations, uploaded_size, dataset_nbytes = build_append_operations(
            dataset,
            Path(temp_dir),
        )
        shard_count = len(shard_operations)
        shard_operations.append(
            build_dataset_card_operation(
                api=api,
                dataset=dataset,
                existing_num_rows=existing_num_rows,
                uploaded_size=uploaded_size,
                dataset_nbytes=dataset_nbytes,
            )
        )
        api.create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=shard_operations,
            commit_message=f"Append {len(dataset)} {SPLIT} rows",
            commit_description=(
                "Resume upload: add new parquet shard files and leave existing "
                "dataset files untouched, then refresh dataset-card split metadata."
            ),
        )

    print(
        f"Appended {len(dataset)} rows in {shard_count} shard(s) to "
        f"https://huggingface.co/datasets/{DATASET_REPO}"
    )


def upload_to_hub() -> None:
    api = HfApi()
    existing_files = list_existing_split_files(api)

    if RESUME_UPLOAD and existing_files:
        print(f"Found {len(existing_files)} existing {SPLIT} parquet shard(s).")
        existing_keys, existing_num_rows = load_existing_upload_state()
        print(
            f"Found {existing_num_rows} existing uploaded row(s) "
            f"with {len(existing_keys)} unique key(s)."
        )

        rows = build_rows(skip_keys=existing_keys)
        if not rows:
            print("No new rows to upload.")
        else:
            dataset = build_dataset(rows)
            print(dataset)
            append_dataset_to_hub(dataset, api, existing_num_rows=existing_num_rows)
    else:
        rows = build_rows()
        dataset = build_dataset(rows)
        print(dataset)
        dataset.push_to_hub(
            DATASET_REPO,
            private=PRIVATE,
            config_name=CONFIG_NAME,
            split=SPLIT,
            data_dir=DATA_DIR,
        )
        print(f"Pushed to https://huggingface.co/datasets/{DATASET_REPO}")

    print_next_start_index_hint()


def main() -> None:
    if PUSH_TO_HUB:
        upload_to_hub()
    else:
        rows = build_rows()
        dataset = build_dataset(rows)
        print(dataset)
        dataset.save_to_disk(str(LOCAL_SAVE_DIR))
        print(f"Saved local dataset to: {LOCAL_SAVE_DIR}")
        print_next_start_index_hint()
        print("Set PUSH_TO_HUB = True to upload.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Upload build failed: {error}", file=sys.stderr)
        raise
