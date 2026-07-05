from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "ura-hcmut/PhoMT"
DATASETS_DIR = Path(r"D:\Code\datasets")
CACHE_DIR = DATASETS_DIR / ".hf_cache"


# Load raw text to text machine translation
def load_raw_datasets(verbose: bool = True) -> DatasetDict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))

    if verbose:
        print(f"Downloaded {DATASET_ID} into cache: {CACHE_DIR}")
        for split_name, split_dataset in dataset.items():
            print(f"{split_name}: {len(split_dataset):,} rows")

    return dataset


def load_train_samples(
    n: int = 100,
    start_index: int = 0,
    end_index: int | None = None,
    verbose: bool = True,
) -> Dataset:
    dataset = load_raw_datasets(verbose=verbose)
    train = dataset["train"]

    if n < 0:
        raise ValueError("n must be non-negative")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    if end_index is None:
        end_index = start_index + n
    if end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")

    if start_index >= len(train):
        return train.select([])

    end_index = min(end_index, len(train))
    return train.select(range(start_index, end_index))


def preview_dataset(n: int = 5, start_index: int = 0) -> None:
    train = load_train_samples(n=n, start_index=start_index)

    print(train.column_names)
    print(train.to_pandas()[["vi", "en"]])


if __name__ == "__main__":
    preview_dataset()
