from __future__ import annotations

import shutil
from pathlib import Path

from datasets import Audio, Dataset, load_dataset

from paths import DATASETS_DIR, HF_CACHE_DIR as CACHE_DIR


# =========================
# Config
# =========================

DATASET_REPO = "anquachdev/PhoMT-en-vi-speech"
SPLIT = "train"
START_INDEX = 0
N_SAMPLES = 20

PREVIEW_DIR = DATASETS_DIR / "audios"


def load_train_dataset(split: str = SPLIT, decode_audio: bool = False) -> Dataset:
    dataset = load_dataset(DATASET_REPO, split=split, cache_dir=str(CACHE_DIR))

    if not decode_audio:
        dataset = dataset.cast_column("audio_en", Audio(decode=False))
        dataset = dataset.cast_column("audio_vi", Audio(decode=False))

    return dataset


def load_samples(split: str = SPLIT) -> Dataset:
    return load_train_dataset(split=split, decode_audio=False)


def load_preview_samples(
    n: int = N_SAMPLES, start_index: int = START_INDEX, split: str = SPLIT
) -> Dataset:
    dataset = load_samples(split=split)
    end_index = min(start_index + n, len(dataset))
    return dataset.select(range(start_index, end_index))


def save_audio(audio: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio.get("bytes") is not None:
        output_path.write_bytes(audio["bytes"])
        return

    source_path = audio.get("path")
    if source_path:
        shutil.copyfile(source_path, output_path)
        return

    raise ValueError(f"Audio value does not contain bytes or a path: {audio}")


def export_preview(samples: Dataset, output_dir: Path = PREVIEW_DIR) -> None:
    for row_number, row in enumerate(samples):
        save_audio(row["audio_en"], output_dir / f"{row_number:04d}_en.wav")
        save_audio(row["audio_vi"], output_dir / f"{row_number:04d}_vi.wav")


def main() -> None:
    dataset = load_samples()
    preview_samples = load_preview_samples()
    print(dataset)
    print(dataset.column_names)

    for index, row in enumerate(preview_samples):
        print(f"{index}:")
        print(f"  en: {row['en']}")
        print(f"  vi: {row['vi']}")

    export_preview(preview_samples)
    print(f"Saved preview audio to: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
