from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_DIR = REPO_ROOT / "remote_dataset" / "fleurs_vi_en"
DEFAULT_PAIRS_DIR = REPO_ROOT / "finetune" / "pairs"
DEFAULT_CACHE_ROOT = REPO_ROOT / "finetune" / "cache"
DEFAULT_RUN_DIR = REPO_ROOT / "finetune" / "runs" / "vn_lora"

DEFAULT_CONFIG_PATH = REPO_ROOT / "weights" / "config.json"
DEFAULT_MODEL_WEIGHT = REPO_ROOT / "weights" / "hibiki-pytorch-77f82164@110.safetensors"
DEFAULT_MIMI_WEIGHT = REPO_ROOT / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
DEFAULT_TOKENIZER = REPO_ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"

VALID_SPLITS = ("train", "validation", "test")
PAIR_FIELDS = (
    "id",
    "split",
    "vi_audio",
    "en_audio",
    "vi_duration_s",
    "en_duration_s",
    "text_vi",
    "text_en",
)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def repo_display_path(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_file(path: str | Path, label: str) -> Path:
    path = resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_dir(path: str | Path, label: str) -> Path:
    path = resolve_repo_path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    path = require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def resolve_manifest_audio_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (manifest_path.parent / path).resolve()


def read_fleurs_manifest(manifest_path: str | Path, split: str) -> list[dict[str, str]]:
    manifest_path = require_file(manifest_path, f"{split} manifest")
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [field for field in PAIR_FIELDS if field != "split" and field not in fieldnames]
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for row in reader:
            item = {
                field: (row.get(field, "") or "").strip()
                for field in PAIR_FIELDS
                if field != "split"
            }
            item["split"] = split
            for audio_field in ("vi_audio", "en_audio"):
                audio_path = resolve_manifest_audio_path(manifest_path, item[audio_field])
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {audio_field} for id={item['id']}: {audio_path}"
                    )
                item[audio_field] = repo_display_path(audio_path)
            if not item["text_en"]:
                raise ValueError(f"Empty text_en in {manifest_path} for id={item['id']}")
            rows.append(item)
    return rows


def write_pair_file(rows: list[dict[str, str]], path: str | Path, fmt: str) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elif fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(PAIR_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"Unsupported pair format: {fmt}")


def read_pair_file(path: str | Path) -> list[dict[str, str]]:
    path = require_file(path, "pair file")
    rows: list[dict[str, str]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append({field: str(row.get(field, "")) for field in PAIR_FIELDS})
    elif path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [field for field in PAIR_FIELDS if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
            rows = [{field: (row.get(field, "") or "") for field in PAIR_FIELDS} for row in reader]
    else:
        raise ValueError(f"Pair file must be .jsonl or .csv: {path}")

    for row in rows:
        for field in PAIR_FIELDS:
            if field not in row or row[field] == "":
                raise ValueError(f"{path} has an empty {field} field")
    return rows
