"""FLEURS manifests and the vi->en pair files built from them."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from training.paths import REPO_ROOT, repo_display_path, require_file, resolve_repo_path

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
