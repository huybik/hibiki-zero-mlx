"""Evaluation rows: the pairs files' rows by set, each heard in one language and scored against
the other's text (`EVAL_DIRECTIONS`), their checks and selection, and a row's source clip as the
runtimes hear it, through the input level stage (`read_source`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import sphn
import torch

from inference.hibiki_mlx.level import Leveller
from training.paths import require_file

EVAL_SET = "eval_set"  # the column read_eval_sets adds: the set a row belongs to
# The columns it adds for a row's direction: the clip heard and the text its translation is
# scored against. A pairs file holds the audio of one language (the key) and is translated into
# the other's text.
SOURCE, REFERENCE = "source_audio", "reference_text"
EVAL_DIRECTIONS = {"vi_audio": "text_en", "en_audio": "text_vi"}


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    path = require_file(path, "eval rows")
    if path.suffix != ".jsonl":
        raise ValueError(f"Eval rows must be .jsonl: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append({key: str(value) for key, value in row.items()})
    return rows


def read_eval_sets(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Each pairs file's rows by set name, in the order given, each row tagged under EVAL_SET and
    holding its clip and reference under SOURCE and REFERENCE. A set is named by its file's folder
    and stem, such as fleurs-vi/dev, and must have rows."""
    sets: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        path = require_file(path, "eval rows")
        name = f"{path.parent.name}/{path.stem}"
        if name in sets:
            raise ValueError(f"Two pairs files are both named {name}")
        rows = read_eval_rows(path)
        if not rows:
            raise ValueError(f"No rows in {path}")
        audio = [column for column in EVAL_DIRECTIONS if column in rows[0]]
        if len(audio) != 1:
            raise ValueError(f"{path} must hold one of the columns {', '.join(EVAL_DIRECTIONS)}, has {audio}")
        for index, row in enumerate(rows):
            if taken := {EVAL_SET, SOURCE, REFERENCE} & set(row):
                raise ValueError(f"{path} has columns {sorted(taken)} of its own")
            if [column for column in EVAL_DIRECTIONS if column in row] != audio:
                raise ValueError(f"{path} row {index} is heard in other audio than its first row's {audio[0]}")
            if missing := {audio[0], EVAL_DIRECTIONS[audio[0]]} - set(row):
                raise ValueError(f"{path} row {index} is missing columns: {', '.join(sorted(missing))}")
            row[EVAL_SET] = name
            row[SOURCE], row[REFERENCE] = row[audio[0]], row[EVAL_DIRECTIONS[audio[0]]]
        sets[name] = rows
    return sets


def selecting_sets(eval_sets: list[str], rows: list[dict[str, str]]) -> list[str]:
    """The sets that select a checkpoint: the first of each direction (the audio a set is heard
    in, `EVAL_DIRECTIONS`), in the order given."""
    heard: dict[str, str] = {}
    for row in rows:
        heard.setdefault(row[EVAL_SET], next(column for column in EVAL_DIRECTIONS if column in row))
    first: dict[str, str] = {}
    for name in eval_sets:
        first.setdefault(heard[name], name)
    return list(first.values())


def validate_eval_rows(
    rows: list[dict[str, str]],
    id_column: str,
    duration_column: str | None = None,
) -> None:
    """Rows of `read_eval_sets`: ids present and unique, and positive durations."""
    if not rows:
        return
    required = [id_column]
    if duration_column is not None:
        required.append(duration_column)
    for index, row in enumerate(rows):
        missing = [column for column in required if column not in row]
        if missing:
            raise ValueError(
                f"Eval row {index} is missing columns: {', '.join(missing)}"
            )
    ids = [row[id_column] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Eval rows must have unique {id_column!r} values")
    if duration_column is not None:
        for row in rows:
            try:
                duration = float(row[duration_column])
            except ValueError:
                duration = math.nan
            if not duration > 0:  # also rejects NaN, which would break the length sort
                raise ValueError(
                    f"Eval {duration_column} must be a positive number for "
                    f"{id_column}={row[id_column]}, got {row[duration_column]!r}"
                )


def select_eval_rows(
    rows: list[dict[str, str]], ids: list[str], id_column: str
) -> list[dict[str, str]]:
    """The rows of `ids`, in that order."""
    by_id = {row[id_column]: row for row in rows}
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Requested ids are missing: {missing[:10]}")
    return [by_id[row_id] for row_id in ids]


def ids_from_args(ids: str, ids_file: Path | None) -> list[str]:
    selected = [item.strip() for item in ids.split(",") if item.strip()]
    if ids_file is not None:
        path = require_file(ids_file, "id file")
        selected.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate ids in --ids/--ids-file")
    return selected


def audio_read(
    fpath: Path, to_sample_rate: int | None = None, mono: bool = False
) -> tuple[torch.Tensor, int]:
    """Read audio fpath and resample at to_sample_rate/transform to mono audio if specified."""
    wav, sr = sphn.read(fpath)
    if to_sample_rate is not None and sr != to_sample_rate:
        wav = sphn.resample(wav, sr, to_sample_rate)
        sr = to_sample_rate
    wav_tensor: torch.Tensor = torch.tensor(wav)
    if wav_tensor.ndim == 1:
        wav_tensor.unsqueeze(0)
    elif wav_tensor.ndim == 2:
        if wav_tensor.shape[0] > 2:
            raise ValueError(
                f"Audio {fpath} has too many channels, got {wav_tensor.shape[0]} but expected 1 or 2."
            )
        elif wav_tensor.shape[0] == 2 and mono:
            print(f"Audio {fpath} is stereo, averaging both channels to get a mono audio.")
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
    elif wav_tensor.ndim >= 3:
        raise ValueError(
            f"Audio {fpath} was loaded into a tensor of unsupported shape {wav_tensor.ndim}"
        )
    return wav_tensor, sr


def read_source(path: Path, sample_rate: int) -> torch.Tensor:
    """A source clip at Mimi's rate (stereo averaged to mono, where the runtimes take the first
    channel), through the runtimes' level stage (inference/hibiki_mlx/level.py) as a stream of
    its own; [1, samples]."""
    wav = audio_read(path, to_sample_rate=sample_rate, mono=True)[0]
    try:
        return torch.from_numpy(Leveller()(wav[0].numpy()))[None]
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
