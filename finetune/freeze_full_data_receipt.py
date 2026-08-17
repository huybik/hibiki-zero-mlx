#!/usr/bin/env python
"""Freeze the direct full-data simultaneous-translation curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune import common  # noqa: E402
from finetune.utils import require_dir, require_file, resolve_repo_path  # noqa: E402

ROWS = 719_120
CACHE_COUNTS = [683_164, 35_956]
CACHE_WEIGHTS = [0.95, 0.05]
SEED = 42
BATCH_SIZE = 16
MAX_FRAMES = 280
VALIDATION_ROWS = 138
VALIDATION_BATCH_SIZE = 4
VALIDATION_MAX_FRAMES = 280
VALIDATION_OBSERVED_MAX_FRAMES = 277
EXPECTED_CACHE = {
    "phomt": {
        "shards": 1_377,
        "bytes": 9_246_911_976,
        "sha256": "db36bd1a87da6318f3011c9106b5ba5c532b44a51d460bc5ab1d95a7f6cd2b29",
    },
    "fleurs_train": {
        "shards": 46,
        "bytes": 34_038_967,
        "sha256": "8fcbb3638d74f2e2cd2e2f1d8b6e5726e5e5e4d14b1268c3d8459cb0fbe3934e",
    },
    "fleurs_validation": {
        "shards": 5,
        "bytes": 3_521_617,
        "sha256": "0a3f3ef160220de385ca4e217aaa32a02ef360848ad344358bfdfbf7750bee8c",
    },
}
EXPECTED_ARTIFACTS = {
    "config": "a99f354a6131034b688fc9f91c889dc10e7eeff96ce65e94447be33d1be325a5",
    "model": "cd78e453b3b80299255bea02be439bcc2552b57c03cd82dbf0e9792e20100db8",
    "mimi": "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    "tokenizer": "c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phomt-cache", type=Path, required=True)
    parser.add_argument("--fleurs-train-cache", type=Path, required=True)
    parser.add_argument("--fleurs-validation-cache", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--mimi-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_receipt(path: Path, cache_index: int) -> dict[str, int | str]:
    digest = hashlib.sha256()
    shards = sorted(path.glob("shard_*.pt"))
    for shard in shards:
        row = {
            "bytes": shard.stat().st_size,
            "cache_index": cache_index,
            "path": shard.name,
            "sha256": sha256_file(shard),
        }
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return {
        "shards": len(shards),
        "bytes": sum(shard.stat().st_size for shard in shards),
        "sha256": digest.hexdigest(),
    }


def atomic_write(path: Path, content: str | bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        if isinstance(content, bytes):
            temporary.write_bytes(content)
        else:
            temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_file(path: Path, content: str | bytes) -> None:
    encoded = content.encode() if isinstance(content, str) else content
    if path.is_file() and path.read_bytes() != encoded:
        raise RuntimeError(f"Frozen receipt changed: {path}")
    if not path.is_file():
        atomic_write(path, content)


def cache_counts(dataset: common.CachedCodeDataset) -> list[int]:
    return [
        sum(sample["cache_index"] == cache_index for sample in dataset.samples)
        for cache_index in range(dataset.cache_count)
    ]


def main() -> None:
    args = parse_args()
    phomt_cache = require_dir(args.phomt_cache, "PhoMT cache")
    fleurs_train_cache = require_dir(args.fleurs_train_cache, "FLEURS train cache")
    fleurs_validation_cache = require_dir(
        args.fleurs_validation_cache, "FLEURS validation cache"
    )
    artifact_paths = {
        "config": require_file(args.config_path, "config"),
        "model": require_file(args.model_weight, "model weight"),
        "mimi": require_file(args.mimi_weight, "Mimi weight"),
        "tokenizer": require_file(args.tokenizer, "tokenizer"),
    }
    artifacts = {name: sha256_file(path) for name, path in artifact_paths.items()}
    if artifacts != EXPECTED_ARTIFACTS:
        raise RuntimeError(f"Upstream artifact hashes changed: {artifacts}")

    cache = {
        "phomt": cache_receipt(phomt_cache, 0),
        "fleurs_train": cache_receipt(fleurs_train_cache, 1),
        "fleurs_validation": cache_receipt(fleurs_validation_cache, 0),
    }
    if cache != EXPECTED_CACHE:
        raise RuntimeError("Published grounded-v2 cache receipt changed")

    dataset = common.CachedCodeDataset(
        [phomt_cache, fleurs_train_cache], False, 0, MAX_FRAMES
    )
    eligible_cache_counts = cache_counts(dataset)
    dataset.select_weighted(CACHE_WEIGHTS, ROWS, SEED)
    dataset.samples.sort(key=lambda sample: sample["frames"])
    dataset.shuffle_batch_order(BATCH_SIZE, SEED)
    if len(dataset) != ROWS or cache_counts(dataset) != CACHE_COUNTS:
        raise RuntimeError("Direct full-data weighted membership changed")
    dataset.require_max_frames(MAX_FRAMES, "Direct simultaneous curriculum")

    manifest = common.sample_manifest_bytes(dataset)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()

    validation = common.CachedCodeDataset(
        fleurs_validation_cache, True, 0, VALIDATION_MAX_FRAMES
    )
    if len(validation) != VALIDATION_ROWS:
        raise RuntimeError(
            f"Direct validation must retain {VALIDATION_ROWS} rows, got {len(validation)}"
        )
    validation.require_max_frames(VALIDATION_MAX_FRAMES, "Direct validation")
    observed_validation_max_frames = max(
        sample["frames"] for sample in validation.samples
    )
    if observed_validation_max_frames != VALIDATION_OBSERVED_MAX_FRAMES:
        raise RuntimeError(
            "Direct validation observed maximum changed: "
            f"{observed_validation_max_frames} != {VALIDATION_OBSERVED_MAX_FRAMES}"
        )

    receipt = {
        "version": 2,
        "strategy": "direct_voice_preserving_simultaneous_translation",
        "artifacts": artifacts,
        "cache": cache,
        "streams": {
            "target_audio": {
                "content": "cached_english_mimi",
                "termination": "native_minus_one_mask",
            },
            "target_text": {
                "content": "cached_ctc_timed_english",
                "termination": "tokenizer_eos",
            },
            "source_audio": {
                "content": "cached_vietnamese_mimi",
                "termination": "explicit_card_eos",
            },
            "transform": None,
        },
        "eligible_cache_counts": eligible_cache_counts,
        "rows": ROWS,
        "cache_counts": CACHE_COUNTS,
        "cache_weights": CACHE_WEIGHTS,
        "selection_seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_frames": MAX_FRAMES,
        "observed_max_frames": max(sample["frames"] for sample in dataset.samples),
        "sample_manifest_sha256": manifest_sha256,
        "validation": {
            "rows": VALIDATION_ROWS,
            "batch_size": VALIDATION_BATCH_SIZE,
            "max_frames": VALIDATION_MAX_FRAMES,
            "observed_max_frames": observed_validation_max_frames,
            "shuffle": False,
        },
    }
    receipt_sha256 = payload_sha256(receipt)
    document = json.dumps(
        {"sha256": receipt_sha256, "full_data_receipt": receipt},
        indent=2,
        sort_keys=True,
    ) + "\n"

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_file(out_dir / "sample_manifest.jsonl", manifest)
    freeze_file(out_dir / "full_data_receipt.json", document)
    print(
        f"Direct full-data receipt passed: rows={ROWS} counts={CACHE_COUNTS} "
        f"manifest={manifest_sha256} receipt={receipt_sha256}"
    )


if __name__ == "__main__":
    main()
