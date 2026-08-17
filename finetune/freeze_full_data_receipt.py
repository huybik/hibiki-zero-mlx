#!/usr/bin/env python
"""Freeze the exact full post-source-EOS training curriculum."""
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
ELIGIBLE_CACHE_COUNTS = [683_114, 1_393]
CACHE_WEIGHTS = [0.95, 0.05]
SEED = 42
BATCH_SIZE = 16
MAX_FRAMES = 280
VALIDATION_ROWS = 148
VALIDATION_BATCH_SIZE = 4
VALIDATION_MAX_FRAMES = 470
EXPECTED_MANIFEST_SHA256 = "63f584a43dc5cba59fb948d5aa294ed72bc29634910968f9cda947c70019d1b5"
EXPECTED_TRANSFORM_SHA256 = "ad0bbcbeb7e14cab0647cae025a5a24110f99613bef692e1643c36d4a88f2dc7"
EXPECTED_VALIDATION_TRANSFORM_SHA256 = (
    "f563ef582207462c90b300c38d08f3907cd00081d5fea3da0180d140c596d3e6"
)
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
    "qualified_ascii_asr_parent": (
        "d37d69103bff8f128b9b69fc9634a018d8ab5c5c58dbb0b5cc98ecf5a26f92ca"
    ),
}
EXPECTED_RECEIPT_SHA256 = "ece5948ddb72f11f14048351f170c4c5218503c484db9c623ea6b4f52796ff0d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phomt-cache", type=Path, required=True)
    parser.add_argument("--fleurs-train-cache", type=Path, required=True)
    parser.add_argument("--fleurs-validation-cache", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True)
    parser.add_argument("--mimi-weight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
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
        "qualified_ascii_asr_parent": require_file(
            args.init_checkpoint, "qualified ASCII-ASR parent"
        ),
    }
    artifacts = {name: sha256_file(path) for name, path in artifact_paths.items()}
    if artifacts != EXPECTED_ARTIFACTS:
        raise RuntimeError(f"Artifact hashes changed: {artifacts} != {EXPECTED_ARTIFACTS}")
    tokenizer = artifact_paths["tokenizer"]
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {
        "phomt": cache_receipt(phomt_cache, 0),
        "fleurs_train": cache_receipt(fleurs_train_cache, 1),
        "fleurs_validation": cache_receipt(fleurs_validation_cache, 0),
    }
    for name, expected in EXPECTED_CACHE.items():
        actual = {key: cache[name][key] for key in ("shards", "bytes", "sha256")}
        if actual != expected:
            raise RuntimeError(f"{name} cache receipt mismatch: {actual} != {expected}")

    dataset = common.CachedCodeDataset(
        [phomt_cache, fleurs_train_cache],
        False,
        0,
        0,
        retained_text_column="text_en",
    )
    tokenizer_sha256 = common.transform_post_source_eos_translation(dataset, tokenizer)
    dataset.filter_max_frames(MAX_FRAMES, "full post-source-EOS pool")
    if cache_counts(dataset) != ELIGIBLE_CACHE_COUNTS:
        raise RuntimeError(
            f"Eligible cache counts changed: {cache_counts(dataset)} != "
            f"{ELIGIBLE_CACHE_COUNTS}"
        )
    dataset.select_weighted(CACHE_WEIGHTS, ROWS, SEED)
    dataset.samples.sort(key=lambda sample: sample["frames"])
    dataset.shuffle_batch_order(BATCH_SIZE, SEED)
    if len(dataset) != ROWS or cache_counts(dataset) != CACHE_COUNTS:
        raise RuntimeError("Final weighted curriculum counts changed")
    dataset.require_max_frames(MAX_FRAMES, "Full post-source-EOS curriculum")

    manifest = common.sample_manifest_bytes(dataset)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Sample manifest SHA-256 changed: {manifest_sha256} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )
    transform_payload = common.post_source_eos_translation_payload(
        dataset, tokenizer_sha256
    )
    transform_sha256 = payload_sha256(transform_payload)
    if transform_sha256 != EXPECTED_TRANSFORM_SHA256:
        raise RuntimeError(
            f"Training transform SHA-256 changed: {transform_sha256} != "
            f"{EXPECTED_TRANSFORM_SHA256}"
        )
    common.freeze_post_source_eos_translation(
        dataset, tokenizer_sha256, out_dir / "post_source_eos_translation.json"
    )

    validation = common.CachedCodeDataset(
        fleurs_validation_cache,
        False,
        0,
        0,
        retained_text_column="text_en",
    )
    validation_tokenizer_sha256 = common.transform_post_source_eos_translation(
        validation, tokenizer
    )
    validation.samples.sort(key=lambda sample: sample["frames"])
    validation_payload = common.post_source_eos_translation_payload(
        validation, validation_tokenizer_sha256
    )
    validation_transform_sha256 = payload_sha256(validation_payload)
    if (
        len(validation) != VALIDATION_ROWS
        or validation_payload["observed_max_frames"] != VALIDATION_MAX_FRAMES
        or validation_transform_sha256 != EXPECTED_VALIDATION_TRANSFORM_SHA256
    ):
        raise RuntimeError("Validation transform receipt changed")
    common.freeze_post_source_eos_translation(
        validation,
        validation_tokenizer_sha256,
        out_dir / "validation_post_source_eos_translation.json",
    )

    receipt = {
        "version": 1,
        "strategy": "full_post_source_eos_translation_curriculum",
        "artifacts": artifacts,
        "cache": cache,
        "eligible_cache_counts": ELIGIBLE_CACHE_COUNTS,
        "rows": ROWS,
        "cache_counts": CACHE_COUNTS,
        "cache_weights": CACHE_WEIGHTS,
        "selection_seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_frames": MAX_FRAMES,
        "observed_max_frames": transform_payload["observed_max_frames"],
        "sample_manifest_sha256": manifest_sha256,
        "post_source_eos_translation_sha256": transform_sha256,
        "validation": {
            "rows": VALIDATION_ROWS,
            "batch_size": VALIDATION_BATCH_SIZE,
            "max_frames": VALIDATION_MAX_FRAMES,
            "observed_max_frames": validation_payload["observed_max_frames"],
            "post_source_eos_translation_sha256": validation_transform_sha256,
        },
    }
    receipt_sha256 = payload_sha256(receipt)
    if receipt_sha256 != EXPECTED_RECEIPT_SHA256:
        raise RuntimeError(
            f"Full-data receipt SHA-256 changed: {receipt_sha256} != "
            f"{EXPECTED_RECEIPT_SHA256}"
        )
    atomic_write(out_dir / "sample_manifest.jsonl", manifest)
    atomic_write(
        out_dir / "full_data_receipt.json",
        json.dumps(
            {"sha256": receipt_sha256, "full_data_receipt": receipt},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        f"Full-data receipt passed: rows={ROWS} counts={CACHE_COUNTS} "
        f"manifest={manifest_sha256} transform={transform_sha256} "
        f"receipt={receipt_sha256}"
    )


if __name__ == "__main__":
    main()
