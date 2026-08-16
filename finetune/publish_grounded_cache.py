#!/usr/bin/env python
"""Validate and publish the complete grounded-v2 PhoMT cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

EXPECTED_SHARDS = 1_377
ARCHIVE_COUNT = 8
CACHE_FORMAT = "hibiki_vn_grounded_cache_v2"
DATASET_REVISION = "33400f73dde07da539e8326313cbabe20b757740"
SAMPLE_ID = re.compile(r"phomt_s(\d{5})r\d{5}")
EXPECTED_CONFIG = {
    "n_q": 32,
    "dep_q": 16,
    "card": 2_048,
    "text_card": 48_000,
    "existing_text_padding_id": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("repo", help="Existing Hugging Face dataset repo as owner/name.")
    parser.add_argument("--prefix", default="grounded-v2")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_ids(cache_dir: Path, pattern: str) -> list[str]:
    result = []
    for path in sorted(cache_dir.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            sample_id = json.loads(line)["id"]
            match = SAMPLE_ID.fullmatch(sample_id)
            if not match or int(match.group(1)) >= EXPECTED_SHARDS:
                raise RuntimeError(f"invalid manifest sample id: {sample_id}")
            result.append(sample_id)
    return result


def validate_cache(cache_dir: Path) -> dict[str, object]:
    paths = sorted(cache_dir.glob("shard_*.pt"))
    expected_names = [f"shard_{index:05d}.pt" for index in range(EXPECTED_SHARDS)]
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} contiguous cache shards")

    sample_ids: set[str] = set()
    accepted = 0
    min_score = 1.0
    max_score = 0.0
    for shard_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != CACHE_FORMAT:
            raise RuntimeError(f"wrong cache format: {path}")
        if payload.get("dataset_revision") != DATASET_REVISION:
            raise RuntimeError(f"wrong dataset revision: {path}")
        if payload.get("alignment_min_score") != 0.5:
            raise RuntimeError(f"wrong alignment threshold: {path}")
        if payload.get("sample_rate") != 24_000 or payload.get("frame_rate") != 12.5:
            raise RuntimeError(f"wrong audio timing metadata: {path}")
        if payload.get("config") != EXPECTED_CONFIG:
            raise RuntimeError(f"wrong cache config: {path}")
        if not payload.get("samples"):
            raise RuntimeError(f"empty cache shard: {path}")
        for sample in payload["samples"]:
            sample_id = sample["id"]
            match = SAMPLE_ID.fullmatch(sample_id)
            if not match or int(match.group(1)) != shard_index:
                raise RuntimeError(f"sample belongs to wrong shard: {sample_id}")
            if sample_id in sample_ids:
                raise RuntimeError(f"duplicate sample id: {sample_id}")
            sample_ids.add(sample_id)
            codes = sample["codes"]
            if (
                codes.dtype != torch.int32
                or codes.ndim != 2
                or codes.shape[0] != 33
                or codes.shape[1] != sample["frames"]
                or sample["frames"] <= 0
            ):
                raise RuntimeError(f"invalid code tensor: {sample_id}")
            if sample.get("text_timing") != "wav2vec2_ctc_word_v1":
                raise RuntimeError(f"wrong text timing: {sample_id}")
            score = float(sample["alignment_score"])
            if not 0.5 <= score <= 1.0:
                raise RuntimeError(f"invalid alignment score: {sample_id}={score}")
            if not sample.get("alignment_text"):
                raise RuntimeError(f"empty alignment text: {sample_id}")
            index_range = sample.get("phomt_index_range")
            timbre_matched = bool(sample.get("cross_lingual_timbre_matched"))
            if index_range is None:
                valid_range = not timbre_matched
            else:
                valid_range = (
                    isinstance(index_range, (list, tuple))
                    and len(index_range) == 2
                    and all(isinstance(index, int) for index in index_range)
                    and index_range[0] <= index_range[1]
                    and timbre_matched == (index_range[0] >= 345_600)
                )
            if not valid_range:
                raise RuntimeError(f"invalid PhoMT source range: {sample_id}")
            min_score = min(min_score, score)
            max_score = max(max_score, score)
            accepted += 1

    pairs = manifest_ids(cache_dir, "pairs_w*.jsonl")
    rejects = manifest_ids(cache_dir, "alignment_rejects_w*.jsonl")
    if len(pairs) != len(set(pairs)) or set(pairs) != sample_ids:
        raise RuntimeError("pair manifests do not exactly match accepted cache samples")
    if len(rejects) != len(set(rejects)):
        raise RuntimeError("rejection manifests contain duplicate ids")
    if set(rejects) & sample_ids:
        raise RuntimeError("accepted and rejected sample ids overlap")
    return {
        "format": CACHE_FORMAT,
        "dataset_revision": DATASET_REVISION,
        "alignment_min_score": 0.5,
        "config": EXPECTED_CONFIG,
        "shards": EXPECTED_SHARDS,
        "accepted_samples": accepted,
        "rejected_samples": len(rejects),
        "min_alignment_score": min_score,
        "max_alignment_score": max_score,
    }


def archive_info(path: Path, first_shard: int, last_shard: int) -> dict[str, object]:
    return {
        "path": path.name,
        "first_shard": first_shard,
        "last_shard": last_shard,
        "shards": last_shard - first_shard + 1,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_stage(cache_dir: Path, prefix: str, stats: dict[str, object]) -> tuple[Path, dict]:
    stage_root = cache_dir / ".publish"
    if stage_root.exists():
        raise RuntimeError(f"publish staging already exists: {stage_root}")
    stage = stage_root / prefix
    stage.mkdir(parents=True)
    archives = []
    env = os.environ | {"COPYFILE_DISABLE": "1"}
    for chunk_index in range(ARCHIVE_COUNT):
        first = chunk_index * EXPECTED_SHARDS // ARCHIVE_COUNT
        stop = (chunk_index + 1) * EXPECTED_SHARDS // ARCHIVE_COUNT
        archive = stage / f"cache_chunk_{chunk_index}.tar.zst"
        names = [f"shard_{index:05d}.pt" for index in range(first, stop)]
        subprocess.run(
            ["tar", "--zstd", "-C", str(cache_dir), "-cf", str(archive), *names],
            check=True,
            env=env,
        )
        archives.append(archive_info(archive, first, stop - 1))
        print(f"Prepared {archive.name}", flush=True)

    auxiliary = []
    for source in sorted(
        [*cache_dir.glob("pairs_w*.jsonl"), *cache_dir.glob("alignment_rejects_w*.jsonl")]
    ):
        target = stage / source.name
        os.link(source, target)
        auxiliary.append(
            {"path": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
        )
    manifest = stats | {
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True
        ).strip(),
        "archives": archives,
        "auxiliary_files": auxiliary,
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "README.md").write_text(
        "# Grounded-v2 PhoMT cache\n\n"
        f"Validated `{stats['format']}` cache with {stats['shards']:,} shards, "
        f"{stats['accepted_samples']:,} accepted samples, and "
        f"{stats['rejected_samples']:,} rejected alignments.\n\n"
        "Restore from the dataset repository root:\n\n"
        "```bash\n"
        "mkdir -p finetune/cache/phomt_grounded_v2\n"
        "for archive in grounded-v2/cache_chunk_*.tar.zst; do\n"
        "  tar --zstd --exclude='._*' --exclude='*/._*' " + "\\\n"
        "    -xf \"$archive\" -C finetune/cache/phomt_grounded_v2\n"
        "done\n"
        "```\n\n"
        "See `manifest.json` for shard ranges and SHA-256 checksums.\n",
        encoding="utf-8",
    )
    return stage_root, manifest


def verify_remote(api: HfApi, repo: str, prefix: str, stage: Path, manifest: dict) -> None:
    expected = {
        item["path"]: item
        for item in [*manifest["archives"], *manifest["auxiliary_files"]]
    }
    remote = {
        item.path.removeprefix(f"{prefix}/"): item
        for item in api.list_repo_tree(
            repo, path_in_repo=prefix, repo_type="dataset", recursive=True, expand=True
        )
        if hasattr(item, "size")
    }
    for name, metadata in expected.items():
        item = remote.get(name)
        if item is None or item.size != metadata["bytes"]:
            raise RuntimeError(f"remote checksum verification failed: {prefix}/{name}")
        if item.lfs is not None:
            if item.lfs.sha256 != metadata["sha256"]:
                raise RuntimeError(f"remote checksum verification failed: {prefix}/{name}")
        else:
            remote_path = Path(
                hf_hub_download(
                    repo,
                    f"{prefix}/{name}",
                    repo_type="dataset",
                    force_download=True,
                    token=api.token,
                )
            )
            if sha256_file(remote_path) != metadata["sha256"]:
                raise RuntimeError(f"remote checksum verification failed: {prefix}/{name}")

    for name in ("manifest.json", "README.md"):
        remote_path = Path(
            hf_hub_download(
                repo,
                f"{prefix}/{name}",
                repo_type="dataset",
                force_download=True,
                token=api.token,
            )
        )
        if remote_path.read_bytes() != (stage / name).read_bytes():
            raise RuntimeError(f"remote metadata verification failed: {prefix}/{name}")
    expected_names = {*expected, "manifest.json", "README.md"}
    if set(remote) != expected_names:
        raise RuntimeError("remote grounded-v2 namespace contains unexpected files")


def publish(cache_dir: Path, repo: str, prefix: str) -> None:
    prefix = prefix.strip("/")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("prefix must be a non-empty relative repository path")
    cache_dir = cache_dir.resolve()
    stats = validate_cache(cache_dir)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing = [
        path
        for path in api.list_repo_files(repo, repo_type="dataset")
        if path == prefix or path.startswith(f"{prefix}/")
    ]
    if existing:
        raise RuntimeError(f"refusing to overwrite non-empty remote namespace: {prefix}")

    stage_root, manifest = prepare_stage(cache_dir, prefix, stats)
    stage = stage_root / prefix
    api.upload_folder(
        repo_id=repo,
        folder_path=stage,
        path_in_repo=prefix,
        repo_type="dataset",
        allow_patterns=[
            "cache_chunk_*.tar.zst",
            "pairs_w*.jsonl",
            "alignment_rejects_w*.jsonl",
        ],
        commit_message="Upload grounded-v2 cache data",
    )
    api.create_commit(
        repo,
        repo_type="dataset",
        commit_message=(
            f"Publish grounded-v2 cache: {stats['shards']} shards / "
            f"{stats['accepted_samples']} samples"
        ),
        operations=[
            CommitOperationAdd(f"{prefix}/manifest.json", stage / "manifest.json"),
            CommitOperationAdd(f"{prefix}/README.md", stage / "README.md"),
        ],
    )
    verify_remote(api, repo, prefix, stage, manifest)
    shutil.rmtree(stage_root)
    print(f"Published and verified https://huggingface.co/datasets/{repo}/tree/main/{prefix}")


def main() -> None:
    args = parse_args()
    publish(args.cache_dir, args.repo, args.prefix)


if __name__ == "__main__":
    main()
