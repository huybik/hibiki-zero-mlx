"""Validate and run the exact attested Qwen MLX retry-v6 production plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from benchmark_vivos_qwen_mlx_retry_v6 import (
    production_attestation_path,
    validate_production_path,
)
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    canonical_json,
    require_mlx_audio_commit,
    sha256_bytes,
    sha256_file,
    verify_mlx_snapshot,
)


SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_production_attestation_v6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    run = commands.add_parser("run")
    for command in (validate, run):
        command.add_argument("production_plan", type=Path)
    run.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--retry-ids", type=Path)
    return parser.parse_args()


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def validate(plan_path: Path) -> tuple[dict, dict]:
    from huggingface_hub import snapshot_download

    plan_path = plan_path.expanduser().resolve()
    plan, rows = validate_production_path(plan_path)
    contract_path = production_attestation_path(plan_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA:
        raise RuntimeError("Unexpected production attestation schema")
    if contract.get("production_plan") != attestation(plan_path):
        raise RuntimeError("Production attestation does not bind the exact plan")
    for item in [contract["policy"], contract["validation_go"], *contract["scripts"]]:
        if attestation(Path(item["path"])) != item:
            raise RuntimeError(f"Production input or script changed: {item['path']}")
    expected_runner = next(
        (item for item in contract["scripts"] if Path(item["path"]) == Path(__file__).resolve()),
        None,
    )
    if expected_runner is None:
        raise RuntimeError("Production attestation does not bind this runner")
    groups_sha = sha256_bytes(canonical_json(plan["groups"]).encode())
    if (
        contract["group_contract"]["rows"] != len(rows)
        or contract["group_contract"]["groups"] != len(plan["groups"])
        or contract["group_contract"]["groups_canonical_sha256"] != groups_sha
    ):
        raise RuntimeError("Production group contract changed")
    runtime = contract["runtime"]
    if version("mlx") != runtime["mlx"] or version("mlx-audio") != runtime["mlx-audio"]:
        raise RuntimeError("Production package versions changed")
    require_mlx_audio_commit()
    model = contract["model"]
    if model["id"] != MLX_MODEL_ID or model["revision"] != MLX_MODEL_REVISION:
        raise RuntimeError("Production model identity changed")
    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    if verify_mlx_snapshot(root) != model["files_sha256"]:
        raise RuntimeError("Production model snapshot changed")
    return plan, contract


def main() -> None:
    args = parse_args()
    plan, contract = validate(args.production_plan)
    if args.action == "validate":
        print(
            json.dumps(
                {
                    "valid": True,
                    "production_plan": contract["production_plan"],
                    "rows": plan["rows"],
                    "groups": len(plan["groups"]),
                    "model_files": len(contract["model"]["files_sha256"]),
                    "runner": next(
                        item for item in contract["scripts"] if Path(item["path"]) == Path(__file__).resolve()
                    ),
                },
                indent=2,
            )
        )
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_vivos_qwen_mlx_retry_v6.py")),
        "run-production",
        str(args.production_plan.expanduser().resolve()),
        "--round",
        str(args.round),
    ]
    if args.retry_ids is not None:
        command.extend(["--retry-ids", str(args.retry_ids.expanduser().resolve())])
    subprocess.run(command, check=True)
    manifest = args.production_plan.parent / f"generation_attempt{args.round}_manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("production_plan") != contract["production_plan"]:
        raise RuntimeError("Completed attempt manifest is not bound to the production plan")


if __name__ == "__main__":
    main()
