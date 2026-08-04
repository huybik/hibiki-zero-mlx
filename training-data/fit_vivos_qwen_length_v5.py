"""Freeze the Qwen output-length scheduler from immutable VIVOS scalar sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from synthesize_vivos import immutable_write, read_jsonl, sha256_file


SCHEMA = "hibiki_vivos_qwen_length_model_v5"
EXPECTED_ROWS = 1489


def metrics(actual, predicted, np):
    error = predicted - actual
    residual = float(np.sum(error**2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rows": int(actual.size),
        "mae_frames": float(np.mean(np.abs(error))),
        "rmse_frames": float(np.sqrt(np.mean(error**2))),
        "r2": 1.0 - residual / total if total else None,
        "p90_absolute_error_frames": float(np.quantile(np.abs(error), 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    root = args.campaign_root.expanduser().resolve()
    interruption = json.loads((root / "interruption_2026-08-04.json").read_text())
    plan_path = root / "generation_plan.jsonl"
    plan = {str(row["id"]): row for row in read_jsonl(plan_path)}
    paths = sorted((root / "attempts" / "attempt0").glob("**/*.json"))
    if len(paths) != EXPECTED_ROWS or interruption["completed_attempt0_rows"] != EXPECTED_ROWS:
        raise RuntimeError("The frozen scalar sidecar count changed")
    digest = hashlib.sha256()
    sidecars = []
    for path in paths:
        sidecar = json.loads(path.read_text())
        if sidecar.get("attempt") != 0 or sha256_file(Path(sidecar["output_wav"])) != sidecar["audio_sha256"]:
            raise RuntimeError(f"Changed scalar artifact: {path}")
        if sidecar["id"] not in plan:
            raise RuntimeError(f"Sidecar outside campaign: {path}")
        file_hash = sha256_file(path)
        digest.update(f"{file_hash}  {path.resolve()}\n".encode())
        sidecars.append((path, sidecar, file_hash))
    expected_manifest = interruption["artifacts"]["attempt0_sidecars"][
        "sorted_absolute_shasum_manifest_sha256"
    ]
    if digest.hexdigest() != expected_manifest:
        raise RuntimeError("Scalar sidecar hash manifest changed")

    model_root = Path(
        snapshot_download(
            repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            revision="a6eb4f68e4b056f1215157bb696209bc82a6db48",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    table = []
    for path, sidecar, file_hash in sidecars:
        row = plan[sidecar["id"]]
        table.append(
            {
                "id": sidecar["id"],
                "token_count": len(tokenizer.encode(str(row["text_en"]))),
                "source_duration_s": float(row["source_audio"]["duration_s"]),
                "reference_duration_s": float(row["reference"]["duration_s"]),
                "output_frames": int(round(float(sidecar["duration_s"]) * 12.5)),
                "sidecar": {"path": str(path.resolve()), "sha256": file_hash},
            }
        )
    x = np.asarray(
        [[1.0, row["token_count"], row["source_duration_s"]] for row in table],
        dtype=np.float64,
    )
    y = np.asarray([row["output_frames"] for row in table], dtype=np.float64)
    validation = np.asarray(
        [int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 5 == 0 for row in table],
        dtype=bool,
    )
    train_coefficients = np.linalg.lstsq(x[~validation], y[~validation], rcond=None)[0]
    frozen_coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": __import__("sys").argv,
        "campaign": {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "revision": sha256_file(plan_path),
            "interruption": {
                "path": str((root / "interruption_2026-08-04.json").resolve()),
                "sha256": sha256_file(root / "interruption_2026-08-04.json"),
            },
        },
        "training_data": {
            "rows": len(table),
            "sorted_absolute_sidecar_manifest_sha256": digest.hexdigest(),
            "selection": "all immutable attempt-0 sidecars present when campaign was stopped",
        },
        "features": ["intercept", "qwen_token_count", "source_duration_s"],
        "excluded_feature": {
            "reference_duration_s": "speaker-prompt property, not a legitimate target-length signal",
        },
        "holdout": {
            "rule": "SHA256(row_id) mod 5 == 0",
            "train": metrics(y[~validation], x[~validation] @ train_coefficients, np),
            "validation": metrics(y[validation], x[validation] @ train_coefficients, np),
            "coefficients": train_coefficients.tolist(),
        },
        "frozen_model": {
            "type": "ordinary least squares",
            "coefficients": frozen_coefficients.tolist(),
            "training_metrics": metrics(y, x @ frozen_coefficients, np),
            "prediction": "max(1, intercept + token_coef*qwen_tokens + source_duration_coef*source_duration_s)",
            "use": "scheduling only; never a generation stop condition",
        },
        "rows": table,
    }
    immutable_write(args.out.expanduser().resolve(), (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode())
    print(json.dumps({"holdout": report["holdout"], "frozen_model": report["frozen_model"]}, indent=2))


if __name__ == "__main__":
    main()
