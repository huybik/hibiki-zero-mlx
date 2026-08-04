"""Time and attest one fresh retry-v6 QA pass."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_vivos_qwen_mlx_batch import json_bytes
from synthesize_vivos import atomic_write_bytes, immutable_write, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()
    command = [
        sys.executable,
        str(Path(__file__).with_name("qa_vivos_qwen_mlx_efficiency_v3.py")),
        str(args.cohort.expanduser().resolve()),
        str(args.candidate.expanduser().resolve()),
        "--out-dir",
        str(args.out_dir.expanduser().resolve()),
    ]
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True)
    wall = time.monotonic() - started
    log_path = args.timing.expanduser().resolve().with_suffix(".log")
    atomic_write_bytes(log_path, (result.stdout + result.stderr).encode())
    if result.returncode:
        raise RuntimeError(f"QA failed with exit {result.returncode}; see {log_path}")
    qa_report = args.out_dir.expanduser().resolve() / "qa_report.json"
    report = {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_retry_qa_timing_v6",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "wall_seconds": wall,
        "scope": {
            "cohort": {"path": str(args.cohort.resolve()), "sha256": sha256_file(args.cohort)},
            "candidate": {
                "path": str((args.candidate / "candidate.json").resolve()),
                "sha256": sha256_file(args.candidate / "candidate.json"),
            },
            "qa_report": {"path": str(qa_report), "sha256": sha256_file(qa_report)},
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        },
    }
    immutable_write(args.timing.expanduser().resolve(), json_bytes(report))
    print(f"QA wall: {wall:.3f} seconds")


if __name__ == "__main__":
    main()
