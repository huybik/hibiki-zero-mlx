"""Exact one-shot construction of the immutable allocator-cache repair contract."""

import hashlib
import json
from pathlib import Path

REPO = Path("/Users/macoblle/MEGA/Projects/sidequest/research/hibiki-zero/code")
ROOT = Path("/Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_retry_v6_full")
ORIGINAL_PLAN = ROOT / "production_plan.json"
ORIGINAL_ATTESTATION = ROOT / "production_attestation.json"
REPAIR_PLAN = ROOT / "production_plan_repair1.json"
REPAIR_ATTESTATION = ROOT / "production_attestation_repair1.json"
DIAGNOSIS = (
    REPO
    / "reports/benchmarks/vivos_tts/derived/2026-08-04_qwen_mlx_memory_diagnosis/metrics.md"
)
CREATED_UTC = "2026-08-04T13:46:23.688665+00:00"
COMMIT = "215467efd9946ce1296f87456bbedb5b8583c9e8"


def attest(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


benchmark = REPO / "training-data/benchmark_vivos_qwen_mlx_retry_v6.py"
runner = REPO / "training-data/run_vivos_qwen_production_v6.py"
compaction = REPO / "training-data/qwen_mlx_compaction.py"
recurrent = REPO / "training-data/qwen_mlx_recurrent.py"
repair = {
    "schema_version": "hibiki_vivos_qwen3_tts_mlx_allocator_cache_repair_v1",
    "created_utc": CREATED_UTC,
    "repository_commit": COMMIT,
    "reason": "MLX Metal allocator cache retained high-water allocations across production groups",
    "original_production_plan": attest(ORIGINAL_PLAN),
    "original_production_attestation": attest(ORIGINAL_ATTESTATION),
    "diagnosis": attest(DIAGNOSIS),
    "stop_boundary": {
        "attempt": 0,
        "completed_groups": 668,
        "completed_rows": 5266,
        "temporary_groups": 0,
        "media_error_rows": 0,
        "last_completed_group": "attempt0_t08_VIVOSSPK22_0023_279c2b5ff272",
        "generator_session": "hibiki_vivos_qwen_v6_attempt0_20260804",
        "supervisor_session": "hibiki_vivos_qwen_v6_postprocess_r1_20260804",
    },
    "implementation": {
        "revision": "allocator-cache-repair1",
        "allocator_cache": "clear_after_each_group",
        "token_count": "codec_frames",
        "legacy_groups_are_immutable": True,
        "same_source_group_rng_model_policy_contract": True,
    },
}
plan = json.loads(ORIGINAL_PLAN.read_text(encoding="utf-8"))
plan["created_utc"] = CREATED_UTC
plan["repository_commit_at_prepare"] = COMMIT
plan["command"] = [
    "training-data/run_vivos_qwen_production_v6.py",
    "run",
    str(REPAIR_PLAN),
    "--round",
    "0",
]
plan["script"] = attest(benchmark)
plan["repair"] = repair
write_new(REPAIR_PLAN, plan)

contract = json.loads(ORIGINAL_ATTESTATION.read_text(encoding="utf-8"))
contract["created_utc"] = CREATED_UTC
contract["production_plan"] = attest(REPAIR_PLAN)
contract["scripts"] = [attest(path) for path in (benchmark, compaction, recurrent, runner)]
contract["repair"] = repair
write_new(REPAIR_ATTESTATION, contract)
