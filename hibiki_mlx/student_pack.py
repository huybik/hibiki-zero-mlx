"""Strict manifest, receipt, q4, and parity contract for mobile student packs."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

PACK_FORMAT = "hibiki_student_pack_v1"
MANIFEST_FORMAT = "hibiki_student_manifest_v1"
SHAPE_FORMAT = "hibiki_student_shape_receipt_v1"
PARITY_FORMAT = "hibiki_parallel_parity_v1"
HASH_RE = re.compile(r"[0-9a-f]{64}")
MIMI_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
TOKENIZER_SHA256 = "c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def validate_student_config(cfg: dict[str, Any]) -> None:
    common_keys = {
        "architecture",
        "artifact_format",
        "card",
        "causal",
        "conditioners",
        "context",
        "cross_attention",
        "delays",
        "dep_q",
        "depformer_dim",
        "depformer_dim_feedforward",
        "depformer_multi_linear",
        "depformer_num_heads",
        "depformer_num_layers",
        "depformer_pos_emb",
        "depformer_weights_per_step",
        "dim",
        "existing_text_padding_id",
        "frame_rate",
        "frame_samples",
        "fuser",
        "gating",
        "head",
        "head_passes",
        "hidden_scale",
        "layer_scale",
        "lm_gen_config",
        "max_period",
        "mimi_name",
        "model_type",
        "moshi_name",
        "n_q",
        "norm",
        "num_heads",
        "num_layers",
        "parent_repo",
        "parent_revision",
        "parity_fixture_name",
        "positional_embedding",
        "quantization_bits",
        "quantization_group_size",
        "sample_rate",
        "selected_parent_layers",
        "text_card",
        "tokenizer_name",
    }
    expected_keys = common_keys | (
        {"parallel_head_dim", "parallel_head_layers"} if cfg.get("head") == "parallel_v1" else set()
    )
    if set(cfg) != expected_keys:
        raise RuntimeError(
            f"Student config keys changed: missing={expected_keys - set(cfg)}, "
            f"extra={set(cfg) - expected_keys}"
        )
    expected = {
        "artifact_format": PACK_FORMAT,
        "architecture": "hibiki_m_12l",
        "parent_repo": "kyutai/hibiki-1b-pytorch-bf16",
        "parent_revision": "65dee9b6a682393d4e9b193ccbe314e401e230c9",
        "sample_rate": 24_000,
        "frame_rate": 12.5,
        "frame_samples": 1_920,
        "n_q": 16,
        "dep_q": 8,
        "dim": 2_048,
        "num_heads": 16,
        "num_layers": 12,
        "hidden_scale": 4.125,
        "causal": True,
        "layer_scale": None,
        "context": 1_500,
        "max_period": 100_000.0,
        "gating": "silu",
        "norm": "rms_norm_f32",
        "positional_embedding": "rope",
        "cross_attention": False,
        "model_type": "hibiki",
        "text_card": 48_000,
        "existing_text_padding_id": 3,
        "card": 2_048,
        "quantization_bits": 4,
        "quantization_group_size": 32,
        "mimi_name": "mimi-pytorch-e351c8d8@125.safetensors",
        "tokenizer_name": "tokenizer_spm_48k_multi6_2.model",
        "moshi_name": "model.q4.safetensors",
        "parity_fixture_name": "parity_fixture.npz",
        "delays": [0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2],
        "depformer_dim": 1_024,
        "depformer_num_heads": 16,
        "depformer_num_layers": 6,
        "depformer_dim_feedforward": None,
        "depformer_multi_linear": True,
        "depformer_pos_emb": "none",
        "depformer_weights_per_step": True,
        "conditioners": {
            "description": {
                "type": "lut",
                "lut": {
                    "n_bins": 31,
                    "dim": 16,
                    "tokenizer": "noop",
                    "possible_values": ["very_bad", "bad", "neutral", "good", "very_good"],
                },
            }
        },
        "fuser": {
            "cross_attention_pos_emb": False,
            "cross_attention_pos_emb_scale": 1,
            "sum": ["description"],
            "prepend": [],
            "cross": [],
        },
        "lm_gen_config": {"temp": 0.8, "temp_text": 0.4, "top_k": 250, "top_k_text": 25},
        "selected_parent_layers": [0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15],
    }
    mismatch = [
        f"{key}={cfg.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if cfg.get(key) != value
    ]
    if mismatch:
        raise RuntimeError("Student config mismatch: " + "; ".join(mismatch))
    if not all(
        isinstance(cfg.get(key), str) and Path(cfg[key]).name == cfg[key]
        for key in ("moshi_name", "mimi_name", "tokenizer_name", "parity_fixture_name")
    ):
        raise RuntimeError("Student pack file names must be simple relative names")
    head = cfg.get("head")
    if head == "ar":
        if cfg.get("head_passes") != 1:
            raise RuntimeError("AR head must use one pass")
    elif head == "parallel_v1":
        if cfg.get("head_passes") not in (1, 2):
            raise RuntimeError("parallel_v1 supports one or two fixed passes")
        if (cfg.get("parallel_head_dim"), cfg.get("parallel_head_layers")) != (512, 2):
            raise RuntimeError("parallel_v1 shape changed")
    else:
        raise RuntimeError(f"Unsupported student audio head: {head!r}")


def pack_files(cfg: dict[str, Any]) -> list[str]:
    return [
        "config.json",
        str(cfg["moshi_name"]),
        str(cfg["mimi_name"]),
        str(cfg["tokenizer_name"]),
        str(cfg["parity_fixture_name"]),
        "shape_receipt.json",
        "qualification_receipt.json",
    ]


def make_student_manifest(pack_dir: Path) -> dict[str, Any]:
    cfg = read_json(pack_dir / "config.json")
    validate_student_config(cfg)
    files = pack_files(cfg)
    missing = [name for name in files if not (pack_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete student pack; missing: {', '.join(missing)}")
    return {
        "format": MANIFEST_FORMAT,
        "architecture": cfg["architecture"],
        "head": cfg["head"],
        "head_passes": cfg["head_passes"],
        "files": {
            name: {"bytes": (pack_dir / name).stat().st_size, "sha256": sha256(pack_dir / name)}
            for name in files
        },
    }


def expected_shape_receipt(cfg: dict[str, Any]) -> dict[str, Any]:
    backbone = 880_400_896
    head = 713_969_664 if cfg["head"] == "ar" else 7_346_176
    total = backbone + head
    state: dict[str, Any] = {
        "head_state": (
            {
                "lifetime": "one_frame",
                "layers": cfg["depformer_num_layers"],
                "key": ["batch", cfg["depformer_num_heads"], "codebooks<=8", 64],
                "value": ["batch", cfg["depformer_num_heads"], "codebooks<=8", 64],
            }
            if cfg["head"] == "ar"
            else {"previous_target_codes": ["batch", cfg["dep_q"]]}
        ),
        "lm_kv_per_layer": {
            "layers": 12,
            "key": ["batch", 16, "frames<=1500", 128],
            "value": ["batch", 16, "frames<=1500", 128],
        },
        "mimi_decoder_io": {
            "codes": ["batch", 8, 1],
            "pcm": ["batch", 1, 1_920],
        },
        "mimi_encoder_io": {
            "codes": ["batch", 1, 8],
            "pcm": ["batch", 1, 1_920],
        },
    }
    return {
        "format": SHAPE_FORMAT,
        "architecture": "hibiki_m_12l",
        "head": cfg["head"],
        "selected_parent_layers": cfg["selected_parent_layers"],
        "parameters": {"backbone": backbone, "head": head, "total": total},
        "estimated_weight_bytes": {
            "bf16": total * 2,
            "q4_upper_bound": total // 2 + total // 32 * 4,
        },
        "state": state,
    }


def validate_qualification(
    receipt: dict[str, Any],
    cfg: dict[str, Any],
    config_sha256: str,
    checkpoint_sha256: str,
) -> None:
    expected = {
        "format": f"hibiki_student_{'parallel' if cfg['head'] == 'parallel_v1' else 'ar'}_qualification_v1",
        "architecture": cfg["architecture"],
        "head": cfg["head"],
        "decision": "pass",
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if receipt != expected:
        raise RuntimeError("Student qualification receipt does not match the exact BF16 artifact")


def parity_metadata(path: Path, cfg: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    required = {
        "metadata_json",
        "hidden",
        "text_ids",
        "text_embedding",
        "previous_codes",
        "logits",
        "next_previous_codes",
    }
    with np.load(path, allow_pickle=False) as fixture:
        if set(fixture.files) != required:
            raise RuntimeError("Parallel parity fixture fields changed")
        metadata = json.loads(str(fixture["metadata_json"].item()))
        arrays = {name: fixture[name] for name in required - {"metadata_json"}}
    expected_shapes = {
        "hidden": [1, 1, 2_048],
        "text_ids": [1, 1],
        "text_embedding": [1, 1, 2_048],
        "previous_codes": [1, 1, 8],
        "logits": [1, 1, 8, 2_048],
        "next_previous_codes": [1, 8],
    }
    expected_metadata = {
        "format": PARITY_FORMAT,
        "architecture": cfg["architecture"],
        "head": "parallel_v1",
        "head_passes": cfg["head_passes"],
        "config_sha256": config_sha256,
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "reference_dtype": "bfloat16",
        "state_rule": "initial_card_then_previous_raw_pre_undelay_head_output",
        "shapes": expected_shapes,
    }
    if metadata != expected_metadata or HASH_RE.fullmatch(metadata["checkpoint_sha256"]) is None:
        raise RuntimeError("Parallel parity metadata does not match the pack config")
    if {name: list(value.shape) for name, value in arrays.items()} != expected_shapes:
        raise RuntimeError("Parallel parity tensor shapes changed")
    if any(not np.isfinite(arrays[name]).all() for name in ("hidden", "text_embedding", "logits")):
        raise RuntimeError("Parallel parity fixture contains non-finite values")
    if arrays["hidden"].dtype != np.float32 or arrays["text_embedding"].dtype != np.float32:
        raise RuntimeError("Parallel parity inputs must be float32")
    if arrays["logits"].dtype != np.float32:
        raise RuntimeError("Parallel parity logits must be float32")
    if any(
        arrays[name].dtype != np.int32
        for name in ("text_ids", "previous_codes", "next_previous_codes")
    ):
        raise RuntimeError("Parallel parity state tensors must be int32")
    if (arrays["previous_codes"] < 0).any() or (arrays["previous_codes"] > 2_048).any():
        raise RuntimeError("Parallel parity previous state is outside [0, card]")
    if not (arrays["previous_codes"] < 2_048).any():
        raise RuntimeError("Parallel parity fixture does not exercise raw generated codes")
    expected_next = arrays["logits"].argmax(axis=-1)[:, -1].astype(np.int32)
    if not np.array_equal(arrays["next_previous_codes"], expected_next):
        raise RuntimeError("Parallel parity next state is not the raw head argmax frame")
    return metadata


def validate_q4(path: Path) -> None:
    if path.stat().st_size > 1_000_000_000:
        raise RuntimeError("Student q4 LM exceeds the 1.0 GB pack gate")
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    bases = {name[: -len(".scales")] for name in header if name.endswith(".scales")}
    if not bases:
        raise RuntimeError("Student weights contain no q4 tensors")
    bias_bases = {name[: -len(".biases")] for name in header if name.endswith(".biases")}
    if bias_bases != bases:
        raise RuntimeError("Student q4 scale/bias tensor sets differ")
    for base in bases:
        weight = header.get(base + ".weight")
        scales = header.get(base + ".scales")
        biases = header.get(base + ".biases")
        if (
            weight is None
            or scales is None
            or biases is None
            or weight["dtype"] != "U32"
            or biases["shape"] != scales["shape"]
        ):
            raise RuntimeError(f"Malformed q4 tensor: {base}")
        in_features = weight["shape"][-1] * 8
        if scales["shape"][-1] == 0 or in_features != scales["shape"][-1] * 32:
            raise RuntimeError(f"q4 group size is not 32: {base}")


def validate_student_pack(pack_dir: Path) -> dict[str, Any]:
    cfg_path = pack_dir / "config.json"
    cfg = read_json(cfg_path)
    validate_student_config(cfg)
    manifest_path = pack_dir / "manifest.json"
    manifest = read_json(manifest_path)
    expected_manifest = make_student_manifest(pack_dir)
    if manifest != expected_manifest:
        raise RuntimeError("Student manifest content or a pack file hash changed")
    if manifest["files"][cfg["mimi_name"]]["sha256"] != MIMI_SHA256:
        raise RuntimeError("Student Mimi hash differs from the frozen codec")
    if manifest["files"][cfg["tokenizer_name"]]["sha256"] != TOKENIZER_SHA256:
        raise RuntimeError("Student tokenizer hash differs from the frozen vocabulary")
    allowed = set(pack_files(cfg)) | {"manifest.json"}
    actual = {path.name for path in pack_dir.iterdir() if path.is_file()}
    if actual != allowed:
        raise RuntimeError(
            f"Student pack files changed: missing={allowed - actual}, extra={actual - allowed}"
        )
    if read_json(pack_dir / "shape_receipt.json") != expected_shape_receipt(cfg):
        raise RuntimeError("Student shape receipt does not match the frozen config")
    config_hash = sha256(cfg_path)
    if cfg["head"] != "parallel_v1":
        raise RuntimeError("Only qualified parallel_v1 student packs are deployable")
    metadata = parity_metadata(pack_dir / cfg["parity_fixture_name"], cfg, config_hash)
    validate_qualification(
        read_json(pack_dir / "qualification_receipt.json"),
        cfg,
        config_hash,
        metadata["checkpoint_sha256"],
    )
    validate_q4(pack_dir / cfg["moshi_name"])
    if sum((pack_dir / name).stat().st_size for name in pack_files(cfg)) > 1_500_000_000:
        raise RuntimeError("Complete student pack exceeds the 1.5 GB gate")
    return cfg
