"""Profile isolated Qwen3-TTS MLX end-to-end optimization levers."""

from __future__ import annotations

import argparse
import json
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import (
    BATCH_SIZES,
    GENERATION,
    batch_path,
    generate_group,
    load_benchmark,
    runtime_environment,
)
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    package_version,
    sha256_file,
    verify_mlx_snapshot,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_optimization_profile_v2"
VARIANTS = ("speaker_cache", "retain_allocator_cache", "full_prefix_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_plan", type=Path)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def load_model() -> tuple[Any, Any, dict[str, str]]:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model as load

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    return load(root), root, verify_mlx_snapshot(root)


def install_speaker_cache(model: Any) -> dict[str, Any]:
    original = model.extract_speaker_embedding
    cache: dict[tuple[int, float], Any] = {}
    stats = {"calls": 0, "misses": 0, "seconds": 0.0}

    def cached(self: Any, audio: Any, sr: int = 24000) -> Any:
        stats["calls"] += 1
        key = (audio.size, float(audio.sum()))
        if key not in cache:
            stats["misses"] += 1
            started = time.monotonic()
            cache[key] = original(audio, sr)
            stats["seconds"] += time.monotonic() - started
        return cache[key]

    model.extract_speaker_embedding = types.MethodType(cached, model)
    return stats


def install_full_prefix_cache(model: Any, mx: Any) -> dict[str, Any]:
    """Cache every reference-derived ICL tensor; target text remains unique."""
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Qwen3BatchInputs

    contexts: dict[tuple[str, int, float, str], dict[str, Any]] = {}
    object_keys: dict[int, tuple[str, int, float, str]] = {}
    stats = {"calls": 0, "misses": 0, "seconds": 0.0}

    def context(ref_audio: Any, ref_text: str, language: str) -> dict[str, Any]:
        object_key = id(ref_audio)
        key = object_keys.get(object_key)
        if key is None:
            key = (ref_text, ref_audio.size, float(ref_audio.sum()), language.casefold())
            object_keys[object_key] = key
        if key in contexts:
            return contexts[key]
        stats["misses"] += 1
        started = time.monotonic()
        config = model.config.talker_config
        fingerprint = (ref_audio.size, float(ref_audio.sum()))
        icl_key = (ref_text, fingerprint)
        ref_codes, ref_text_ids = model._icl_cache.get(icl_key, (None, None))
        audio_for_spk = ref_audio
        if ref_codes is None:
            encoded = ref_audio
            if encoded.ndim == 1:
                encoded = encoded[None, None, :]
            elif encoded.ndim == 2:
                encoded = encoded[None, :]
            ref_codes = model.speech_tokenizer.encode(encoded)
            mx.eval(ref_codes)
        if ref_text_ids is None:
            ref_chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
            ref_ids = mx.array(model.tokenizer.encode(ref_chat))[None, :]
            ref_text_ids = ref_ids[:, 3:-2]
        if icl_key not in model._icl_cache:
            mx.eval(ref_text_ids)
            model._icl_cache[icl_key] = (ref_codes, ref_text_ids)

        tts_tokens = mx.array(
            [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]]
        )
        tts_embeds = model.talker.text_projection(
            model.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos, tts_eos, tts_pad = (
            tts_embeds[:, 0:1, :],
            tts_embeds[:, 1:2, :],
            tts_embeds[:, 2:3, :],
        )
        first = ref_codes[:, 0, :]
        ref_codec = model.talker.get_input_embeddings()(first)
        for index in range(config.num_code_groups - 1):
            ref_codec = ref_codec + model.talker.code_predictor.codec_embedding[index](
                ref_codes[:, index + 1, :]
            )
        codec_bos = model.talker.get_input_embeddings()(mx.array([[config.codec_bos_id]]))
        codec_icl = mx.concatenate([codec_bos, ref_codec], axis=1)
        codec_pad = model.talker.get_input_embeddings()(mx.array([[config.codec_pad_id]]))
        language_id = None
        if language.casefold() != "auto" and config.codec_language_id:
            language_id = config.codec_language_id.get(language.casefold())
        codec_prefill = (
            [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
            if language_id is None
            else [config.codec_think_id, config.codec_think_bos_id, language_id, config.codec_think_eos_id]
        )
        prefix = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
        suffix = model.talker.get_input_embeddings()(mx.array([[config.codec_pad_id, config.codec_bos_id]]))
        speaker = model.extract_speaker_embedding(audio_for_spk)
        prefix = mx.concatenate([prefix, speaker.reshape(1, 1, -1), suffix], axis=1)
        role_ids = mx.array(model.tokenizer.encode("<|im_start|>assistant\n"))[None, :]
        role = model.talker.text_projection(model.talker.get_text_embeddings()(role_ids))
        pad_count = prefix.shape[1] - 2
        combined_prefix = mx.concatenate(
            [mx.broadcast_to(tts_pad, (1, pad_count, tts_pad.shape[-1])), tts_bos], axis=1
        ) + prefix[:, :-1, :]
        value = {
            "ref_codes": ref_codes,
            "ref_text_ids": ref_text_ids,
            "tts_eos": tts_eos,
            "tts_pad": tts_pad,
            "codec_icl": codec_icl,
            "codec_pad": codec_pad,
            "role": role,
            "combined_prefix": combined_prefix,
        }
        mx.eval(*value.values())
        contexts[key] = value
        stats["seconds"] += time.monotonic() - started
        return value

    def prepare_one(
        self: Any,
        text: str,
        ref_audio: Any,
        ref_text: str,
        language: str = "auto",
    ) -> tuple[Any, Any, Any, Any]:
        stats["calls"] += 1
        common = context(ref_audio, ref_text, language)
        target_chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        target_ids = mx.array(self.tokenizer.encode(target_chat))[None, :]
        text_ids = target_ids[:, 3:-5]
        combined = mx.concatenate([common["ref_text_ids"], text_ids], axis=1)
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(combined)
        )
        text_embed = mx.concatenate([text_embed, common["tts_eos"]], axis=1)
        text_with_pad = text_embed + mx.broadcast_to(
            common["codec_pad"], (1, text_embed.shape[1], common["codec_pad"].shape[-1])
        )
        codec_with_pad = common["codec_icl"] + mx.broadcast_to(
            common["tts_pad"],
            (1, common["codec_icl"].shape[1], common["tts_pad"].shape[-1]),
        )
        icl = mx.concatenate([text_with_pad, codec_with_pad], axis=1)
        inputs = mx.concatenate([common["role"], common["combined_prefix"], icl], axis=1)
        return inputs, common["tts_pad"], common["tts_pad"], common["ref_codes"]

    model._prepare_icl_generation_inputs = types.MethodType(prepare_one, model)
    return stats


def run_variant(
    name: str,
    cohort: list[dict[str, Any]],
    config: dict[str, Any],
    cohort_path: Path,
    output_root: Path,
    mx: Any,
    np: Any,
    sf: Any,
) -> dict[str, Any]:
    model, _, snapshot = load_model()
    environment = runtime_environment(mx)
    patch_stats: dict[str, Any] = {}
    original_clear = mx.clear_cache
    if name == "speaker_cache":
        patch_stats = install_speaker_cache(model)
    elif name == "retain_allocator_cache":
        mx.clear_cache = lambda: None
    elif name == "full_prefix_cache":
        patch_stats = install_full_prefix_cache(model, mx)
    else:
        raise RuntimeError(name)
    records = []
    try:
        for number, group in enumerate(config["variants"]["8"], 1):
            record = generate_group(
                model,
                mx,
                np,
                sf,
                cohort,
                group,
                batch_path(output_root / name, 8, group["group_id"]),
                cohort_path,
                environment,
                snapshot,
            )
            records.append(record)
            print(f"[{name} {number}/8] {record['rows_per_minute']:.3f} rows/min", flush=True)
    finally:
        mx.clear_cache = original_clear
    wall = sum(record["wall_seconds"] for record in records)
    rows = [row for record in records for row in record["rows"]]
    baseline = {}
    for path in sorted((cohort_path.parent / "batch_size_8" / "batches").glob("*/batch.json")):
        for row in json.loads(path.read_text())["rows"]:
            baseline[row["id"]] = row["audio_sha256"]
    return {
        "schema_version": SCHEMA,
        "variant": name,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "wall_seconds": wall,
        "rows_per_minute": 60 * len(rows) / wall,
        "audio_seconds_per_wall_second": sum(row["duration_s"] for row in rows) / wall,
        "talker_seconds": sum(record["stage_timing"]["prepare_prefill_talker_seconds_reported"] for record in records),
        "decode_seconds": sum(record["stage_timing"]["sequential_decode_and_yield_seconds"] for record in records),
        "peak_mlx_memory_bytes": max(record["peak_mlx_memory_bytes"] for record in records),
        "peak_process_rss_bytes": max(record["peak_process_rss_bytes"] for record in records),
        "exact_audio_hash_matches_vs_baseline": sum(
            baseline.get(row["id"]) == row["audio_sha256"] for row in rows
        ),
        "patch_stats": patch_stats,
        "records": records,
    }


def compile_attempts(cohort: list[dict[str, Any]], config: dict[str, Any], mx: Any) -> list[dict[str, Any]]:
    attempts = []
    group = config["variants"]["8"][0]
    selected = [{row["id"]: row for row in cohort}[row_id] for row_id in group["ids"]]
    for target in ("talker", "code_predictor"):
        model, _, _ = load_model()
        started = time.monotonic()
        try:
            if target == "talker":
                model.talker = mx.compile(model.talker)
            else:
                model.talker.code_predictor = mx.compile(model.talker.code_predictor)
            list(
                model.batch_generate(
                    texts=[row["text_en"] for row in selected],
                    ref_audio=selected[0]["reference"]["reference_audio_path"],
                    ref_text=selected[0]["reference"]["reference_text_vi"],
                    max_tokens=GENERATION["max_tokens"],
                    temperature=GENERATION["temperature"],
                    top_k=GENERATION["top_k"],
                    top_p=GENERATION["top_p"],
                    repetition_penalty=GENERATION["repetition_penalty_requested"],
                    lang_code=GENERATION["lang_code"],
                    stream=False,
                )
            )
            status, error = "completed", None
        except BaseException as exc:
            status, error = "failed", {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        attempts.append(
            {
                "target": target,
                "status": status,
                "wall_seconds": time.monotonic() - started,
                "error": error,
                "reason": "test whether mx.compile supports recurrent mutable-KV module boundary",
            }
        )
    return attempts


def vocoder_microbench(cohort_path: Path, model: Any, mx: Any, np: Any) -> dict[str, Any]:
    import soundfile as sf

    first_path = next(
        iter(sorted((cohort_path.parent / "batch_size_8" / "batches").glob("*/batch.json")))
    )
    first = json.loads(first_path.read_text())
    codes = []
    for row in first["rows"]:
        audio, rate = sf.read(row["output_wav"], dtype="float32")
        if rate != 24_000:
            raise RuntimeError("Unexpected vocoder microbench rate")
        value = model.speech_tokenizer.encode(mx.array(np.asarray(audio))[None, None, :])
        mx.eval(value)
        codes.append(mx.transpose(value, (0, 2, 1)))
    started = time.monotonic()
    separate = []
    for value in codes:
        audio, lengths = model.speech_tokenizer.decode(value)
        mx.eval(audio, lengths)
        separate.append(audio[0, : int(lengths[0])])
    separate_seconds = time.monotonic() - started
    max_len = max(value.shape[1] for value in codes)
    padded = mx.concatenate(
        [
            mx.pad(value, [(0, 0), (0, max_len - value.shape[1]), (0, 0)])
            for value in codes
        ],
        axis=0,
    )
    started = time.monotonic()
    batched, lengths = model.speech_tokenizer.decode(padded)
    mx.eval(batched, lengths)
    batched_seconds = time.monotonic() - started
    max_abs = []
    for index, reference in enumerate(separate):
        candidate = batched[index, : reference.shape[0]]
        max_abs.append(float(mx.max(mx.abs(reference - candidate))))
    return {
        "rows": len(codes),
        "separate_seconds": separate_seconds,
        "batched_seconds": batched_seconds,
        "speedup": separate_seconds / batched_seconds,
        "max_abs_waveform_delta_by_row": max_abs,
        "exact_within_1e-6": all(value <= 1e-6 for value in max_abs),
        "caveat": "re-encoded preserved B8 WAVs isolate decoder execution from stochastic talker generation",
    }


def main() -> None:
    args = parse_args()
    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    results = []
    failures = []
    for name in VARIANTS:
        try:
            result = run_variant(
                name,
                cohort,
                config,
                cohort_path,
                args.output_root.expanduser().resolve(),
                mx,
                np,
                sf,
            )
            results.append(result)
        except BaseException as error:
            failures.append(
                {
                    "variant": name,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    compile_results = compile_attempts(cohort, config, mx)
    model, _, _ = load_model()
    vocoder = vocoder_microbench(cohort_path, model, mx, np)
    baseline_summary = json.loads(
        (args.report_dir.expanduser().resolve() / "benchmark_summary.json").read_text()
    )
    baseline = next(row for row in baseline_summary["variants"] if row["batch_size"] == 8)
    summary = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        "runtime": {
            "python": package_version("mlx-audio"),
            "mlx": package_version("mlx"),
        },
        "baseline": baseline,
        "variants": [
            {
                key: value
                for key, value in row.items()
                if key != "records"
            }
            | {"throughput_gain_vs_baseline": row["rows_per_minute"] / baseline["rows_per_minute"] - 1}
            for row in results
        ],
        "compile_attempts": compile_results,
        "batch_vocoder_microbench": vocoder,
        "failures": failures,
        "source_evidence": {
            "talker_rotary_and_swiglu_already_compiled": True,
            "vocoder_decoder_already_compiled_at_model_load": True,
            "batch_generate_grows_attention_mask_with_per_step_concatenate": True,
            "batch_generate_repetition_penalty_has_python_batch_loop": True,
            "batch_generate_decodes_each_completed_sequence_serially": True,
        },
    }
    report_dir = args.report_dir.expanduser().resolve()
    atomic_write_bytes(report_dir / "optimization_profile.json", json_bytes(summary))
    atomic_write_bytes(
        report_dir / "optimization_raw_results.jsonl",
        jsonl_bytes([record for row in results for record in row["records"]]),
    )
    atomic_write_bytes(report_dir / "optimization_failures.jsonl", jsonl_bytes(failures))
    print(json.dumps(summary["variants"], indent=2))


if __name__ == "__main__":
    main()
