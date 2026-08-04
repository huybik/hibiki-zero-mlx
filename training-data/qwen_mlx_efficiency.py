"""Reusable Apple-Silicon optimizations for the pinned Qwen3-TTS MLX model."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable


SENSITIVE_PATHS = (
    "codec_embedding",
    "text_embedding",
    "codec_head",
    "lm_head",
    "text_projection",
    "small_to_mtp_projection",
    "speaker_encoder",
    "speech_tokenizer",
)


def install_full_reference_cache(model: Any, mx: Any) -> dict[str, Any]:
    """Cache all reference-derived ICL tensors while leaving target text unique.

    The installed method preserves ``Model._prepare_icl_generation_inputs``'s
    exact tensor construction. It deliberately does not replace or suppress
    ``mx.clear_cache``: allocator clearing remains owned by mlx-audio's batch
    generation boundary.
    """

    contexts: dict[tuple[str, int, float, str], dict[str, Any]] = {}
    object_keys: dict[int, tuple[str, int, float, str]] = {}
    stats = {"calls": 0, "misses": 0, "seconds": 0.0}

    def context(ref_audio: Any, ref_text: str, language: str) -> dict[str, Any]:
        import time

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
            ref_text_ids = mx.array(model.tokenizer.encode(ref_chat))[None, 3:-2]
        if icl_key not in model._icl_cache:
            mx.eval(ref_text_ids)
            model._icl_cache[icl_key] = (ref_codes, ref_text_ids)

        special = mx.array(
            [[model.config.tts_bos_token_id, model.config.tts_eos_token_id,
              model.config.tts_pad_token_id]]
        )
        special_embeds = model.talker.text_projection(
            model.talker.get_text_embeddings()(special)
        )
        tts_bos, tts_eos, tts_pad = (
            special_embeds[:, 0:1, :],
            special_embeds[:, 1:2, :],
            special_embeds[:, 2:3, :],
        )
        ref_codec = model.talker.get_input_embeddings()(ref_codes[:, 0, :])
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
            else [config.codec_think_id, config.codec_think_bos_id,
                  language_id, config.codec_think_eos_id]
        )
        prefix = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
        suffix = model.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )
        speaker = model.extract_speaker_embedding(ref_audio)
        prefix = mx.concatenate([prefix, speaker.reshape(1, 1, -1), suffix], axis=1)
        role_ids = mx.array(model.tokenizer.encode("<|im_start|>assistant\n"))[None, :]
        role = model.talker.text_projection(model.talker.get_text_embeddings()(role_ids))
        pad_count = prefix.shape[1] - 2
        combined_prefix = mx.concatenate(
            [mx.broadcast_to(tts_pad, (1, pad_count, tts_pad.shape[-1])), tts_bos],
            axis=1,
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

    def prepare(
        self: Any,
        text: str,
        ref_audio: Any,
        ref_text: str,
        language: str = "auto",
    ) -> tuple[Any, Any, Any, Any]:
        stats["calls"] += 1
        common = context(ref_audio, ref_text, language)
        target = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        target_ids = mx.array(self.tokenizer.encode(target))[None, :]
        text_ids = target_ids[:, 3:-5]
        combined_ids = mx.concatenate([common["ref_text_ids"], text_ids], axis=1)
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(combined_ids)
        )
        text_embed = mx.concatenate([text_embed, common["tts_eos"]], axis=1)
        text_with_pad = text_embed + mx.broadcast_to(
            common["codec_pad"], (1, text_embed.shape[1], text_embed.shape[-1])
        )
        codec_with_pad = common["codec_icl"] + mx.broadcast_to(
            common["tts_pad"],
            (1, common["codec_icl"].shape[1], common["tts_pad"].shape[-1]),
        )
        inputs = mx.concatenate(
            [common["role"], common["combined_prefix"], text_with_pad, codec_with_pad],
            axis=1,
        )
        return inputs, common["tts_pad"], common["tts_pad"], common["ref_codes"]

    model._prepare_icl_generation_inputs = types.MethodType(prepare, model)
    return stats


@dataclass(frozen=True)
class QuantStage:
    bits: int
    group_size: int
    selector: str


QUANTIZATION_CANDIDATES: dict[str, tuple[QuantStage, ...]] = {
    "main_mlp_q8_g64": (QuantStage(8, 64, "main_mlp"),),
    "main_all_q8_g64": (QuantStage(8, 64, "main_all"),),
    "code_predictor_q8_g64": (QuantStage(8, 64, "code_predictor"),),
    "combined_q8_g64": (
        QuantStage(8, 64, "main_all"),
        QuantStage(8, 64, "code_predictor"),
    ),
    "main_mlp_q6_rest_q8_g64": (
        QuantStage(6, 64, "main_mlp"),
        QuantStage(8, 64, "main_attention"),
        QuantStage(8, 64, "code_predictor"),
    ),
    "main_mlp_q4_rest_q8_g64": (
        QuantStage(4, 64, "main_mlp"),
        QuantStage(8, 64, "main_attention"),
        QuantStage(8, 64, "code_predictor"),
    ),
}


def _selected(path: str, selector: str) -> bool:
    main = path.startswith("talker.model.layers.")
    code = path.startswith("talker.code_predictor.model.layers.")
    if selector == "main_mlp":
        return main and ".mlp." in path
    if selector == "main_attention":
        return main and ".self_attn." in path
    if selector == "main_all":
        return main and (".mlp." in path or ".self_attn." in path)
    if selector == "code_predictor":
        return code and (".mlp." in path or ".self_attn." in path)
    raise ValueError(selector)


def quantization_inventory(model: Any, selector: str, nn: Any) -> list[dict[str, Any]]:
    rows = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear) and _selected(path, selector):
            if any(sensitive in path for sensitive in SENSITIVE_PATHS):
                raise RuntimeError(f"Sensitive module selected: {path}")
            rows.append(
                {
                    "path": path,
                    "shape": list(module.weight.shape),
                    "parameters": int(module.weight.size),
                }
            )
    return sorted(rows, key=lambda row: row["path"])


def apply_quantization(model: Any, candidate: str, nn: Any, mx: Any) -> dict[str, Any]:
    stages = QUANTIZATION_CANDIDATES[candidate]
    records = []
    claimed: set[str] = set()
    for stage in stages:
        inventory = [
            row for row in quantization_inventory(model, stage.selector, nn)
            if row["path"] not in claimed
        ]
        paths = {row["path"] for row in inventory}
        if not paths:
            raise RuntimeError(f"Quantization stage selected no modules: {stage}")

        def predicate(path: str, module: Any, selected: set[str] = paths) -> bool:
            return isinstance(module, nn.Linear) and f"talker.{path}" in selected

        nn.quantize(
            model.talker,
            bits=stage.bits,
            group_size=stage.group_size,
            class_predicate=predicate,
        )
        claimed.update(paths)
        records.append(
            {
                "selector": stage.selector,
                "bits": stage.bits,
                "group_size": stage.group_size,
                "modules": inventory,
                "module_count": len(inventory),
                "parameter_count": sum(row["parameters"] for row in inventory),
            }
        )
    mx.eval(model.parameters())
    return {
        "candidate": candidate,
        "stages": records,
        "module_count": sum(row["module_count"] for row in records),
        "parameter_count": sum(row["parameter_count"] for row in records),
        "sensitive_paths_excluded": list(SENSITIVE_PATHS),
    }


def capture_generated_codes(model: Any, mx: Any, np: Any) -> tuple[list[Any], Callable[[], None]]:
    """Capture the exact generated code arrays without replacing decoder behavior."""

    captured: list[Any] = []
    original = model._decode_icl_generated_codes

    def wrapped(self: Any, generated_codes: list[Any], ref_codes: Any) -> Any:
        value = mx.stack(generated_codes, axis=1)
        mx.eval(value)
        captured.append(np.asarray(value, dtype=np.int32))
        return original(generated_codes, ref_codes)

    model._decode_icl_generated_codes = types.MethodType(wrapped, model)

    def restore() -> None:
        model._decode_icl_generated_codes = original

    return captured, restore
