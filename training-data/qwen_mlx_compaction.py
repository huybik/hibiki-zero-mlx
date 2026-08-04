"""Deterministic active-lane generation for the pinned Qwen3-TTS MLX model."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


RNG_SCHEMA = "hibiki-qwen-mlx-row-rng-v1"


@dataclass
class LaneGeneration:
    codes: list[list[Any]]
    token_ids: list[list[int]]
    token_caps: list[int]
    stop_reasons: list[str]
    generation_seconds: float
    prefill_seconds: float
    talker_lane_steps: int
    useful_talker_lane_steps: int
    predictor_lane_steps: int
    useful_predictor_lane_steps: int
    active_widths: list[int]
    completed_order: list[int]


def row_root_digest(campaign_revision: str, row_id: str, attempt: int) -> bytes:
    payload = (
        f"{RNG_SCHEMA}\0{campaign_revision}\0{row_id}\0attempt={attempt}".encode()
    )
    return hashlib.sha256(payload).digest()


def frame_keys(
    campaign_revision: str,
    row_id: str,
    attempt: int,
    frame: int,
    mx: Any,
) -> Any:
    """Return 16 keys: talker first, then the 15 codebook predictors."""
    root = row_root_digest(campaign_revision, row_id, attempt)
    frame_digest = hashlib.sha256(root + frame.to_bytes(8, "big")).digest()
    seed = int.from_bytes(frame_digest[:4], "big")
    return mx.random.split(mx.random.key(seed), num=16)


def rng_contract(campaign_revision: str) -> dict[str, Any]:
    return {
        "schema_version": RNG_SCHEMA,
        "campaign_revision": campaign_revision,
        "root": "SHA256(schema + NUL + campaign_revision + NUL + row_id + NUL + attempt)",
        "frame": "SHA256(root_digest + uint64_be(frame_index))",
        "mlx_frame_key": "mx.random.key(uint32_be(frame_digest[0:4]))",
        "split": "mx.random.split(frame_key, 16)",
        "key_assignment": {"0": "main talker token", "1..15": "codebooks 1..15"},
        "properties": [
            "row output is independent of batch position, peer rows, scheduling, and compaction",
            "attempt 1 is distinct from attempt 0",
            "frame index counts accepted non-EOS codec frames from zero",
        ],
    }


def _sample(
    logits: Any,
    *,
    key: Any,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens: list[int] | None = None,
    suppress_tokens: list[int] | None = None,
    eos_token_id: int | None = None,
) -> Any:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.sample_utils import apply_top_k, apply_top_p

    value = logits[:, -1, :]
    if suppress_tokens:
        indices = mx.array(suppress_tokens, dtype=mx.int32)
        value = mx.put_along_axis(
            value,
            indices[None, :],
            mx.array(float("-inf"), value.dtype),
            axis=-1,
        )
    if generated_tokens and repetition_penalty != 1.0:
        valid = sorted({token for token in generated_tokens if token < value.shape[-1]})
        if valid:
            indices = mx.array(valid, dtype=mx.int32)
            selected = mx.take(value, indices, axis=-1)
            penalized = mx.where(
                selected < 0,
                selected * repetition_penalty,
                selected / repetition_penalty,
            )
            value = mx.put_along_axis(value, indices[None, :], penalized, axis=-1)
    if temperature <= 0:
        return mx.argmax(value, axis=-1, keepdims=True)
    if temperature != 1.0:
        value = value / temperature
    eos_logit = None
    if eos_token_id is not None and eos_token_id < value.shape[-1]:
        eos_logit = value[:, eos_token_id : eos_token_id + 1]
    if 0 < top_k < value.shape[-1]:
        value = apply_top_k(value, top_k)
    if 0.0 < top_p < 1.0:
        logprobs = apply_top_p(nn.log_softmax(value, axis=-1), top_p)
        value = mx.where(logprobs == -mx.inf, -float("inf"), value)
    if eos_logit is not None:
        value = mx.put_along_axis(
            value, mx.array([[eos_token_id]], dtype=mx.int32), eos_logit, axis=-1
        )
    return mx.random.categorical(value, key=key)[:, None]


def _sample_rows(
    logits: Any,
    originals: list[int],
    rows: list[dict[str, Any]],
    frames: list[int],
    attempts: list[int],
    campaign_revision: str,
    *,
    key_index: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens: list[list[int]] | None = None,
    suppress_tokens: list[int] | None = None,
    eos_token_id: int | None = None,
) -> Any:
    import mlx.core as mx

    sampled = []
    for lane, original in enumerate(originals):
        keys = frame_keys(
            campaign_revision,
            str(rows[original]["id"]),
            attempts[original],
            frames[original],
            mx,
        )
        sampled.append(
            _sample(
                logits[lane : lane + 1],
                key=keys[key_index],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(
                    generated_tokens[original] if generated_tokens is not None else None
                ),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )
        )
    return mx.concatenate(sampled, axis=0)


def _filter_talker_cache(cache: list[Any], indices: list[int], mx: Any) -> None:
    selection = mx.array(indices, dtype=mx.int32)
    for layer in cache:
        if layer.keys is not None:
            layer.keys = mx.take(layer.keys, selection, axis=0)
            layer.values = mx.take(layer.values, selection, axis=0)


def _predict_codes(
    model: Any,
    hidden: Any,
    first: Any,
    originals: list[int],
    rows: list[dict[str, Any]],
    frames: list[int],
    attempts: list[int],
    campaign_revision: str,
    generation: dict[str, Any],
    adapter: Any = None,
) -> tuple[list[Any], Any]:
    import mlx.core as mx

    tokens = [first]
    flat_kv: tuple[Any, ...] = ()
    cache = None if adapter is not None else model.talker.code_predictor.make_cache()
    for index in range(model.config.talker_config.num_code_groups - 1):
        if adapter is not None:
            logits, flat_kv = adapter.run_step(
                index, hidden[:, -1:, :], tokens[-1], flat_kv
            )
        else:
            if index == 0:
                embedded = model.talker.get_input_embeddings()(tokens[-1])
                inputs = mx.concatenate([hidden[:, -1:, :], embedded], axis=1)
            else:
                inputs = model.talker.code_predictor.codec_embedding[index - 1](
                    tokens[-1]
                )
            logits, cache, _ = model.talker.code_predictor(
                inputs, cache=cache, generation_step=index
            )
        tokens.append(
            _sample_rows(
                logits,
                originals,
                rows,
                frames,
                attempts,
                campaign_revision,
                key_index=index + 1,
                temperature=generation["temperature"],
                top_k=generation["top_k"],
                top_p=generation["top_p"],
            )
        )
    return tokens, mx.concatenate(tokens, axis=1)


def generate_lanes(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    ref_audio: Any,
    ref_text: str,
    generation: dict[str, Any],
    campaign_revision: str,
    attempts: list[int] | None = None,
    compact: bool,
    adapter: Any = None,
) -> LaneGeneration:
    """Generate codes while preserving original-row ownership across compaction."""
    import mlx.core as mx

    size = len(rows)
    if not size:
        raise ValueError("Cannot generate an empty batch")
    attempts = attempts or [0] * size
    inputs = model._prepare_batch_inputs(
        [str(row["text_en"]) for row in rows],
        language=generation["lang_code"],
        ref_audio=ref_audio,
        ref_text=ref_text,
        return_metadata=True,
    )
    mx.eval(
        inputs.input_embeds,
        inputs.trailing_text_hidden,
        inputs.tts_pad_embed,
        inputs.attention_mask,
    )
    current = inputs.input_embeds
    trailing = inputs.trailing_text_hidden
    attention = inputs.attention_mask
    pad = inputs.tts_pad_embed
    cache = model.talker.make_cache()
    active = list(range(size))
    generated: list[list[Any]] = [[] for _ in rows]
    token_ids: list[list[int]] = [[] for _ in rows]
    frames = [0] * size
    token_caps = [
        min(
            int(generation["max_tokens"]),
            max(75, len(model.tokenizer.encode(str(row["text_en"]))) * 6),
        )
        for row in rows
    ]
    stop_reasons = [""] * size
    finished = [False] * size
    trailing_indices = mx.zeros((size, 1), dtype=mx.int32)
    eos = int(model.config.talker_config.codec_eos_token_id)
    suppressed = model._suppress_codec_tokens(eos)
    talker_lanes = useful_talker_lanes = 0
    predictor_lanes = useful_predictor_lanes = 0
    active_widths = []
    completed_order = []
    started = time.monotonic()
    prefill_seconds = 0.0
    step = 0
    while active and step < int(generation["max_tokens"]):
        active_widths.append(len(active))
        talker_lanes += len(active)
        useful = [lane for lane, original in enumerate(active) if not finished[original]]
        useful_talker_lanes += len(useful)
        call_started = time.monotonic()
        logits, hidden = model.talker(current, cache=cache, attention_mask=attention)
        if step == 0:
            mx.eval(logits, hidden)
            prefill_seconds = time.monotonic() - call_started
        sampled = _sample_rows(
            logits,
            active,
            rows,
            frames,
            attempts,
            campaign_revision,
            key_index=0,
            temperature=generation["temperature"],
            top_k=generation["top_k"],
            top_p=generation["top_p"],
            repetition_penalty=generation["repetition_penalty_requested"],
            generated_tokens=token_ids,
            suppress_tokens=suppressed,
            eos_token_id=eos,
        )
        mx.eval(sampled)
        sampled_cpu = [int(value) for value in sampled[:, 0].tolist()]
        survivors = []
        for lane, original in enumerate(active):
            if finished[original]:
                continue
            if sampled_cpu[lane] == eos:
                finished[original] = True
                stop_reasons[original] = "eos"
                completed_order.append(original)
            else:
                survivors.append(lane)

        if compact:
            if not survivors:
                break
            _filter_talker_cache(cache, survivors, mx)
            selection = mx.array(survivors, dtype=mx.int32)
            hidden_for_codes = mx.take(hidden, selection, axis=0)
            first_for_codes = mx.take(sampled, selection, axis=0)
            code_originals = [active[lane] for lane in survivors]
        else:
            hidden_for_codes = hidden
            eos_fill = mx.full((len(active), 1), eos, dtype=mx.int32)
            alive_mask = mx.array(
                [not finished[original] for original in active], dtype=mx.bool_
            )
            first_for_codes = mx.where(alive_mask[:, None], sampled, eos_fill)
            code_originals = list(active)

        predictor_lanes += len(code_originals) * 15
        useful_predictor_lanes += len(survivors) * 15
        code_tokens, all_codes = _predict_codes(
            model,
            hidden_for_codes,
            first_for_codes,
            code_originals,
            rows,
            frames,
            attempts,
            campaign_revision,
            generation,
            adapter,
        )
        mx.eval(all_codes)
        all_codes_by_original = {
            original: all_codes[lane : lane + 1]
            for lane, original in enumerate(code_originals)
        }
        for lane in survivors:
            original = active[lane]
            token_ids[original].append(sampled_cpu[lane])
            generated[original].append(all_codes_by_original[original])
            frames[original] += 1
            if frames[original] >= token_caps[original]:
                finished[original] = True
                stop_reasons[original] = "token_cap"
                completed_order.append(original)

        if compact:
            keep_originals = [original for original in code_originals if not finished[original]]
            if not keep_originals:
                break
            keep_in_codes = [
                index for index, original in enumerate(code_originals) if not finished[original]
            ]
            keep_in_active = [survivors[index] for index in keep_in_codes]
            if len(keep_in_codes) != len(code_originals):
                _filter_talker_cache(cache, keep_in_codes, mx)
            selection = mx.array(keep_in_codes, dtype=mx.int32)
            active_selection = mx.array(keep_in_active, dtype=mx.int32)
            code_tokens = [mx.take(value, selection, axis=0) for value in code_tokens]
            trailing = mx.take(trailing, active_selection, axis=0)
            trailing_indices = mx.take(
                trailing_indices, active_selection, axis=0
            )
            current = model._next_batch_input_embeds(
                trailing,
                pad,
                trailing_indices,
                code_tokens,
                pad_when_index_clamped=True,
            )
            trailing_indices = trailing_indices + 1
            attention = mx.take(
                attention, active_selection, axis=0
            )
            attention = mx.concatenate(
                [attention, mx.ones((len(keep_originals), 1))], axis=1
            )
            active = keep_originals
        else:
            current = model._next_batch_input_embeds(
                trailing,
                pad,
                trailing_indices,
                code_tokens,
                pad_when_index_clamped=True,
            )
            advance = mx.array(
                [not finished[original] for original in active], dtype=mx.int32
            )[:, None]
            trailing_indices = trailing_indices + advance
            attention = mx.concatenate(
                [attention, mx.ones((len(active), 1))], axis=1
            )
            if all(finished):
                break
        mx.eval(current, attention)
        step += 1

    for original in range(size):
        if not stop_reasons[original]:
            stop_reasons[original] = "max_tokens"
            completed_order.append(original)
    return LaneGeneration(
        codes=generated,
        token_ids=token_ids,
        token_caps=token_caps,
        stop_reasons=stop_reasons,
        generation_seconds=time.monotonic() - started,
        prefill_seconds=prefill_seconds,
        talker_lane_steps=talker_lanes,
        useful_talker_lane_steps=useful_talker_lanes,
        predictor_lane_steps=predictor_lanes,
        useful_predictor_lane_steps=useful_predictor_lanes,
        active_widths=active_widths,
        completed_order=completed_order,
    )
