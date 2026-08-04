"""Functional recurrent adapters for the pinned Qwen3-TTS MLX code predictor."""

from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StepTiming:
    calls: int = 0
    seconds: float = 0.0


class FunctionalCodePredictor:
    """Run Qwen's 15-position code predictor with explicit array-only KV state."""

    def __init__(self, model: Any, mx: Any, nn: Any, *, compiled: bool):
        from mlx_audio.tts.models.qwen3_tts.talker import apply_rotary_pos_emb

        self.model = model
        self.mx = mx
        self.nn = nn
        self.predictor = model.talker.code_predictor
        self.inner = self.predictor.model
        self.apply_rope = apply_rotary_pos_emb
        self.compiled = compiled
        self.num_steps = self.predictor.num_code_groups - 1
        self.num_layers = len(self.inner.layers)
        self.timings = [StepTiming() for _ in range(self.num_steps)]
        self.compile_seconds = [None] * self.num_steps
        eager_steps = [self._make_step(index) for index in range(self.num_steps)]
        self.eager_steps = eager_steps
        self.steps = (
            [mx.compile(step) for step in eager_steps]
            if compiled
            else eager_steps
        )

    def _forward(self, inputs: Any, flat_kv: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
        mx = self.mx
        nn = self.nn
        if self.predictor.small_to_mtp_projection is not None:
            inputs = self.predictor.small_to_mtp_projection(inputs)

        batch, seq_len, _ = inputs.shape
        offset = 0 if not flat_kv else flat_kv[0].shape[2]
        position_ids = mx.broadcast_to(
            mx.arange(offset, offset + seq_len)[None, :], (batch, seq_len)
        )
        position_embeddings = self.inner.rotary_emb(inputs, position_ids)
        mask = None
        if seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(
                inputs.dtype
            )

        x = inputs
        new_kv = []
        for layer_index, layer in enumerate(self.inner.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attention = layer.self_attn
            q = attention.q_proj(normalized).reshape(
                batch, seq_len, attention.num_heads, attention.head_dim
            )
            k = attention.k_proj(normalized).reshape(
                batch, seq_len, attention.num_kv_heads, attention.head_dim
            )
            v = attention.v_proj(normalized).reshape(
                batch, seq_len, attention.num_kv_heads, attention.head_dim
            )
            q = mx.transpose(attention.q_norm(q), (0, 2, 1, 3))
            k = mx.transpose(attention.k_norm(k), (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            q, k = self.apply_rope(q, k, *position_embeddings)
            if flat_kv:
                k = mx.concatenate([flat_kv[2 * layer_index], k], axis=2)
                v = mx.concatenate([flat_kv[2 * layer_index + 1], v], axis=2)
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=attention.scale, mask=mask
            )
            attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
                batch, seq_len, -1
            )
            x = residual + attention.o_proj(attended)
            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
            new_kv.extend((k, v))

        return self.inner.norm(x), tuple(new_kv)

    def _make_step(self, index: int) -> Callable[..., tuple[Any, tuple[Any, ...]]]:
        if index == 0:

            def step(code_hidden: Any, token: Any) -> tuple[Any, tuple[Any, ...]]:
                first_embed = self.model.talker.get_input_embeddings()(token)
                x, kv = self._forward(
                    self.mx.concatenate([code_hidden, first_embed], axis=1), ()
                )
                return self.predictor.lm_head[0](x), kv

            return step

        def step(token: Any, *flat_kv: Any) -> tuple[Any, tuple[Any, ...]]:
            embed = self.predictor.codec_embedding[index - 1](token)
            x, kv = self._forward(embed, flat_kv)
            return self.predictor.lm_head[index](x), kv

        return step

    def run_step(
        self,
        index: int,
        code_hidden: Any,
        token: Any,
        flat_kv: tuple[Any, ...],
        *,
        evaluate: bool = False,
    ) -> tuple[Any, tuple[Any, ...]]:
        started = time.monotonic()
        if index == 0:
            logits, new_kv = self.steps[index](code_hidden, token)
        else:
            logits, new_kv = self.steps[index](token, *flat_kv)
        if evaluate:
            self.mx.eval(logits, *new_kv)
        elapsed = time.monotonic() - started
        timing = self.timings[index]
        timing.calls += 1
        timing.seconds += elapsed
        if self.compiled and self.compile_seconds[index] is None:
            self.compile_seconds[index] = elapsed
        return logits, tuple(new_kv)

    def predict(
        self,
        owner: Any,
        first_token: Any,
        hidden: Any,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        code_cache: Any = None,
    ) -> tuple[list[Any], Any]:
        del code_cache
        code_tokens = [first_token]
        code_hidden = hidden[:, -1:, :]
        flat_kv: tuple[Any, ...] = ()
        for index in range(self.num_steps):
            logits, flat_kv = self.run_step(
                index, code_hidden, code_tokens[-1], flat_kv
            )
            code_tokens.append(
                owner._sample_token_batch(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
        return code_tokens, self.mx.concatenate(code_tokens, axis=1)

    def timing_report(self) -> dict[str, Any]:
        return {
            "compiled": self.compiled,
            "closure_count": len(self.steps),
            "cold_compile_seconds_by_position": self.compile_seconds,
            "positions": [
                {"index": index, "calls": row.calls, "seconds": row.seconds}
                for index, row in enumerate(self.timings)
            ],
        }


class TalkerSplitLayer:
    """Compile fixed-shape talker work around eager cache update and SDPA."""

    def __init__(self, layer: Any, mx: Any):
        from mlx_audio.tts.models.qwen3_tts.talker import (
            apply_multimodal_rotary_pos_emb,
        )

        self.layer = layer
        self.mx = mx
        self.apply_rope = apply_multimodal_rotary_pos_emb
        self.pre = mx.compile(self._pre)
        self.post = mx.compile(self._post)
        self.pre_calls = 0
        self.post_calls = 0

    def _pre(self, x: Any, cos: Any, sin: Any) -> tuple[Any, Any, Any]:
        mx = self.mx
        attention = self.layer.self_attn
        batch, seq_len, _ = x.shape
        normalized = self.layer.input_layernorm(x)
        q = attention.q_proj(normalized).reshape(
            batch, seq_len, attention.num_heads, attention.head_dim
        )
        k = attention.k_proj(normalized).reshape(
            batch, seq_len, attention.num_kv_heads, attention.head_dim
        )
        v = attention.v_proj(normalized).reshape(
            batch, seq_len, attention.num_kv_heads, attention.head_dim
        )
        q = mx.transpose(attention.q_norm(q), (0, 2, 1, 3))
        k = mx.transpose(attention.k_norm(k), (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q, k = self.apply_rope(q, k, cos, sin)
        return q, k, v

    def _post(self, residual: Any, attended: Any) -> Any:
        mx = self.mx
        attention = self.layer.self_attn
        batch, seq_len = residual.shape[:2]
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch, seq_len, -1
        )
        x = residual + attention.o_proj(attended)
        return x + self.layer.mlp(self.layer.post_attention_layernorm(x))

    def __call__(
        self,
        x: Any,
        position_embeddings: tuple[Any, Any],
        mask: Any = None,
        cache: Any = None,
    ) -> Any:
        if x.shape[1] != 1 or cache is None:
            return self.layer(x, position_embeddings, mask, cache)
        q, k, v = self.pre(x, *position_embeddings)
        self.pre_calls += 1
        k, v = cache.update_and_fetch(k, v)
        attended = self.mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.layer.self_attn.scale, mask=mask
        )
        self.post_calls += 1
        return self.post(x, attended)


def install_talker_layer_split(model: Any, mx: Any) -> list[TalkerSplitLayer]:
    """Wrap every main-talker layer with the Hibiki-style compiled split."""

    wrappers = [TalkerSplitLayer(layer, mx) for layer in model.talker.model.layers]
    model.talker.model.layers = wrappers
    return wrappers


def install_functional_code_predictor(
    model: Any, mx: Any, nn: Any, *, compiled: bool
) -> FunctionalCodePredictor:
    """Install a repository-owned predictor adapter without changing package files."""

    adapter = FunctionalCodePredictor(model, mx, nn, compiled=compiled)

    def predict(owner: Any, first_token: Any, hidden: Any, **kwargs: Any):
        return adapter.predict(owner, first_token, hidden, **kwargs)

    model._predict_code_tokens = types.MethodType(predict, model)
    model._functional_code_predictor = adapter
    return adapter
