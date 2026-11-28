# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in this directory.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import sampling
from .transformer import RotatingKVCache, Transformer, TransformerConfig


@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int
    weights_per_step_schedule: list[int] | None = None
    output_layer_norm: bool = False
    # Hibiki-Zero phone student: the last slice predicts the last `parallel_codebooks` codebooks at
    # once (the config's "audio_heads": "depformer_parallel", training/student/slices.py).
    parallel_codebooks: int = 1


@dataclass
class LmConfig:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: list[int]
    # Hibiki-Zero phone student: parallel codebook heads in place of the depformer (the
    # config's "audio_heads": "parallel", training/student/heads.py).
    parallel_heads: bool = False

    @property
    def generated_codebooks(self):
        return self.depformer.num_slices + self.depformer.parallel_codebooks - 1

    @property
    def other_codebooks(self):
        return self.audio_codebooks - self.generated_codebooks

    @classmethod
    def from_config_dict(cls, data: dict) -> "LmConfig":
        # Upstream moshi's options no Hibiki-Zero model sets fail here, not at load.
        unsupported = [
            key
            for key in ("conditioners", "demux_second_stream", "cross_attention",
                        "extra_heads_num_heads", "depformer_low_rank_embeddings")
            if data.get(key)
        ]
        if data.get("layer_scale") is not None:
            unsupported.append("layer_scale")
        if data["positional_embedding"] != "rope_concat" or data["depformer_pos_emb"] != "none":
            unsupported.append("positional embeddings other than rope_concat and none")
        if unsupported:
            raise ValueError(f"unsupported model config: {', '.join(unsupported)}")
        # hibiki-zero: feedforward width is `hidden_scale * dim` (not 4*dim) and
        # the main transformer uses grouped-query attention (`kv_repeat` > 1).
        hidden_scale = data.get("hidden_scale", 4)
        audio_heads = data.get("audio_heads")
        if audio_heads not in (None, "parallel", "depformer_parallel"):
            raise ValueError(f"unsupported audio_heads {audio_heads!r}")
        parallel = data["parallel_codebooks"] if audio_heads == "depformer_parallel" else 1
        num_slices = data["dep_q"] - parallel + 1
        schedule = data.get("depformer_weights_per_step_schedule", None)
        if parallel > 1 and schedule != [min(j, num_slices - 1) for j in range(data["dep_q"])]:
            raise ValueError("a depformer_parallel model has a slice a codebook, the last one the last codebooks'")

        def scaled_dimension(dim: int) -> int:
            return int(round(hidden_scale * dim))

        transformer = TransformerConfig(
            d_model=data["dim"],
            num_heads=data["num_heads"],
            num_layers=data["num_layers"],
            dim_feedforward=scaled_dimension(data["dim"]),
            bias_ff=False,
            bias_attn=False,
            context=data["context"],
            max_period=data["max_period"],
            positional_embedding=data["positional_embedding"],
            kv_repeat=data.get("kv_repeat", 1),
            max_seq_len=4096,
        )
        depformer = DepFormerConfig(
            transformer=TransformerConfig(
                d_model=data["depformer_dim"],
                num_heads=data["depformer_num_heads"],
                num_layers=data["depformer_num_layers"],
                dim_feedforward=(
                    data.get("depformer_dim_feedforward")
                    or scaled_dimension(data["depformer_dim"])
                ),
                bias_ff=False,
                bias_attn=data.get("depformer_layer_scale", False),
                context=data.get("depformer_context", data["dep_q"]),
                max_period=data.get("depformer_max_period", 8),
                positional_embedding=data["depformer_pos_emb"],
                kv_repeat=1,
                max_seq_len=4096,
            ),
            num_slices=num_slices,
            weights_per_step_schedule=schedule,
            output_layer_norm=data.get("depformer_output_layer_norm", "depformer_norm" in data),
            parallel_codebooks=parallel,
        )
        return LmConfig(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,
            text_out_vocab_size=data["text_card"],
            audio_vocab_size=data["card"] + 1,
            audio_delays=data["delays"][1:],  # the first delay is for the text token.
            audio_codebooks=data["n_q"],
            parallel_heads=audio_heads == "parallel",
        )

    @property
    def audio_eos_token(self) -> int:
        # The source EOS is `card`, as training's caches and upstream's `add_input_eos` write
        # it: the same value as the padding token.
        return self.audio_vocab_size - 1

    @property
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1


class DepFormerSlice(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        cfg: DepFormerConfig,
    ):
        super().__init__()

        dim = cfg.transformer.d_model
        self.emb = nn.Embedding(in_vocab_size, dim)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        self.transformer = Transformer(cfg.transformer)
        # Hibiki-Zero: learned per-slice output LayerNorm applied before linear_out
        # when requested by the model config.
        self.norm = nn.LayerNorm(dim, 1e-5) if cfg.output_layer_norm else None
        self._step_c = None

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def _step(self, main_transformer_out: mx.array, last_token: mx.array, kv: list):
        # One whole slice forward for a single token, pure in the KV state: kv is
        # [[k, v], ...] per layer with width == slice index, so every shape is
        # fixed per slice and the whole step can be mx.compile'd (one compile per
        # slice). The depformer context (<= num_slices) never exceeds the buffer,
        # so no trimming or rotating is needed. Depformer has no RoPE.
        xs = self.linear_in(main_transformer_out) + self.emb(last_token)
        new_kv = []
        for idx, layer in enumerate(self.transformer.layers):
            q, k, v = layer.self_attn._qkv(layer.norm1(xs), 0)
            if kv:
                k = mx.concatenate([kv[idx][0], k], axis=2)
                v = mx.concatenate([kv[idx][1], v], axis=2)
            attn = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=layer.self_attn.scale, mask=None
            )
            xs = layer._attn_post(xs, attn)
            new_kv.append([k, v])
        if self.norm is not None:
            xs = self.norm(xs)
        return self.linear_out(xs), new_kv


class DepFormer(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        self.slices: list[DepFormerSlice] = []
        last = cfg.depformer.num_slices - 1
        self.parallel_codebooks = cfg.depformer.parallel_codebooks
        for slice_idx in range(cfg.depformer.num_slices):
            in_vs = cfg.text_in_vocab_size if slice_idx == 0 else cfg.audio_vocab_size
            # The last slice's rows j * card ... are codebook last + j's (PyTorch's linears).
            codebooks = self.parallel_codebooks if slice_idx == last else 1
            slice = DepFormerSlice(
                in_vs,
                (cfg.audio_vocab_size - 1) * codebooks,
                main_transformer_dim=cfg.transformer.d_model,
                cfg=cfg.depformer,
            )
            self.slices.append(slice)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def sample(
        self, main_transformer_out: mx.array, sampler: sampling.Sampler, text_token: mx.array
    ) -> mx.array:
        # KV threaded functionally through one compiled step per slice (fixed shapes -> each
        # compiles once), so no cache is allocated or reset a frame.
        tokens = []
        last_token = text_token
        kv: list = []
        for slice in self.slices:
            if slice._step_c is None:
                slice._step_c = mx.compile(slice._step)
            logits, kv = slice._step_c(main_transformer_out, last_token, kv)
            if slice is self.slices[-1] and self.parallel_codebooks > 1:
                # [B, 1, P * card] -> P tokens [B, 1], the last slice's codebooks at once
                b, t, _ = logits.shape
                sampled = sampler(logits.reshape(b, t, self.parallel_codebooks, -1).swapaxes(1, 2))
                tokens.extend(sampled[:, j] for j in range(self.parallel_codebooks))
                break
            last_token = sampler(logits)
            tokens.append(last_token)
        return mx.stack(tokens, axis=1)


class ParallelHeads(nn.Module):
    """Hibiki-Zero phone student's audio heads (training/student/heads.py): one linear head a
    codebook, stacked into one Linear, reading the main transformer's normed output plus the
    step's text token's embedding (`Lm.sample` adds it). All codebooks sample in one pass; the
    delays (codebook j by min(j, 8)) are the config's, which `LmGen` applies."""

    def __init__(self, dim: int, card: int, num_slices: int):
        super().__init__()
        self.num_slices = num_slices
        # Rows j * card ... (j + 1) * card - 1 are head j (PyTorch's `depformer.heads.{j}`).
        self.linear = nn.Linear(dim, num_slices * card, bias=False)

    def __call__(self, _: mx.array) -> mx.array:
        raise ValueError("not implemented")

    def sample(self, context: mx.array, sampler: sampling.Sampler) -> mx.array:
        """[B, T, dim] -> tokens [B, num_slices, T], as `DepFormer.sample` returns them."""
        b, t, _ = context.shape
        logits = self.linear(context).reshape(b, t, self.num_slices, -1).transpose(0, 2, 1, 3)
        return sampler(logits)


class Lm(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model
        self.transformer: Transformer = Transformer(cfg.transformer)
        self.depformer = (
            ParallelHeads(dim, cfg.audio_vocab_size - 1, cfg.depformer.num_slices)
            if cfg.parallel_heads
            else DepFormer(cfg)
        )
        self.text_emb = nn.Embedding(cfg.text_in_vocab_size, dim)
        self.cfg: LmConfig = cfg
        self.out_norm = nn.RMSNorm(dim, 1e-8)
        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [
            nn.Embedding(cfg.audio_vocab_size, dim) for _ in range(cfg.audio_codebooks)
        ]
        self.transformer_cache: list[RotatingKVCache] = self.transformer.make_rot_cache()

    def sample(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
    ) -> tuple[mx.array, mx.array]:
        xs = self.text_emb(text_token_ids)
        for token_ids, emb in zip(audio_token_ids, self.audio_embs):
            xs = xs + emb(token_ids)
        transformer_out = self.transformer(xs, cache=self.transformer_cache)
        transformer_out = self.out_norm(transformer_out)
        text_token = text_sampler(self.text_linear(transformer_out))
        if self.cfg.parallel_heads:
            audio_tokens = self.depformer.sample(
                transformer_out + self.text_emb(text_token), audio_sampler
            )
        else:
            audio_tokens = self.depformer.sample(transformer_out, audio_sampler, text_token)
        return text_token, audio_tokens

    def warmup(self, text_sampler: sampling.Sampler, audio_sampler: sampling.Sampler):
        """One step on the samplers a stream will use, so its compiled regions are traced, then
        an empty cache."""
        text, audio = self.sample(
            mx.array([[self.cfg.text_out_vocab_size]]),
            [mx.array([[0]])] * self.cfg.other_codebooks,
            text_sampler=text_sampler,
            audio_sampler=audio_sampler,
        )
        mx.eval(text, audio)
        for c in self.transformer_cache:
            c.reset()
