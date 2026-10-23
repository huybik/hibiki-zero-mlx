"""VieNeu-TTS v3 Turbo in MLX: batched, voice-cloned Vietnamese speech on Apple Silicon.

A port of the batched engine in vieneu 3.8.3 (``vieneu.v3_turbo_serve``) on the
``update/`` weights. Each frame the acoustic decoder samples the 16 MOSS codes from the
backbone's last hidden state and the backbone reads them back. Unlike upstream's static batches,
:class:`Generator` batches continuously: a chunk that stops frees its slot for the next prompt.
The MOSS codec is mlx-audio's port, with windowed attention and its transformers in fp16.
Reference enrollment (x-vector) is upstream's ONNX code on the CPU.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soxr
from huggingface_hub import snapshot_download
from mlx.utils import tree_map
from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import MultiheadAttention
from vieneu_utils.core_utils import max_expected_frames

REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
REVISION = "61b85e3d937fbbacb387714180e8182823512523"
SUBFOLDER = "update"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CODEC_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"

SAMPLE_RATE = 48_000
FRAME_SAMPLES = 3840  # one MOSS frame, 12.5 Hz
REF_SECONDS = 8.0  # upstream trims references to this
ENCODER_PAD_CODE = 455

# Upstream's defaults (VieNeu.infer and V3TurboBatchEngine).
TEMPERATURE = 0.8
TOP_K = 25
TOP_P = 0.95
REPETITION_PENALTY = 1.2
REPETITION_WINDOW = 64
MAX_NEW_FRAMES = 300
RETRIES = 2  # chunks that stop implausibly early are generated again, keeping the longest
CHECK_EVERY = 8  # frames between the host's looks at which slots finished
ENCODE_FRAMES = 3200  # reference frames encoded in one codec call (batch x longest): 32 of 8 s
DECODE_FRAMES = 2048  # codes decoded in one codec call (batch x longest row)
# MLX keeps freed buffers for reuse. The codec's shapes change every call, so without a limit its
# cache grew to 27 GB while 2 GB was in use.
CACHE_BYTES = 2 * 1024**3
_MARKUP = re.compile(r"<\|emotion_\d+\|>|</?en>")


def model_dir() -> Path:
    return Path(
        snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=[f"{SUBFOLDER}/*", "speaker_encoder.onnx", "denoiser.onnx"],
        )
    )


def min_expected_frames(phonemes: str) -> int:
    """Upstream's early-EOS floor (v3_turbo_serve.engine): 0.25 frames per phoneme character."""
    return max(3, math.ceil(0.25 * len(_MARKUP.sub("", phonemes))))


def _groups(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices in order of length, cut into groups whose size x longest fits ``budget``."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    groups, s = [], 0
    while s < len(order):
        e = s + 1
        while e < len(order) and (e - s + 1) * lengths[order[e]] <= budget:
            e += 1
        groups.append(order[s:e])
        s = e
    return groups


def trim_silence(wav: np.ndarray, top_db: float = 30.0, frame: int = 2048, hop: int = 512) -> np.ndarray:
    """librosa.effects.trim, which upstream runs on every reference: drop edges quieter than
    ``top_db`` below the loudest frame."""
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wav, frame // 2), frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(rms.max(), 1e-10))
    loud = np.flatnonzero(db > -top_db)
    if not len(loud):
        return wav
    return wav[loud[0] * hop : min(len(wav), (loud[-1] + 1) * hop)]


@dataclass
class Voice:
    speaker_emb: np.ndarray  # (192,) x-vector
    ref: np.ndarray  # 48 kHz mono reference, encoded into the in-context codes


class Enroller:
    """Upstream's reference enrollment on the CPU, without its denoiser: trim the edges, keep 8 s,
    take the x-vector. The denoiser costs 3.6 CPU-seconds a clip and measured no better voice
    match or intelligibility on en_speech."""

    def __init__(self):
        import onnxruntime as ort
        from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.speaker = OnnxSpeakerEncoder(str(model_dir() / "speaker_encoder.onnx"), sess_options=so)

    def __call__(self, wav: np.ndarray, sr: int) -> Voice:
        wav = trim_silence(wav)[: int(REF_SECONDS * sr)]
        return Voice(self.speaker.embed(wav, sr), soxr.resample(wav, sr, SAMPLE_RATE).astype(np.float32))


class WindowedAttention(MultiheadAttention):
    """The codec's causal attention over its last ``context`` positions, in blocks of half the
    context that each attend only the keys in their band. mlx-audio's version builds a (T, T)
    mask and attends every key, 2.7x slower on 20 s rows at twice the memory. Same outputs: decoded
    audio within 2e-5, reference codes identical."""

    def __call__(self, x: mx.array, input_lengths: mx.array) -> mx.array:
        B, T, _ = x.shape
        H, D, W = self.num_heads, self.head_dim, self.context
        qkv = self.in_proj(x).reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)
        rope = dict(traditional=True, base=self.max_period, scale=1.0, offset=0)
        q, k, v = mx.fast.rope(qkv[0], D, **rope), mx.fast.rope(qkv[1], D, **rope), qkv[2]

        def attend(q, k, v, mask):
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)

        out = attend(q[:, :, :W], k[:, :, :W], v[:, :, :W], "causal")  # the first W see no further back
        if T > W:
            C = W // 2
            n = -(-T // C)
            pad = [(0, 0), (0, 0), (0, n * C - T), (0, 0)]
            q, k, v = (mx.pad(a, pad).reshape(B, H, n, C, D) for a in (q, k, v))
            # Block i >= 2 attends blocks i-2..i: key j of 3C is allowed for query i when i < j <= i + W.
            k, v = (mx.concatenate([a[:, :, s : n - 2 + s] for s in range(3)], axis=3) for a in (k, v))

            def fold(a):
                return a[:, :, -(n - 2) :].transpose(0, 2, 1, 3, 4).reshape(B * (n - 2), H, a.shape[3], D)

            i, j = mx.arange(C)[:, None], mx.arange(3 * C)[None, :]
            tail = attend(fold(q), fold(k), fold(v), (j > i) & (j <= i + W))
            tail = tail.reshape(B, n - 2, H, C, D).transpose(0, 2, 1, 3, 4).reshape(B, H, -1, D)
            out = mx.concatenate([out, tail], axis=2)[:, :, :T]
        valid = mx.arange(T)[None, None, :, None] < input_lengths[:, None, None, None]
        out = mx.where(valid, out, mx.zeros((), dtype=out.dtype))
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim))


class Attention(nn.Module):
    """Qwen3 attention: GQA, RMSNorm on each head's queries and keys, RoPE."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden, self.heads, self.kv_heads = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.rope_theta = cfg["rope_theta"]
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

    def project(self, x: mx.array, offset):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        rope = dict(traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        return mx.fast.rope(q, self.head_dim, **rope), mx.fast.rope(k, self.head_dim, **rope), v

    def attend(self, q, k, v, mask) -> mx.array:
        B, _, T, _ = q.shape
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BackboneLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])


class Backbone(nn.Module):
    """Each cache row holds one prompt from column 0, then a column per frame fed back."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = [BackboneLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def prefill(self, x: mx.array, cache, rows: mx.array) -> mx.array:
        """Right-padded prompts ``x`` (G, T, H) into cache ``rows`` at columns 0..T-1."""
        T = x.shape[1]
        for layer, (keys, values) in zip(self.layers, cache):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), 0)
            keys[rows, :, :T] = k
            values[rows, :, :T] = v
            x = x + layer.self_attn.attend(q, k, v, "causal")
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)

    def step(self, x: mx.array, cache, pos: mx.array, width: int) -> mx.array:
        """One input per row, ``x`` (B, 1, H), written at column ``pos`` (B,) and attending
        columns 0..pos; ``width`` bounds pos + 1."""
        rows = mx.arange(x.shape[0])
        mask = (mx.arange(width)[None] <= pos[:, None])[:, None, None]
        for layer, (keys, values) in zip(self.layers, cache):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), pos)
            keys[rows, :, pos] = k[:, :, 0]
            values[rows, :, pos] = v[:, :, 0]
            x = x + layer.self_attn.attend(q, keys[:, :, :width], values[:, :, :width], mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)


class AcousticLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden, intermediate = cfg["hidden_size"], cfg["local_intermediate_size"]
        self.heads = cfg["local_num_attention_heads"]
        self.head_dim = hidden // self.heads
        self.norm1 = nn.RMSNorm(hidden, eps=cfg["rms_norm_eps"])
        self.norm2 = nn.RMSNorm(hidden, eps=cfg["rms_norm_eps"])
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.attn.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.attn.q_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.attn.k_norm = nn.RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.ff_gate = nn.Linear(hidden, intermediate, bias=False)
        self.ff_up = nn.Linear(hidden, intermediate, bias=False)
        self.ff_down = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x, past):
        """New slots ``x`` (B, S, H) attend the cached slots ``past`` and, causally, each other."""
        B, S, H = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(B, S, 3, self.heads, self.head_dim)
        q = self.attn.q_norm(qkv[:, :, 0]).transpose(0, 2, 1, 3)
        k = self.attn.k_norm(qkv[:, :, 1]).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if past is not None:
            k = mx.concatenate([past[0], k], axis=2)
            v = mx.concatenate([past[1], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask="causal" if S > 1 else None
        )
        x = x + self.attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, S, H))
        h = self.norm2(x)
        return x + self.ff_down(nn.silu(self.ff_gate(h)) * self.ff_up(h)), (k, v)


class AcousticDecoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = [AcousticLayer(cfg) for _ in range(cfg["local_num_hidden_layers"])]
        self.slot_pos_emb = nn.Embedding(cfg["n_vq"] + 1, cfg["hidden_size"])
        self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def __call__(self, x, slot: int, cache):
        x = x + self.slot_pos_emb.weight[slot : slot + x.shape[1]]
        new_cache = []
        for layer, past in zip(self.layers, cache):
            x, kv = layer(x, past)
            new_cache.append(kv)
        return self.norm(x), new_cache


class VieNeuTurbo(nn.Module):
    """The network; generation lives in :class:`Generator`."""

    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg["hidden_size"]
        self.text_embeddings = nn.Embedding(cfg["text_vocab_size"], H)
        self.audio_embeddings = [nn.Embedding(cfg["audio_vocab_size"], H) for _ in range(cfg["n_vq"])]
        self.semantic_backbone = Backbone(cfg)
        self.acoustic_decoder = AcousticDecoder(cfg)
        self.xvec_proj = [nn.Linear(cfg["speaker_embedding_dim"], H), nn.LayerNorm(H)]


def _sample(logits, penalized, temperature, top_k, top_p, penalty):
    """Upstream's sampler: CTRL repetition penalty, temperature, top-k, then top-p within the k."""
    logits = mx.where(penalized, mx.where(logits < 0, logits * penalty, logits / penalty), logits)
    logits = logits / temperature
    idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[:, :top_k]
    cand = mx.take_along_axis(logits, idx, axis=-1)
    order = mx.argsort(-cand, axis=-1)
    cand = mx.take_along_axis(cand, order, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    probs = mx.softmax(cand, axis=-1)
    keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
    choice = mx.random.categorical(mx.where(keep, cand, -mx.inf))
    return mx.take_along_axis(idx, choice[:, None], axis=-1)[:, 0]


class Synthesizer:
    """The model and the MOSS codec: reference encoding, prompts, frame sampling and decoding."""

    def __init__(self, dtype=mx.bfloat16):
        mx.set_cache_limit(CACHE_BYTES)
        root = model_dir()
        self.cfg = json.loads((root / SUBFOLDER / "config.json").read_text())
        self.n_vq = self.cfg["n_vq"]
        self.model = VieNeuTurbo(self.cfg)
        weights = mx.load(str(root / SUBFOLDER / "model.safetensors"))
        self.model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
        mx.eval(self.model.parameters())
        m = self.model
        # The output heads are tied to the embeddings.
        self.audio_table = mx.stack([e.weight for e in m.audio_embeddings])  # (n_vq, V, H)
        self.audio_flat = self.audio_table.reshape(-1, self.audio_table.shape[-1])
        self.code_offsets = mx.arange(self.n_vq) * self.cfg["audio_vocab_size"]
        sgs = self.cfg["speech_generation_start_token_id"]
        self.sgs_emb = m.text_embeddings.weight[sgs]
        self.eos_id = self.cfg["speech_generation_end_token_id"]
        self.audio_pad = self.cfg["audio_pad_token_id"]
        self.state = [mx.random.state]
        self.frame = mx.compile(self._frame, inputs=self.state, outputs=self.state)

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(root / SUBFOLDER / "tokenizer.json"))
        self.codec = MossAudioTokenizer.from_pretrained(
            snapshot_download(CODEC_REPO, revision=CODEC_REVISION, allow_patterns=["*.json", "*.safetensors"])
        )
        for module in self.codec.modules():
            if isinstance(module, MultiheadAttention):
                module.__class__ = WindowedAttention
        # The encoder and decoder transformers run in fp16, the quantizer in fp32. Measured against
        # fp32 (2026-10-24): decoding 1.6x faster with audio 53 dB from fp32's (bf16: 35 dB), far
        # finer than Opus at 32 kbps; encoding 1.5x faster with 96.8% of reference codes the same,
        # and voice similarity and Whisper CER on 120 validation rows unchanged within noise.
        for module in self.codec.encoder + self.codec.decoder:
            module.update(tree_map(lambda p: p.astype(mx.float16), module.parameters()))

    def encode_refs(self, refs: list[np.ndarray]) -> list[np.ndarray]:
        """48 kHz mono references -> MOSS codes (frames, n_vq). Each is zero-padded to whole
        frames first, as upstream does since its #198 fix (a codec-padded tail encodes a blip)."""
        frames = [-(-len(r) // FRAME_SAMPLES) for r in refs]
        out: list[np.ndarray | None] = [None] * len(refs)
        for idx in _groups(frames, ENCODE_FRAMES):
            n = [frames[i] for i in idx]
            x = np.zeros((len(idx), 2, n[-1] * FRAME_SAMPLES), np.float32)  # the codec is stereo
            for j, i in enumerate(idx):
                x[j, :, : len(refs[i])] = refs[i]
            codes, _, _ = self.codec._encode_frame(mx.array(x, mx.float16), mx.array(n) * FRAME_SAMPLES)
            codes = np.array(codes)
            for j, i in enumerate(idx):
                out[i] = codes[:, j, : n[j]].T
        return out

    def decode(self, codes: list[np.ndarray]) -> list[np.ndarray]:
        """MOSS codes (frames, n_vq) -> 48 kHz mono audio (the mean of the codec's two channels)."""
        out: list[np.ndarray | None] = [None] * len(codes)
        for idx in _groups([len(c) for c in codes], DECODE_FRAMES):
            lengths = [len(codes[i]) for i in idx]
            x = np.zeros((self.n_vq, len(idx), max(1, lengths[-1])), np.int32)
            for j, i in enumerate(idx):
                x[:, j, : lengths[j]] = codes[i].T
            hidden, n = self.codec.quantizer.decode_codes(mx.array(x)).astype(mx.float16), mx.array(lengths)
            for module in self.codec.decoder:
                hidden, n = module(hidden, n)
            audio, _ = self.codec._restore_channels_from_codec(hidden, n)
            audio = np.array(audio.mean(axis=1))
            if not np.isfinite(audio).all():
                raise FloatingPointError("the fp16 codec decoder overflowed")
            for j, i in enumerate(idx):
                out[i] = audio[j, : lengths[j] * FRAME_SAMPLES]
        return out

    def prompt_rows(self, phonemes: str, ref_codes: np.ndarray) -> np.ndarray:
        """Upstream's build_prompt_2d: style token, phonemes, then the reference frames."""
        c = self.cfg
        ids = [c["default_style_token_id"], c["text_prompt_start_token_id"]]
        ids += self.tokenizer.encode(phonemes, add_special_tokens=False).ids
        ids += [c["text_prompt_end_token_id"]]
        text = np.full((len(ids), self.n_vq + 1), self.audio_pad, dtype=np.int32)
        text[:, 0] = ids
        ref = np.full((len(ref_codes), self.n_vq + 1), self.audio_pad, dtype=np.int32)
        ref[:, 0] = c["audio_ref_slot_token_id"]
        ref[:, 1:] = ref_codes
        return np.concatenate([text, ref])

    def speaker_anchor(self, speaker_emb: mx.array) -> mx.array:
        linear, norm = self.model.xvec_proj
        return norm(linear(speaker_emb.astype(linear.weight.dtype)))

    def embed_rows(self, rows: mx.array, anchor: mx.array) -> mx.array:
        """(B, T, n_vq+1) prompt rows -> (B, T, H), audio pads contributing nothing."""
        x = self.model.text_embeddings(rows[..., 0])
        codes = rows[..., 1:]
        valid = codes != self.audio_pad
        audio = self.audio_flat[mx.where(valid, codes, 0) + self.code_offsets] * valid[..., None].astype(x.dtype)
        return x + audio.sum(axis=-2) + anchor[:, None, :]

    def _frame(self, hidden, penalized):
        """One audio frame for every row: (codes (B, n_vq), eos (B,))."""
        m = self.model
        dec = m.acoustic_decoder
        x = mx.stack([hidden, mx.broadcast_to(self.sgs_emb, hidden.shape)], axis=1)
        out, cache = dec(x, 0, [None] * len(dec.layers))
        eos = mx.argmax(out[:, 0] @ m.text_embeddings.weight.T, axis=-1) == self.eos_id
        sample = dict(
            temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, penalty=REPETITION_PENALTY
        )
        codes = [_sample((out[:, 1] @ self.audio_table[0].T).astype(mx.float32), penalized[:, 0], **sample)]
        for ch in range(1, self.n_vq):
            x = self.audio_table[ch - 1][codes[-1]][:, None, :]
            out, cache = dec(x, ch + 1, cache)
            logits = (out[:, 0] @ self.audio_table[ch].T).astype(mx.float32)
            codes.append(_sample(logits, penalized[:, ch], **sample))
        return mx.stack(codes, axis=1), eos


@dataclass
class _Job:
    key: object
    prompt: np.ndarray  # (T, n_vq+1)
    anchor: mx.array  # (H,)
    cap: int
    floor: int
    tries: int = 0
    best: np.ndarray | None = None


class Generator:
    """Continuous batching over ``batch`` slots. Every CHECK_EVERY frames the host collects the
    chunks that stopped (EOS, kept, or their frame cap) and prefills queued prompts into the freed
    slots, so no slot idles until the batch's longest chunk ends. A chunk that stops implausibly
    early is queued again, up to RETRIES times, and keeps its longest take."""

    def __init__(self, synth: Synthesizer, batch: int):
        self.s = synth
        cfg = synth.cfg
        self.width = cfg["max_position_embeddings"]  # cache columns: prompt plus frames
        dtype = synth.audio_table.dtype
        shape = (batch, cfg["num_key_value_heads"], self.width, cfg["head_dim"])
        self.cache = [(mx.zeros(shape, dtype), mx.zeros(shape, dtype)) for _ in synth.model.semantic_backbone.layers]
        H, V = cfg["hidden_size"], cfg["audio_vocab_size"]
        self.vocab = mx.arange(V)
        self.hidden = mx.zeros((batch, H), dtype)
        self.step_input = mx.zeros((batch, H), dtype)  # SGS token + speaker anchor
        self.counts = mx.zeros((batch, synth.n_vq, V), mx.int32)  # codes in the repetition window
        self.window = mx.full((batch, REPETITION_WINDOW, synth.n_vq), V, mx.int32)  # V: no code
        self.pos = mx.zeros((batch,), mx.int32)  # column of the next input
        self.frames = mx.zeros((batch,), mx.int32)
        self.length = mx.zeros((batch,), mx.int32)  # frames kept
        self.cap = mx.zeros((batch,), mx.int32)
        self.done = mx.ones((batch,), mx.bool_)
        self.t = 0
        self.jobs: list[_Job | None] = [None] * batch
        self.taken: list[list[np.ndarray]] = [[] for _ in range(batch)]
        self.bound = np.zeros(batch, np.int64)  # host upper bound on each running slot's pos
        self.queue: deque[_Job] = deque()

    @property
    def busy(self) -> bool:
        return bool(self.queue) or any(j is not None for j in self.jobs)

    def submit(self, key, prompt: np.ndarray, anchor: mx.array, phonemes: str) -> None:
        cap = min(MAX_NEW_FRAMES, max_expected_frames(phonemes))
        if len(prompt) + cap > self.width:
            raise ValueError(f"{key}: a {len(prompt)}-row prompt and {cap} frames exceed {self.width} positions")
        self.queue.append(_Job(key, prompt, anchor, cap, min_expected_frames(phonemes)))

    def run(self) -> list[tuple[object, np.ndarray]]:
        """Fill free slots, generate CHECK_EVERY frames, and return the chunks that finished as
        (key, codes (frames, n_vq))."""
        self._fill()
        if not any(j is not None for j in self.jobs):
            return []
        steps = [self._step() for _ in range(CHECK_EVERY)]
        codes = np.array(mx.stack(steps, axis=1))
        done, length = np.array(self.done), np.array(self.length)
        finished = []
        for b, job in enumerate(self.jobs):
            if job is None:
                continue
            self.taken[b].append(codes[b])
            if not done[b]:
                continue
            take = np.concatenate(self.taken[b])[: length[b]]
            self.jobs[b], self.taken[b], self.bound[b] = None, [], 0
            if job.best is None or len(take) > len(job.best):
                job.best = take
            if len(job.best) < job.floor and job.tries < RETRIES:
                job.tries += 1
                self.queue.appendleft(job)
            else:
                finished.append((job.key, job.best))
        return finished

    def _fill(self) -> None:
        free = [b for b, j in enumerate(self.jobs) if j is None][: len(self.queue)]
        if not free:
            return
        jobs = [self.queue.popleft() for _ in free]
        lengths = np.array([len(j.prompt) for j in jobs])
        rows = np.zeros((len(jobs), lengths.max(), self.s.n_vq + 1), np.int32)
        rows[..., 1:] = self.s.audio_pad
        for g, j in enumerate(jobs):
            rows[g, : len(j.prompt)] = j.prompt
        anchors = mx.stack([j.anchor for j in jobs])
        slots = mx.array(free)
        x = self.s.embed_rows(mx.array(rows), anchors)
        h = self.s.model.semantic_backbone.prefill(x, self.cache, slots)
        self.hidden[slots] = h[mx.arange(len(jobs)), mx.array(lengths - 1)]
        self.step_input[slots] = self.s.sgs_emb + anchors
        self.counts[slots] = 0
        self.window[slots] = self.vocab.size
        self.pos[slots] = mx.array(lengths, mx.int32)
        self.frames[slots] = 0
        self.length[slots] = 0
        self.cap[slots] = mx.array([j.cap for j in jobs], mx.int32)
        self.done[slots] = False
        for b, j in zip(free, jobs):
            self.jobs[b] = j
        self.bound[free] = lengths

    def _step(self) -> mx.array:
        s = self.s
        codes, eos = s.frame(self.hidden, self.counts > 0)
        self.length = mx.where(self.done, self.length, self.frames + 1)
        self.done = self.done | eos | (self.frames + 1 >= self.cap)
        self.frames = self.frames + 1
        slot = self.t % REPETITION_WINDOW
        self.t += 1
        old = self.window[:, slot]
        self.window[:, slot] = codes
        self.counts = (
            self.counts
            + (codes[..., None] == self.vocab).astype(mx.int32)
            - (old[..., None] == self.vocab).astype(mx.int32)
        )
        x = (s.audio_flat[codes + s.code_offsets].sum(axis=1) + self.step_input)[:, None, :]
        width = min(self.width, int(self.bound.max()) + 1)
        self.hidden = s.model.semantic_backbone.step(x, self.cache, self.pos, width)[:, 0]
        self.pos = mx.where(self.done, self.pos, self.pos + 1)
        self.bound[self.bound > 0] += 1
        mx.async_eval(self.hidden, self.counts, self.window, self.pos, self.frames, self.done, self.length, codes)
        return codes
