"""MPS/CUDA throughput patch for vieneu v3-turbo batched inference.

Replaces V3TurboBatchEngine.generate_batch on the live engine with a version that
avoids per-frame GPU->CPU syncs (vectorized repetition penalty, device-side EOS
tracking) and decodes all rows through the MOSS codec in one batched call.
On CUDA the whole per-frame acoustic step (16 sequential decoder steps + sampling
+ EOS head) replays as ONE CUDA graph — unlike stock vieneu's graph path, the
repetition penalty survives capture because its history is a static device tensor.
Also provides fp16 conversion and a per-text-voice batch entrypoint.
"""
from __future__ import annotations

import copy
import math
import types
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Codec placement, all measured (2026-07-30): stays on the GPU, synchronous.
# CPU offload (subprocess, 6-10 threads) reaches only ~40x RT on real
# length-sorted batches (padding + O(T^2) attention; ~28x under the EN workers'
# CPU load) — below the LM pace it must hide behind, a net loss. In-process
# async GPU decode crashes: torch MPS forbids concurrent command encoding from
# two threads (MTLCommandBuffer assertion).

from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence


@torch.no_grad()
def _sample_fast(logits, temperature, top_k, top_p, repetition_penalty, seen_ch):
    """Same sampling as batched_acoustic._sample_batched, but the repetition
    penalty is applied via a (B, V) bool mask instead of per-row .item() loops.
    CTRL/MOSS rule: logit<0 -> *penalty, else /penalty, applied BEFORE temperature."""
    if seen_ch is not None:
        penalized = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        logits = torch.where(seen_ch, penalized, logits)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_k and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(s_logits, dim=-1)
        drop = probs.cumsum(dim=-1) > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        s_logits = s_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, s_idx, s_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def _generate_frame_fast(model, backbone_hidden, *, temperature, top_k, top_p,
                         repetition_penalty, seen):
    """batched_acoustic.generate_frame_batched with the seen-code history kept as a
    persistent (B, n_vq, V) bool tensor on device (updated with scatter_)."""
    cfg = model.config
    n_vq, H = cfg.n_vq, cfg.hidden_size
    dec = model.acoustic_decoder
    L = len(dec.layers)
    dt = next(dec.parameters()).dtype
    dev = backbone_hidden.device
    B = backbone_hidden.shape[0]

    def _sample_ch(ch, vec):
        logits = model.audio_lm_heads[ch](vec).float()
        seen_ch = seen[:, ch] if seen is not None else None
        code = _sample_fast(logits, temperature, top_k, top_p, repetition_penalty, seen_ch)
        if seen is not None:
            seen[:, ch].scatter_(1, code.unsqueeze(1), True)
        return code

    cond = backbone_hidden.to(dt)
    sgs_ids = torch.full((B,), cfg.speech_generation_start_token_id, device=dev, dtype=torch.long)
    txt = model.text_embeddings(sgs_ids).to(dt)
    tok = torch.stack([cond, txt], dim=1)
    pos = torch.arange(2, device=dev, dtype=torch.long)
    hidden, pk, pv = dec.cached_step(tok, pos, [None] * L, [None] * L)
    prefill_out = hidden
    codes: List[torch.Tensor] = [_sample_ch(0, hidden[:, 1])]
    for ch in range(1, n_vq):
        emb = model.audio_embeddings[ch - 1](codes[-1]).to(dt)
        pos = torch.arange(ch + 1, ch + 2, device=dev, dtype=torch.long)
        hidden, pk, pv = dec.cached_step(emb.view(B, 1, H), pos, pk, pv)
        codes.append(_sample_ch(ch, hidden[:, 0]))
    return torch.stack(codes, dim=1), prefill_out


# A/B kill-switch for the static-KV MPS decode loop below.
MPS_FAST = True


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class _MPSFastDecoder:
    """Static-KV per-frame decode loop for MPS (the MPS analog of the CUDA graph).

    Replaces the HF Qwen3 per-step forward (DynamicCache torch.cat re-copies the
    whole KV every frame and each call pays HF mask/rope/dispatch overhead) with a
    hand-rolled step over preallocated KV buffers written in place. Attention
    kv-length is bucketed to KV_BUCKET so MPS sees a handful of kernel shapes
    instead of one per frame, and KV allocations are bucketed to T_BUCKET so the
    caching allocator reuses a few size classes across batches. Also owns the
    lean acoustic frame (precomputed slot constants, static 17-slot KV) and
    sort-free sampling (top-k -> nucleus within candidates -> exponential-race
    argmax; same distribution as the stock sampler, no full-vocab sort and no
    multinomial)."""

    KV_BUCKET = 64
    T_BUCKET = 128

    def __init__(self, engine):
        model = engine.model
        cfg = model.config
        bb = model.semantic_backbone
        self.model = model
        self.cfg = cfg
        self.dev = engine.tts.device
        self.dt = next(bb.parameters()).dtype
        self.layers = list(bb.layers)
        self.final_norm = bb.norm
        qcfg = bb.config
        self.n_heads = qcfg.num_attention_heads
        self.n_kv = qcfg.num_key_value_heads
        self.hd = qcfg.head_dim
        self.scale = self.hd ** -0.5

        pos = torch.arange(qcfg.max_position_embeddings, device=self.dev).unsqueeze(0)
        cos, sin = bb.rotary_emb(torch.zeros(1, dtype=self.dt, device=self.dev), pos)
        self.rope_cos = cos[0].to(self.dt)                      # (T_max, hd)
        self.rope_sin = sin[0].to(self.dt)

        emb_w = torch.stack([e.weight for e in model.audio_embeddings])
        self.audio_emb_flat = emb_w.reshape(-1, cfg.hidden_size).to(self.dt)
        self.ch_offsets = (
            torch.arange(cfg.n_vq, device=self.dev) * cfg.audio_vocab_size
        ).view(1, -1)
        sgs = torch.tensor([cfg.speech_generation_start_token_id], device=self.dev)
        self.sgs_text_emb = model.text_embeddings(sgs)[0].to(self.dt)

        dec = model.acoustic_decoder
        self.dec_layers = list(dec.layers)
        self.dec_norm = dec.norm
        self.dec_pos = dec.slot_pos_emb.weight[: cfg.n_vq + 1].to(self.dt)  # (n_vq+1, H)
        a0 = dec.layers[0].attn
        self.a_heads, self.a_hd = a0.num_heads, a0.head_dim

    # ── backbone ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def prefill(self, bb_wrap, embeds_list, batch_spk, max_new_frames):
        h, cache, mask, cur_pos = bb_wrap.prefill(embeds_list)
        B, P = mask.shape
        L = len(self.layers)
        T = -(-(P + max_new_frames + 2) // self.T_BUCKET) * self.T_BUCKET
        self.kc = torch.zeros(L, B, self.n_kv, T, self.hd, dtype=self.dt, device=self.dev)
        self.vc = torch.zeros_like(self.kc)
        for i in range(L):
            self.kc[i, :, :, :P] = cache.layers[i].keys
            self.vc[i, :, :, :P] = cache.layers[i].values
        neg = torch.finfo(self.dt).min
        amask = torch.full((B, 1, 1, T), neg, dtype=self.dt, device=self.dev)
        amask[:, 0, 0, :P] = torch.where(
            mask.bool(), torch.zeros((), dtype=self.dt, device=self.dev),
            torch.full((), neg, dtype=self.dt, device=self.dev),
        )
        self.amask = amask
        self.col = P
        self.positions = cur_pos.clone()
        slot_const = self.sgs_text_emb
        if self.model.xvec_proj is not None and batch_spk is not None:
            slot_const = slot_const + self.model.xvec_proj(batch_spk.to(self.dt))
        self.slot_const = slot_const                            # (H,) or (B, H)
        return h

    @torch.no_grad()
    def slot_embed(self, codes):
        """Decode-slot embedding: sgs text emb + sum of audio embs + speaker anchor.
        Generated codes are always < audio_vocab_size, so no pad masking needed."""
        B = codes.shape[0]
        flat = (codes + self.ch_offsets).reshape(-1)
        emb = F.embedding(flat, self.audio_emb_flat).view(B, self.cfg.n_vq, -1).sum(1)
        return emb + self.slot_const

    @torch.no_grad()
    def bb_step(self, x):
        """One backbone decode step. x is (B, H); returns final-norm hidden (B, H)."""
        B = x.shape[0]
        col = self.col
        self.positions += 1
        self.amask[:, 0, 0, col] = 0.0
        S = min(-(-(col + 1) // self.KV_BUCKET) * self.KV_BUCKET, self.kc.shape[3])
        cos = self.rope_cos[self.positions].view(B, 1, self.hd)
        sin = self.rope_sin[self.positions].view(B, 1, self.hd)
        amask = self.amask[..., :S]

        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(B, self.n_heads, self.hd))
            k = a.k_norm(a.k_proj(h).view(B, self.n_kv, self.hd))
            v = a.v_proj(h).view(B, self.n_kv, self.hd)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            self.kc[i, :, :, col] = k
            self.vc[i, :, :, col] = v
            o = F.scaled_dot_product_attention(
                q.unsqueeze(2), self.kc[i, :, :, :S], self.vc[i, :, :, :S],
                attn_mask=amask, enable_gqa=True,
            ).reshape(B, -1)
            x = x + a.o_proj(o)
            h2 = layer.post_attention_layernorm(x)
            mlp = layer.mlp
            x = x + mlp.down_proj(F.silu(mlp.gate_proj(h2)) * mlp.up_proj(h2))
        self.col = col + 1
        return self.final_norm(x)

    # ── acoustic frame ──────────────────────────────────────────────────────

    def _dec_step(self, x, pk, pv):
        """One acoustic-decoder step; x is (B, S, H) with pos emb already added.
        Functional KV (torch.cat is cheap at this size and keeps SDPA fused).
        S=2 is the aligned prefill (is_causal), S=1 steps attend everything."""
        B, Sq, _ = x.shape
        nk, nv = [], []
        for i, layer in enumerate(self.dec_layers):
            a = layer.attn
            h = layer.norm1(x)
            qkv = a.qkv(h).view(B, Sq, 3, self.a_heads, self.a_hd)
            q, k, v = qkv.unbind(dim=2)
            q = a.q_norm(q.transpose(1, 2))
            k = a.k_norm(k.transpose(1, 2))
            v = v.transpose(1, 2)
            if pk[i] is not None:
                k = torch.cat([pk[i], k], dim=2)
                v = torch.cat([pv[i], v], dim=2)
            nk.append(k)
            nv.append(v)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=Sq > 1)
            x = x + a.o_proj(o.transpose(1, 2).reshape(B, Sq, -1))
            n2 = layer.norm2(x)
            x = x + layer.ff_down(F.silu(layer.ff_gate(n2)) * layer.ff_up(n2))
        return self.dec_norm(x), nk, nv

    def _sample(self, logits, temperature, top_k, top_p, repetition_penalty, seen_ch):
        """Distribution-identical to _sample_fast: top-k first, nucleus within the
        candidates, then exponential-race argmax instead of multinomial."""
        if seen_ch is not None:
            pen = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
            logits = torch.where(seen_ch, pen, logits)
        if temperature <= 0:
            return logits.argmax(dim=-1)
        k = min(top_k, logits.shape[-1]) if top_k and top_k > 0 else logits.shape[-1]
        cand, idx = torch.topk(logits, k, dim=-1)
        probs = (cand / max(temperature, 1e-6)).softmax(-1)
        if 0.0 < top_p < 1.0:
            probs = probs * (probs.cumsum(-1) - probs < top_p)
        race = probs / torch.empty_like(probs).exponential_()
        return idx.gather(-1, race.argmax(-1, keepdim=True)).squeeze(-1)

    @torch.no_grad()
    def frame(self, h, *, temperature, top_k, top_p, repetition_penalty, seen):
        """Sample one frame's n_vq codes + the EOS flag from backbone hidden (B, H)."""
        cfg = self.cfg
        B = h.shape[0]
        L = len(self.dec_layers)

        def sample_ch(ch, vec):
            logits = self.model.audio_lm_heads[ch](vec).float()
            code = self._sample(logits, temperature, top_k, top_p, repetition_penalty,
                                seen[:, ch] if seen is not None else None)
            if seen is not None:
                seen[:, ch].scatter_(1, code.unsqueeze(1), True)
            return code

        tok = torch.stack([h.to(self.dt), self.sgs_text_emb.expand(B, -1)], dim=1)
        tok = tok + self.dec_pos[:2]
        hidden, pk, pv = self._dec_step(tok, [None] * L, [None] * L)
        is_eos = (
            self.model.text_lm_head(hidden[:, 0]).float().argmax(-1)
            == cfg.speech_generation_end_token_id
        )
        codes = [sample_ch(0, hidden[:, 1])]
        for ch in range(1, cfg.n_vq):
            emb = F.embedding(codes[-1], self.model.audio_embeddings[ch - 1].weight).to(self.dt)
            x = (emb + self.dec_pos[ch + 1]).view(B, 1, -1)
            hidden, pk, pv = self._dec_step(x, pk, pv)
            codes.append(sample_ch(ch, hidden[:, 0]))
        return torch.stack(codes, dim=1), is_eos


def _get_mps_fast(engine) -> _MPSFastDecoder:
    fast = getattr(engine, "_mps_fast", None)
    if fast is None:
        fast = engine._mps_fast = _MPSFastDecoder(engine)
    return fast


class _GraphedFrameFast:
    """CUDA-graph capture of _generate_frame_fast (+ EOS head) for a fixed batch.

    The `seen` repetition-penalty history is a static buffer read AND scatter_-
    updated inside the graph, so it persists across replays within a batch;
    zero it before each new batch."""

    def __init__(self, model, batch_size: int, *, temperature: float, top_k: int,
                 top_p: float, repetition_penalty: float, warmup: int = 3):
        cfg = model.config
        dev = next(model.parameters()).device
        dt = next(model.acoustic_decoder.parameters()).dtype
        self.model = model
        self.eos_id = int(cfg.speech_generation_end_token_id)
        self._sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                              repetition_penalty=repetition_penalty)
        self.static_hidden = torch.zeros(batch_size, cfg.hidden_size, device=dev, dtype=dt)
        self.seen = (
            torch.zeros(batch_size, cfg.n_vq, cfg.audio_vocab_size, dtype=torch.bool, device=dev)
            if not math.isclose(repetition_penalty, 1.0) else None
        )

        # Warm up on a side stream, then capture (capture records without executing,
        # so `seen` stays clean after the post-warmup zero).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                self._run_once()
        torch.cuda.current_stream().wait_stream(s)
        if self.seen is not None:
            self.seen.zero_()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_codes, self.static_eos = self._run_once()

    def _run_once(self):
        codes, prefill_out = _generate_frame_fast(
            self.model, self.static_hidden, seen=self.seen, **self._sampling
        )
        is_eos = self.model.text_lm_head(prefill_out[:, 0]).float().argmax(-1) == self.eos_id
        return codes, is_eos

    @torch.no_grad()
    def run(self, backbone_hidden: torch.Tensor):
        self.static_hidden.copy_(backbone_hidden)
        self.graph.replay()
        return self.static_codes.clone(), self.static_eos.clone()


def _get_fast_graph(engine, B, temperature, top_k, top_p, repetition_penalty):
    graphs = getattr(engine, "_fast_graphs", None)
    if graphs is None:
        graphs = engine._fast_graphs = {}
    key = (B, round(temperature, 4), top_k, round(top_p, 4), round(repetition_penalty, 4))
    if key not in graphs:
        graphs[key] = _GraphedFrameFast(
            engine.model, B, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    return graphs[key]


@torch.no_grad()
def _generate_batch_fast(
    self,
    requests: List[dict],
    *,
    temperature: float = 0.8,
    top_k: int = 25,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_new_frames: int = 300,
    use_cudagraph: bool = True,
) -> List[np.ndarray]:
    """Drop-in replacement for V3TurboBatchEngine.generate_batch.

    Semantics match the stock loop (each row keeps frames up to and including its
    EOS frame), but finished/lengths live on device — finished.all() is synced
    only every 8 frames — and the codec decode runs once, batched and right-padded
    (the codec is causal, so valid samples match per-row decode; trimmed by
    audio_lengths)."""
    cfg = self.config
    n_vq = cfg.n_vq
    eos_id = cfg.speech_generation_end_token_id
    sgs = cfg.speech_generation_start_token_id
    pad = cfg.audio_pad_token_id
    dev = self.tts.device

    embeds_list = [self._prompt_embeds(r) for r in requests]
    B = len(requests)

    spk_list = [self.tts._resolve_speaker_emb(r.get("speaker_emb")) for r in requests]
    batch_spk = torch.cat(spk_list, dim=0) if (spk_list and spk_list[0] is not None) else None

    use_rep = not math.isclose(repetition_penalty, 1.0)
    seen = (torch.zeros(B, n_vq, cfg.audio_vocab_size, dtype=torch.bool, device=dev)
            if use_rep else None)
    graphed = fast = None
    if use_cudagraph and dev.type == "cuda":
        graphed = _get_fast_graph(self, B, temperature, top_k, top_p, repetition_penalty)
        seen = graphed.seen
        if seen is not None:
            seen.zero_()
    elif MPS_FAST and dev.type == "mps":
        fast = _get_mps_fast(self)

    if fast is not None:
        h = fast.prefill(self.bb, embeds_list, batch_spk, max_new_frames)
    else:
        h, cache, mask, pos = self.bb.prefill(embeds_list)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    lengths = torch.zeros(B, dtype=torch.long, device=dev)
    frames: List[torch.Tensor] = []

    for step in range(max_new_frames):
        if graphed is not None:
            codes, is_eos = graphed.run(h)
        elif fast is not None:
            codes, is_eos = fast.frame(
                h, temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, seen=seen,
            )
        else:
            codes, prefill_out = _generate_frame_fast(
                self.model, h, temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, seen=seen,
            )
            is_eos = self.model.text_lm_head(prefill_out[:, 0]).float().argmax(-1) == eos_id
        frames.append(codes)
        lengths += (~finished).long()   # active rows record this frame (incl. their EOS frame)
        finished |= is_eos
        if (step + 1) % 8 == 0 and bool(finished.all()):
            break

        if fast is not None:
            h = fast.bb_step(fast.slot_embed(codes))
            continue
        slot = torch.full((B, 1, n_vq + 1), pad, dtype=torch.long, device=dev)
        slot[:, :, 0] = sgs
        slot[:, 0, 1:] = codes
        se = self.model._build_inputs_embeds(slot, speaker_emb=batch_spk)
        h, cache, mask, pos = self.bb.decode_step(se, cache, mask, pos)

    if not frames:
        return [np.zeros(0, dtype=np.float32) for _ in range(B)]

    codes_all = torch.stack(frames, dim=1)                      # (B, T, n_vq)
    T = codes_all.shape[1]
    # Give every row the SAME decode length by repeating its last valid frame:
    # shorter rows in a mixed batch otherwise hit fully-masked fp16 attention
    # rows inside the codec, whose NaNs poison the entire row (the historical
    # ~0.45% "all-silent" wavs). The codec is causal, so the first lens[b]
    # frames decode identically; the repeated tail is trimmed below.
    gather_idx = torch.minimum(
        torch.arange(T, device=dev).unsqueeze(0), lengths.unsqueeze(1) - 1
    )
    codes_uniform = codes_all.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, n_vq))
    codes_list = [codes_uniform[b].transpose(0, 1) for b in range(B)]  # (n_vq, T)
    out = self.tts.audio_tokenizer.batch_decode(codes_list)
    audio = out.audio.float().mean(1).cpu().numpy()             # (B, S_max), channels averaged
    spf = int(out.audio_lengths[0].item()) // T                 # samples per frame
    lens = lengths.tolist()
    results = [audio[b, : lens[b] * spf].copy() for b in range(B)]

    corrupt = [
        b for b, r in enumerate(results)
        if r.size and (not np.isfinite(r).all() or float(np.abs(r).max()) == 0.0)
    ]
    if corrupt:
        # Metal silently emits all-zero rows for ~0.4% of MPS decodes (biased to
        # long-T batches; multi-GB mask/workspace pressure) — the same failure
        # class as the kokoro Core ML buffer bug, and the residual cause of the
        # historical "silent wav"s after the NaN fix. Re-decode those rows on a
        # CPU clone of the codec; CPU execution is immune.
        tok_cpu = getattr(self, "_codec_cpu", None)
        if tok_cpu is None:
            tok_cpu = copy.deepcopy(self.tts.audio_tokenizer).to("cpu").eval()
            self._codec_cpu = tok_cpu
        cpu_codes = codes_uniform[corrupt].cpu()
        out_cpu = tok_cpu.batch_decode(
            [cpu_codes[i].transpose(0, 1) for i in range(len(corrupt))]
        )
        audio_cpu = out_cpu.audio.float().mean(1).numpy()
        for i, b in enumerate(corrupt):
            r = audio_cpu[i, : lens[b] * spf].copy()
            if not r.size or not np.isfinite(r).all() or float(np.abs(r).max()) == 0.0:
                raise RuntimeError("codec produced silent/NaN row on both GPU and CPU")
            results[b] = r

    if dev.type == "mps":
        # The KV cache reallocates a new block every decode step; the MPS caching
        # allocator hoards every freed size class, ballooning to swap over a long
        # run. Release after each batch (costs ~ms, saves tens of GB).
        torch.mps.empty_cache()
    return results


def _patch_shared_codec_mask(tok) -> None:
    """All our codec decodes are uniform-length (silent-wav fix), so the per-row
    (B,1,S,S) bool attention masks collapse to one broadcast (1,1,S,S) row —
    the stock builder allocates ~330 MB of masks per batch at T=100."""
    for module in tok.modules():
        cls = type(module)
        if cls.__name__ != "MossAudioTokenizerMultiheadAttention":
            continue
        if getattr(cls, "_shared_mask_patched", False):
            return
        orig = cls._build_non_streaming_sdpa_bias

        def build(self, input_lengths, T, device, _orig=orig):
            return _orig(self, input_lengths[:1], T, device)

        cls._build_non_streaming_sdpa_bias = build
        cls._shared_mask_patched = True
        return


def apply(tts) -> None:
    """Patch the tts instance's batch engine with the MPS-optimized generate_batch."""
    engine = tts._get_batch_engine()
    if engine is None:
        raise RuntimeError("vieneu batch engine unavailable (PyTorch backend required).")
    engine.generate_batch = types.MethodType(_generate_batch_fast, engine)
    _patch_shared_codec_mask(engine.tts.audio_tokenizer)


def apply_fp16(tts) -> None:
    """Convert the backbone + acoustic decoder + lm heads to fp16 and run the MOSS
    codec in bf16 autocast (its autocast helper is CUDA-only upstream). The codec
    uses bf16, not fp16: fp16's narrow exponent occasionally overflows to NaN in
    the codec convs, which writes out as an all-silent wav."""
    tts.engine.model.half()

    tok = tts.engine.audio_tokenizer
    tok.set_compute_dtype("bf16")

    @contextmanager
    def _codec_autocast():
        device = next(tok.parameters()).device
        if device.type in ("cuda", "mps") and tok.compute_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=tok.compute_dtype):
                yield
        else:
            yield

    tok._codec_inference_autocast = _codec_autocast


def infer_batch_voices(
    tts,
    texts: List[str],
    voices: List[str],
    max_new_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """Vieneu.infer_batch with a per-text voice list (preset/registered voices only).

    Chunks, phonemizes and joins exactly like infer_batch; no watermarking (the
    installed perth watermarker is a no-op anyway)."""
    if not texts:
        return []
    if len(voices) != len(texts):
        raise ValueError("voices must have one entry per text")

    engine = tts._get_batch_engine()
    if engine is None:
        raise RuntimeError("vieneu batch engine unavailable (PyTorch backend required).")

    sampling = dict(
        temperature=0.8, top_k=25, top_p=0.95,
        max_new_frames=300 if max_new_frames is None else int(max_new_frames),
        repetition_penalty=1.2,
    )
    refs = {v: tts._resolve_ref(v, None, True, True) for v in set(voices)}

    per_text_gaps: List[list] = []
    flat: List[tuple] = []                                     # (chunk, text_index)
    for ti, text in enumerate(texts):
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=256)
        per_text_gaps.append(gaps)
        for c in chunks:
            flat.append((c, ti))

    empty = np.array([], dtype=np.float32)
    if not flat:
        return [empty for _ in texts]

    requests = []
    for chunk, ti in flat:
        speaker_emb, ref_codes = refs[voices[ti]]
        requests.append({
            "phonemes": phonemize_text_with_emotions(chunk),
            "speaker_emb": speaker_emb, "ref_codes": ref_codes,
            "style": "tu_nhien", "use_ref_codes": True,
        })

    # Batch chunks by phoneme length, not text order: each batch runs to its
    # longest member, so mixing lengths burns tail compute on finished rows.
    order = sorted(range(len(requests)), key=lambda i: len(requests[i]["phonemes"]), reverse=True)
    wavs: List[np.ndarray] = [empty] * len(requests)
    bs = tts.max_batch_size
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        for j, wav in zip(idx, engine.generate_batch([requests[k] for k in idx], **sampling)):
            wavs[j] = wav

    grouped: List[List[np.ndarray]] = [[] for _ in texts]
    for wav, (_, ti) in zip(wavs, flat):
        grouped[ti].append(wav)

    results: List[np.ndarray] = []
    for ti in range(len(texts)):
        if not grouped[ti]:
            results.append(empty)
            continue
        results.append(join_audio_chunks(
            grouped[ti], tts.sample_rate, silence_ps=gaps_to_silence(per_text_gaps[ti])
        ))
    return results
