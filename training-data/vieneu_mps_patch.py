"""MPS throughput patch for vieneu v3-turbo batched inference.

Replaces V3TurboBatchEngine.generate_batch on the live engine with a version that
avoids per-frame GPU->CPU syncs (vectorized repetition penalty, device-side EOS
tracking) and decodes all rows through the MOSS codec in one batched call.
Also provides fp16 conversion and a per-text-voice batch entrypoint.
"""
from __future__ import annotations

import math
import types
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
    use_cudagraph: bool = False,
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

    h, cache, mask, pos = self.bb.prefill(embeds_list)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    lengths = torch.zeros(B, dtype=torch.long, device=dev)
    frames: List[torch.Tensor] = []

    for step in range(max_new_frames):
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

        slot = torch.full((B, 1, n_vq + 1), pad, dtype=torch.long, device=dev)
        slot[:, :, 0] = sgs
        slot[:, 0, 1:] = codes
        se = self.model._build_inputs_embeds(slot, speaker_emb=batch_spk)
        h, cache, mask, pos = self.bb.decode_step(se, cache, mask, pos)

    if not frames:
        return [np.zeros(0, dtype=np.float32) for _ in range(B)]

    codes_all = torch.stack(frames, dim=1)                      # (B, T, n_vq)
    lens = lengths.tolist()
    codes_list = [codes_all[b, : lens[b]].transpose(0, 1) for b in range(B)]  # (n_vq, T_b)
    out = self.tts.audio_tokenizer.batch_decode(codes_list)
    audio = out.audio.float().mean(1).cpu().numpy()             # (B, S_max), channels averaged
    audio_lens = out.audio_lengths.tolist()
    return [audio[b, : audio_lens[b]].copy() for b in range(B)]


def apply(tts) -> None:
    """Patch the tts instance's batch engine with the MPS-optimized generate_batch."""
    engine = tts._get_batch_engine()
    if engine is None:
        raise RuntimeError("vieneu batch engine unavailable (PyTorch backend required).")
    engine.generate_batch = types.MethodType(_generate_batch_fast, engine)


def apply_fp16(tts) -> None:
    """Convert the backbone + acoustic decoder + lm heads to fp16 and run the MOSS
    codec in fp16 autocast (its autocast helper is CUDA-only upstream)."""
    tts.engine.model.half()

    tok = tts.engine.audio_tokenizer
    tok.set_compute_dtype("fp16")

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

    wavs: List[np.ndarray] = []
    bs = tts.max_batch_size
    for i in range(0, len(requests), bs):
        wavs.extend(engine.generate_batch(requests[i : i + bs], **sampling))

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
