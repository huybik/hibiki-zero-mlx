"""VieNeu-TTS v3 Turbo's generation with continuous batching, for voice.py
([tranche-tts-scaleup](../../docs/agent-memory/tranche-tts-scaleup.md#optimizing-vieneu-and-voxcpm2)).

The SDK's fused frame (vieneu 3.7.1, `v3_turbo_serve.fused`: the acoustic frame, sampling with the repetition
penalty, EOS and caps, the backbone step, one CUDA graph), with each slot's own cache position, frame counter and
repetition window, so a finished slot takes the next chunk while the others go on; the SDK's static batches run
until their longest row ends. Prompts are prefilled by the SDK's BatchedBackbone (HF Qwen3, left-padded) and their
keys copied into the slot from position 0, as the prefill numbered them.

Before capture, each backbone layer's projections, norms, rotary and MLP, and the acoustic frame with its
sampling, go through Inductor; the cache writes stay eager (compiled, the in-place writes copy the whole cache), and
the backbone's attention reads each slot's live keys only (decode_attention.py; the dense masked form read the whole
1,024-position cache of every slot each frame). The sampler draws from the top-k values: the SDK's distribution
without its sort over the vocabulary.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from vieneu.v3_turbo_serve.batched_backbone import BatchedBackbone

from training.tranche.decode_attention import attend


def sample_topk_first(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """fused.sample_gpu's draw from the top-k values alone: top-p over k values and the draw among them, where the
    SDK sorts, softmaxes and scatters the whole vocabulary (probabilities outside the top k are zero either way)."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    vals, idx = torch.topk(logits / max(temperature, 1e-6), top_k, dim=-1)  # largest first
    if 0.0 < top_p < 1.0:
        drop = F.softmax(vals, dim=-1).cumsum(dim=-1) > top_p
        drop = torch.cat([torch.zeros_like(drop[..., :1]), drop[..., :-1]], dim=-1)
        vals = vals.masked_fill(drop, float("-inf"))
    pick = torch.multinomial(F.softmax(vals, dim=-1), num_samples=1)
    return idx.gather(-1, pick).squeeze(-1)


def acoustic_frame(model, hidden: torch.Tensor, rep: "SlotRepetition", *, temperature: float, top_k: int,
                   top_p: float, repetition_penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fused.acoustic_frame_gpu with sample_topk_first, the local decoder's attention (one query over at most 17
    keys) as plain math: (codes [B, n_vq], is_eos [B])."""
    cfg = model.config
    dec = model.acoustic_decoder
    dt = next(dec.parameters()).dtype
    B, H = hidden.shape
    use_rep = not math.isclose(repetition_penalty, 1.0)

    def sample(ch: int, vec: torch.Tensor) -> torch.Tensor:
        logits = model.audio_lm_heads[ch](vec).float()
        if use_rep:
            logits = rep.penalise(logits, ch, repetition_penalty)
        code = sample_topk_first(logits, temperature, top_k, top_p)
        if use_rep:
            rep.add(ch, code)
        return code

    with sdpa_kernel(SDPBackend.MATH):
        start = torch.full((B,), cfg.speech_generation_start_token_id, device=hidden.device, dtype=torch.long)
        tok = torch.stack([hidden.to(dt), model.text_embeddings(start).to(dt)], dim=1)
        pos = torch.arange(2, device=hidden.device, dtype=torch.long)
        out, pk, pv = dec.cached_step(tok, pos, [None] * len(dec.layers), [None] * len(dec.layers))
        is_eos = model.text_lm_head(out[:, 0]).float().argmax(-1) == cfg.speech_generation_end_token_id
        codes = [sample(0, out[:, 1])]
        for ch in range(1, cfg.n_vq):
            emb = model.audio_embeddings[ch - 1](codes[-1]).to(dt)
            pos = torch.arange(ch + 1, ch + 2, device=hidden.device, dtype=torch.long)
            out, pk, pv = dec.cached_step(emb.view(B, 1, H), pos, pk, pv)
            codes.append(sample(ch, out[:, 0]))
    if use_rep:
        rep.frame += 1
    return torch.stack(codes, dim=1), is_eos


class SlotRepetition:
    """fused.GpuRepHistory (the codes of each slot's last `window` frames, per codebook) with a frame counter per
    slot, so slots start at different times."""

    def __init__(self, B: int, n_vq: int, vocab: int, window: int, device):
        self.window = int(window)
        self.counts = torch.zeros(B, n_vq, vocab, dtype=torch.int32, device=device)
        self.ring = torch.zeros(B, n_vq, max(self.window, 1), dtype=torch.long, device=device)
        self.frame = torch.zeros(B, dtype=torch.long, device=device)
        self._ones = torch.ones(B, 1, dtype=torch.int32, device=device)

    def reset(self, slots: torch.Tensor) -> None:
        self.counts[slots] = 0
        self.ring[slots] = 0
        self.frame[slots] = 0

    def penalise(self, logits: torch.Tensor, ch: int, penalty: float) -> torch.Tensor:
        seen = self.counts[:, ch] > 0
        return torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)

    def add(self, ch: int, code: torch.Tensor) -> None:
        counts = self.counts[:, ch]
        if self.window > 0:  # the code `window` frames back leaves the window, once the ring has wrapped
            slot = torch.remainder(self.frame, self.window).view(-1, 1)
            old = self.ring[:, ch].gather(1, slot)
            evict = (self.frame >= self.window).to(torch.int32).view(-1, 1)
            counts.scatter_add_(1, old, -(self._ones * evict))
            self.ring[:, ch].scatter_(1, slot, code.view(-1, 1))
        counts.scatter_add_(1, code.view(-1, 1), self._ones)


class SlotBackbone:
    """fused.StaticBackbone (the Qwen3 decode step over a preallocated cache) with a write position per slot."""

    def __init__(self, model, B: int, max_len: int):
        bb = model.semantic_backbone
        cfg = bb.config
        p = next(bb.parameters())
        self.layers, self.norm, self.rotary = bb.layers, bb.norm, bb.rotary_emb
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.B, self.max_len = B, int(max_len)
        self.k = torch.zeros(len(self.layers), B, self.n_kv, self.max_len, self.hd, device=p.device, dtype=p.dtype)
        self.v = torch.zeros_like(self.k)
        self.cur = torch.zeros(B, device=p.device, dtype=torch.long)  # each slot's next write position
        self.pos = torch.zeros(B, 1, device=p.device, dtype=torch.long)
        self.ar = torch.arange(B, device=p.device)
        self.pre = torch.compile(self._pre, dynamic=False)
        self.post = torch.compile(self._post, dynamic=False)

    def _pre(self, layer, x, cos, sin):
        B, a = x.shape[0], layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(h).view(B, 1, self.n_heads, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(B, 1, self.n_kv, self.hd)).transpose(1, 2)
        v = a.v_proj(h).view(B, 1, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def _post(self, layer, x, attn):
        x = x + layer.self_attn.o_proj(attn.reshape(x.shape[0], 1, -1))
        return x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.no_grad()
    def load(self, slots: list[int], cache, attn_mask: torch.Tensor, cur_pos: torch.Tensor) -> None:
        """Rows of a BatchedBackbone prefill (left-padded, `attn_mask` [n, T]) into `slots`, from position 0: every
        layer's keys shifted left by its row's padding in one gather (the positions after a row's prompt get copies of
        its last key, never read before the slot writes them)."""
        T = attn_mask.shape[1]
        k = torch.stack([layer.keys for layer in cache.layers])
        v = torch.stack([layer.values for layer in cache.layers])
        n = attn_mask.sum(1)
        src = (torch.arange(T, device=n.device)[None] + (T - n)[:, None]).clamp(max=T - 1)  # [rows, T]
        src = src.view(1, -1, 1, T, 1).expand(k.shape[0], -1, k.shape[2], -1, k.shape[4])
        idx = torch.tensor(slots, device=self.cur.device)
        self.k[:, idx, :, :T] = k.gather(3, src)
        self.v[:, idx, :, :T] = v.gather(3, src)
        self.cur[idx] = n
        self.pos[idx] = (cur_pos + 1).view(-1, 1)

    @torch.no_grad()
    def step(self, x: torch.Tensor, running: torch.Tensor) -> torch.Tensor:
        """`x` [B, 1, H] -> the last hidden [B, H]; slots not `running` keep their position."""
        idx = self.cur
        live = idx + 1  # the slot's keys so far and the new one's
        cos, sin = self.rotary(x, self.pos)
        for i, layer in enumerate(self.layers):
            q, k, v = self.pre(layer, x, cos, sin)
            self.k[i][self.ar, :, idx] = k[:, :, 0]
            self.v[i][self.ar, :, idx] = v[:, :, 0]
            x = self.post(layer, x, attend(q[:, :, 0], self.k[i], self.v[i], live, layer.self_attn.scaling))
        step = running.long()
        self.cur += step
        self.pos += step.view(-1, 1)
        return self.norm(x)[:, 0]


class ContinuousFrame:
    """B slots, each generating its own chunk, one frame of all of them per CUDA graph replay. `admit` puts
    prompts into free slots; `step` runs a frame and hands back the chunks that ended in it."""

    def __init__(self, model, B: int, max_len: int, max_frames: int, *, temperature: float, top_k: int,
                 top_p: float, repetition_penalty: float, repetition_window: int):
        cfg = model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.model, self.B, self.max_frames = model, B, int(max_frames)
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)  # a graph per layer
        self.rep = SlotRepetition(B, cfg.n_vq, cfg.audio_vocab_size, repetition_window, dev)
        self.bb = SlotBackbone(model, B, max_len)
        self.prefiller = BatchedBackbone(model)
        self.acoustic = torch.compile(acoustic_frame, dynamic=False)
        self.h = torch.zeros(B, cfg.hidden_size, device=dev, dtype=dt)
        self.codes = torch.zeros(self.max_frames, B, cfg.n_vq, device=dev, dtype=torch.long)
        self.finished = torch.ones(B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(B, device=dev, dtype=torch.long)
        self.fr = torch.zeros(B, device=dev, dtype=torch.long)
        self.caps = torch.full((B,), self.max_frames, device=dev, dtype=torch.long)
        self.spk = torch.zeros(B, cfg.speaker_embedding_dim, device=dev, dtype=torch.float32)
        self.ar = torch.arange(B, device=dev)
        self._pad, self._start = cfg.audio_pad_token_id, cfg.speech_generation_start_token_id
        self.busy: set[int] = set()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):  # compiles, then warms the kernels the capture records
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._body()
        self.finished.fill_(True)

    def _body(self) -> None:
        running = ~self.finished
        codes, is_eos = self.acoustic(self.model, self.h, self.rep, **self.sampling)
        self.codes[self.fr.clamp(max=self.max_frames - 1), self.ar] = codes
        n = self.fr + 1
        done_now = running & (is_eos | (n >= self.caps))
        self.length.copy_(torch.where(done_now, n, self.length))
        self.finished.logical_or_(done_now)
        self.fr.copy_(torch.where(running, n, self.fr))
        B, n_vq = codes.shape
        slot = torch.full((B, 1, n_vq + 1), self._pad, dtype=torch.long, device=codes.device)
        slot[:, :, 0] = self._start
        slot[:, 0, 1:] = codes
        se = self.model._build_inputs_embeds(slot, speaker_emb=self.spk)
        self.h.copy_(self.bb.step(se, running & ~done_now))

    @property
    def free(self) -> list[int]:
        return [s for s in range(self.B) if s not in self.busy]

    @torch.no_grad()
    def admit(self, slots: list[int], embeds: list[torch.Tensor], speakers: list, caps: list[int]) -> None:
        """Prompt embeddings [T, H] into free `slots`, each to make at most its cap of frames (and no more than
        its slot's cache holds after the prompt)."""
        caps = [min(c, self.max_frames, self.bb.max_len - 1 - e.shape[0]) for c, e in zip(caps, embeds)]
        h, cache, mask, pos = self.prefiller.prefill(embeds)
        self.bb.load(slots, cache, mask, pos)
        idx = torch.tensor(slots, device=self.h.device)
        self.h[idx] = h
        self.spk[idx] = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in speakers]).to(idx.device)
        self.caps[idx] = torch.tensor(caps, device=idx.device)
        self.length[idx] = 0
        self.fr[idx] = 0
        self.rep.reset(idx)
        self.finished[idx] = False
        self.busy.update(slots)

    @torch.no_grad()
    def step(self) -> list[tuple[int, torch.Tensor]]:
        """One frame for every slot: (slot, its chunk's codes [T, n_vq]) for each chunk that ended in it."""
        self.graph.replay()
        finished = self.finished.tolist()
        ended = [s for s in self.busy if finished[s]]
        if not ended:
            return []
        lengths = self.length.tolist()
        self.busy.difference_update(ended)
        return [(s, self.codes[: lengths[s], s].clone()) for s in ended]
