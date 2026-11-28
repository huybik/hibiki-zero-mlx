"""Distillation without the full logits: each output linear and its KL from the teacher's run in
chunks of positions with an exact analytical backward (`compute_distill_losses_chunked`, where
`can_chunked_distill` holds), the losses `losses.compute_batch_losses` gives with a teacher.
"""
from __future__ import annotations

from typing import Any

import torch

from training.finetune import lm_training
from training.finetune.losses import LossWeights, audio_ar_end, loss_means
from training.student import slices


@torch.compile(dynamic=True)
def _kl_chunk_fwd_bwd(
    z_s: torch.Tensor, z_t: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused chunk KL(p_t || p_s) and exact analytical logit gradient w * (p_s - p_t)."""
    log_pt = torch.log_softmax(z_t.float(), dim=-1)
    log_ps = torch.log_softmax(z_s.float(), dim=-1)
    pt = log_pt.exp()
    kl = (pt * (log_pt - log_ps)).sum(dim=-1)
    dz = ((log_ps.exp() - pt) * w.unsqueeze(-1)).to(z_s.dtype)
    return kl, dz


@torch.compile(dynamic=True)
def _kl_chunk_fwd_only(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    log_pt = torch.log_softmax(z_t.float(), dim=-1)
    log_ps = torch.log_softmax(z_s.float(), dim=-1)
    return (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1)


def _chunk_mm_dtype(x: torch.Tensor) -> torch.dtype:
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


class _ChunkedLinearKL(torch.autograd.Function):
    """Exact chunked linear projection + forward KL(teacher || student) with analytical backward
    (zero full-vocab logit or log_softmax materialization, zero recomputation)."""

    @staticmethod
    def forward(
        ctx: Any,
        h_s: torch.Tensor,
        w_s: torch.Tensor,
        h_t: torch.Tensor,
        w_t: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        need_h = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[1])
        mm_dtype = _chunk_mm_dtype(h_s)
        h_s_mm = h_s.to(mm_dtype)
        w_s_mm = w_s.to(mm_dtype)
        h_t_mm = h_t.to(mm_dtype)
        w_t_mm = w_t.to(mm_dtype)
        N = h_s_mm.shape[0]
        kl_out = torch.empty((N,), device=h_s.device, dtype=torch.float32)
        dh_s = torch.empty_like(h_s_mm) if need_h else None
        dw_s_acc = torch.zeros(w_s.shape, device=w_s.device, dtype=torch.float32) if need_w else None

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            z_t = torch.nn.functional.linear(h_t_mm[s:e], w_t_mm)
            z_s = torch.nn.functional.linear(h_s_mm[s:e], w_s_mm)
            if need_h or need_w:
                kl_c, dz = _kl_chunk_fwd_bwd(z_s, z_t, weights[s:e])
                if need_h:
                    assert dh_s is not None
                    dh_s[s:e] = torch.matmul(dz, w_s_mm)
                if need_w:
                    assert dw_s_acc is not None
                    dw_s_acc.add_(torch.matmul(dz.T, h_s_mm[s:e]).float())
            else:
                kl_c = _kl_chunk_fwd_only(z_s, z_t)
            kl_out[s:e] = kl_c

        weighted_sum = (kl_out * weights).sum()
        dw_s = dw_s_acc.to(w_s.dtype) if dw_s_acc is not None else None
        ctx.h_dtype = h_s.dtype
        ctx.w_dtype = w_s.dtype
        ctx.save_for_backward(
            dh_s if dh_s is not None else h_s.new_empty((0,)),
            dw_s if dw_s is not None else w_s.new_empty((0,)),
        )
        ctx.mark_non_differentiable(kl_out)
        return weighted_sum, kl_out

    @staticmethod
    def backward(ctx: Any, grad_sum: torch.Tensor, _grad_kl: Any) -> tuple[Any, ...]:
        dh_s, dw_s = ctx.saved_tensors
        gh = (dh_s * grad_sum.to(dh_s.dtype)).to(ctx.h_dtype) if ctx.needs_input_grad[0] else None
        gw = (dw_s * grad_sum.to(dw_s.dtype)).to(ctx.w_dtype) if ctx.needs_input_grad[1] else None
        return gh, gw, None, None, None, None


class _ChunkedBatchedLinearKL(torch.autograd.Function):
    """Batched chunked linear+KL across K independent codebooks `[K, N, D]` using `torch.bmm`."""

    @staticmethod
    def forward(
        ctx: Any,
        h_s: torch.Tensor,
        w_s: torch.Tensor,
        h_t: torch.Tensor,
        w_t: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        need_h = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[1])
        mm_dtype = _chunk_mm_dtype(h_s)
        h_s_mm = h_s.to(mm_dtype)
        w_s_mm = w_s.to(mm_dtype)
        h_t_mm = h_t.to(mm_dtype)
        w_t_mm = w_t.to(mm_dtype)
        w_s_t = w_s_mm.transpose(1, 2)
        w_t_t = w_t_mm.transpose(1, 2)
        K, N, _ = h_s_mm.shape
        kl_out = torch.empty((K, N), device=h_s.device, dtype=torch.float32)
        dh_s = torch.empty_like(h_s_mm) if need_h else None
        dw_s_acc = torch.zeros(w_s.shape, device=w_s.device, dtype=torch.float32) if need_w else None

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            z_t = torch.bmm(h_t_mm[:, s:e], w_t_t)
            z_s = torch.bmm(h_s_mm[:, s:e], w_s_t)
            if need_h or need_w:
                kl_c, dz = _kl_chunk_fwd_bwd(z_s, z_t, weights[:, s:e])
                if need_h:
                    assert dh_s is not None
                    dh_s[:, s:e] = torch.bmm(dz, w_s_mm)
                if need_w:
                    assert dw_s_acc is not None
                    dw_s_acc.add_(torch.bmm(dz.transpose(1, 2), h_s_mm[:, s:e]).float())
            else:
                kl_c = _kl_chunk_fwd_only(z_s, z_t)
            kl_out[:, s:e] = kl_c

        weighted_sum = (kl_out * weights).sum()
        dw_s = dw_s_acc.to(w_s.dtype) if dw_s_acc is not None else None
        ctx.h_dtype = h_s.dtype
        ctx.w_dtype = w_s.dtype
        ctx.save_for_backward(
            dh_s if dh_s is not None else h_s.new_empty((0,)),
            dw_s if dw_s is not None else w_s.new_empty((0,)),
        )
        ctx.mark_non_differentiable(kl_out)
        return weighted_sum, kl_out

    @staticmethod
    def backward(ctx: Any, grad_sum: torch.Tensor, _grad_kl: Any) -> tuple[Any, ...]:
        dh_s, dw_s = ctx.saved_tensors
        gh = (dh_s * grad_sum.to(dh_s.dtype)).to(ctx.h_dtype) if ctx.needs_input_grad[0] else None
        gw = (dw_s * grad_sum.to(dw_s.dtype)).to(ctx.w_dtype) if ctx.needs_input_grad[1] else None
        return gh, gw, None, None, None, None


class _ChunkedSharedInputLinearKL(torch.autograd.Function):
    """Chunked linear+KL where K parallel student heads (`w_s [K, V, D_s]`) share ONE student
    hidden state (`h_s [N, D_s]`, e.g. slice 8 emitting codebooks 8..15) against K teacher
    hidden states (`h_t [K, N, D_t]`, `w_t [K, V, D_t]`). Runs one fused GEMM for the student."""

    @staticmethod
    def forward(
        ctx: Any,
        h_s: torch.Tensor,
        w_s: torch.Tensor,
        h_t: torch.Tensor,
        w_t: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        need_h = bool(ctx.needs_input_grad[0])
        need_w = bool(ctx.needs_input_grad[1])
        mm_dtype = _chunk_mm_dtype(h_s)
        h_s_mm = h_s.to(mm_dtype)
        w_s_mm = w_s.to(mm_dtype)
        h_t_mm = h_t.to(mm_dtype)
        w_t_mm = w_t.to(mm_dtype)
        K, V, D_s = w_s_mm.shape
        N = h_s_mm.shape[0]
        w_s_flat = w_s_mm.reshape(K * V, D_s)
        w_t_t = w_t_mm.transpose(1, 2)
        kl_out = torch.empty((K, N), device=h_s.device, dtype=torch.float32)
        dh_s = torch.empty_like(h_s_mm) if need_h else None
        dw_s_acc = torch.zeros((K * V, D_s), device=w_s.device, dtype=torch.float32) if need_w else None

        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            C = e - s
            z_t = torch.bmm(h_t_mm[:, s:e], w_t_t)  # [K, C, V]
            z_s = torch.nn.functional.linear(h_s_mm[s:e], w_s_flat).view(C, K, V).transpose(0, 1)
            if need_h or need_w:
                kl_c, dz = _kl_chunk_fwd_bwd(z_s, z_t, weights[:, s:e])  # dz: [K, C, V]
                dz_flat = dz.transpose(0, 1).reshape(C, K * V)
                if need_h:
                    assert dh_s is not None
                    dh_s[s:e] = torch.matmul(dz_flat, w_s_flat)
                if need_w:
                    assert dw_s_acc is not None
                    dw_s_acc.add_(torch.matmul(dz_flat.T, h_s_mm[s:e]).float())
            else:
                kl_c = _kl_chunk_fwd_only(z_s, z_t)
            kl_out[:, s:e] = kl_c

        weighted_sum = (kl_out * weights).sum()
        dw_s = dw_s_acc.view(K, V, D_s).to(w_s.dtype) if dw_s_acc is not None else None
        ctx.h_dtype = h_s.dtype
        ctx.w_dtype = w_s.dtype
        ctx.save_for_backward(
            dh_s if dh_s is not None else h_s.new_empty((0,)),
            dw_s if dw_s is not None else w_s.new_empty((0,)),
        )
        ctx.mark_non_differentiable(kl_out)
        return weighted_sum, kl_out

    @staticmethod
    def backward(ctx: Any, grad_sum: torch.Tensor, _grad_kl: Any) -> tuple[Any, ...]:
        dh_s, dw_s = ctx.saved_tensors
        gh = (dh_s * grad_sum.to(dh_s.dtype)).to(ctx.h_dtype) if ctx.needs_input_grad[0] else None
        gw = (dw_s * grad_sum.to(dw_s.dtype)).to(ctx.w_dtype) if ctx.needs_input_grad[1] else None
        return gh, gw, None, None, None, None


def chunked_linear_kl(
    h_s: torch.Tensor,
    w_s: torch.Tensor,
    h_t: torch.Tensor,
    w_t: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ChunkedLinearKL.apply(h_s, w_s, h_t, w_t, weights, chunk_size)


def chunked_batched_linear_kl(
    h_s: torch.Tensor,
    w_s: torch.Tensor,
    h_t: torch.Tensor,
    w_t: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ChunkedBatchedLinearKL.apply(h_s, w_s, h_t, w_t, weights, chunk_size)


def chunked_shared_input_linear_kl(
    h_s: torch.Tensor,
    w_s: torch.Tensor,
    h_t: torch.Tensor,
    w_t: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ChunkedSharedInputLinearKL.apply(h_s, w_s, h_t, w_t, weights, chunk_size)


def can_chunked_distill(lm: Any, teacher: Any | None) -> bool:
    """Whether `compute_distill_losses_chunked` applies (matching delays, unbiased output linears)."""
    if teacher is None:
        return False
    if getattr(lm, "linears", None) is None or getattr(teacher, "linears", None) is None:
        return False
    s_audio = list(lm.delays[lm.audio_offset : lm.audio_offset + lm.dep_q])
    t_audio = list(teacher.delays[teacher.audio_offset : teacher.audio_offset + lm.dep_q])
    if lm.delays[0] != teacher.delays[0] or s_audio != t_audio:
        return False
    if lm.delays[0] != 0 or s_audio[0] != 0:
        return False
    if len(s_audio) > 1 and len(set(s_audio[1:])) != 1:
        return False
    if slices.is_slices(teacher) and teacher.parallel_codebooks != lm.dep_q - audio_ar_end(lm):
        return False  # its last slice feeds other codebooks than the student's parallel ones

    if lm.text_linear.bias is not None or teacher.text_linear.bias is not None:
        return False
    return all(
        lm.linears[cb].bias is None and teacher.linears[cb].bias is None
        for cb in range(lm.dep_q)
    )


def compute_distill_losses_chunked(
    lm: Any,
    teacher: Any,
    student_codes: torch.Tensor,
    teacher_codes: torch.Tensor,
    condition_tensors: Any | None,
    teacher_condition_tensors: Any | None,
    weights: LossWeights,
    freeze_trunk: bool = False,
) -> dict[str, Any]:
    """Exact hidden-state distillation avoiding full-vocab logit materialization (~5.5 GiB saved).
    When `freeze_trunk` is True, runs the student trunk under `no_grad` and skips `text_linear`
    and `text_kl` for both teacher and student."""
    B, _, T = student_codes.shape
    dep_q = int(lm.dep_q)
    ar_end = audio_ar_end(lm)
    d_ac = int(lm.delays[lm.audio_offset + 1]) if dep_q > 1 else 0
    T_ac = max(0, T - d_ac)
    is_slices_model = slices.is_slices(lm)

    with torch.no_grad():
        h_t_trunk, delayed_t = lm_training.forward_trunk_hidden(
            teacher, teacher_codes, teacher_condition_tensors
        )
        normed_t = lm_training.forward_depformer_hidden(teacher, delayed_t[:, :, 1:], h_t_trunk)
        del delayed_t
        if freeze_trunk:
            h_t_trunk = None

    if freeze_trunk:
        with torch.no_grad():
            h_s_trunk, delayed_s = lm_training.forward_trunk_hidden(
                lm, student_codes, condition_tensors
            )
    else:
        h_s_trunk, delayed_s = lm_training.forward_trunk_hidden(
            lm, student_codes, condition_tensors
        )
    normed_s = lm_training.forward_depformer_hidden(lm, delayed_s[:, :, 1:], h_s_trunk)
    del delayed_s

    D_s = normed_s[0].shape[-1]
    D_t = normed_t[0].shape[-1]
    audio_targets = student_codes[:, lm.audio_offset : lm.audio_offset + dep_q]

    # Codebook 0 (semantic, delay 0).
    valid_0 = audio_targets[:, 0] != lm.zero_token_id  # [B, T]
    w_0 = valid_0.float() * weights.first_codebook
    sem_weighted_sum, kl_0 = chunked_linear_kl(
        normed_s[0].reshape(-1, D_s),
        lm.linears[0].weight,
        normed_t[0].reshape(-1, D_t),
        teacher.linears[0].weight,
        w_0.reshape(-1),
        chunk_size=2048,
    )
    semantic_sum = (kl_0 * valid_0.reshape(-1)).sum()
    semantic_n = valid_0.sum()
    audio_sum = sem_weighted_sum
    audio_n = w_0.sum()
    zero = audio_sum.new_zeros(())

    # AR acoustic codebooks 1..ar_end-1 (delay d_ac).
    if ar_end > 1 and T_ac > 0:
        valid_ar = (audio_targets[:, 1:ar_end, :T_ac] != lm.zero_token_id).transpose(0, 1).reshape(ar_end - 1, -1)
        w_ar = valid_ar.float() * weights.acoustic_ar
        h_s_ar = torch.stack([normed_s[cb][:, d_ac:].reshape(-1, D_s) for cb in range(1, ar_end)], dim=0)
        w_s_ar = torch.stack([lm.linears[cb].weight for cb in range(1, ar_end)], dim=0)
        h_t_ar = torch.stack([normed_t[cb][:, d_ac:].reshape(-1, D_t) for cb in range(1, ar_end)], dim=0)
        w_t_ar = torch.stack([teacher.linears[cb].weight for cb in range(1, ar_end)], dim=0)
        ar_weighted_sum, kl_ar = chunked_batched_linear_kl(
            h_s_ar, w_s_ar, h_t_ar, w_t_ar, w_ar, chunk_size=2048
        )
        del h_s_ar, h_t_ar
        acoustic_ar_sum = (kl_ar * valid_ar).sum()
        acoustic_ar_n = valid_ar.sum()
        audio_sum = audio_sum + ar_weighted_sum
        audio_n = audio_n + w_ar.sum()
    else:
        acoustic_ar_sum = acoustic_ar_n = zero

    # Parallel (or upper) acoustic codebooks ar_end..dep_q-1 (delay d_ac).
    if ar_end < dep_q and T_ac > 0:
        k_par = dep_q - ar_end
        valid_par = (audio_targets[:, ar_end:, :T_ac] != lm.zero_token_id).transpose(0, 1).reshape(k_par, -1)
        w_par = valid_par.float()
        w_s_par = torch.stack([lm.linears[cb].weight for cb in range(ar_end, dep_q)], dim=0)
        if slices.is_slices(teacher):  # a slices teacher (self-distillation): its last slice feeds them all
            h_t_par = normed_t[-1][:, d_ac:].reshape(-1, D_t).expand(k_par, -1, -1)
        else:
            h_t_par = torch.stack([normed_t[cb][:, d_ac:].reshape(-1, D_t) for cb in range(ar_end, dep_q)], dim=0)
        w_t_par = torch.stack([teacher.linears[cb].weight for cb in range(ar_end, dep_q)], dim=0)
        if is_slices_model:
            h_s_par = normed_s[-1][:, d_ac:].reshape(-1, D_s)
            par_weighted_sum, kl_par = chunked_shared_input_linear_kl(
                h_s_par, w_s_par, h_t_par, w_t_par, w_par, chunk_size=2048
            )
        else:
            h_s_par = torch.stack([normed_s[cb][:, d_ac:].reshape(-1, D_s) for cb in range(ar_end, dep_q)], dim=0)
            par_weighted_sum, kl_par = chunked_batched_linear_kl(
                h_s_par, w_s_par, h_t_par, w_t_par, w_par, chunk_size=2048
            )
        del h_s_par, h_t_par
        acoustic_par_sum = (kl_par * valid_par).sum()
        acoustic_par_n = valid_par.sum()
        audio_sum = audio_sum + par_weighted_sum
        audio_n = audio_n + w_par.sum()
    else:
        acoustic_par_sum = acoustic_par_n = zero

    del normed_t
    acoustic_sum = acoustic_ar_sum + acoustic_par_sum
    acoustic_n = acoustic_ar_n + acoustic_par_n

    # Text KL (delay 0); skipped completely when trunk is frozen.
    if freeze_trunk:
        text_sum = text_n = content_sum = content_n = pad_sum = pad_n = zero
    else:
        assert h_t_trunk is not None
        text_targets = student_codes[:, :1]
        text_valid = text_targets != lm.zero_token_id
        pad = (text_targets == lm.text_padding_token_id) & text_valid
        content = text_valid & ~pad
        text_weight = torch.where(pad, weights.text_pad, 1.0) * text_valid
        text_sum, kl_text = chunked_linear_kl(
            h_s_trunk.reshape(-1, h_s_trunk.shape[-1]),
            lm.text_linear.weight,
            h_t_trunk.reshape(-1, h_t_trunk.shape[-1]),
            teacher.text_linear.weight,
            text_weight.reshape(-1),
            chunk_size=1024,
        )
        text_n = text_weight.sum()
        content_sum = (kl_text * content.reshape(-1)).sum()
        content_n = content.sum()
        pad_sum = (kl_text * pad.reshape(-1)).sum()
        pad_n = pad.sum()

    sums = {
        "audio_sum": audio_sum,
        "audio_n": audio_n,
        "semantic_sum": semantic_sum,
        "semantic_n": semantic_n,
        "acoustic_sum": acoustic_sum,
        "acoustic_n": acoustic_n,
        "acoustic_ar_sum": acoustic_ar_sum,
        "acoustic_ar_n": acoustic_ar_n,
        "acoustic_par_sum": acoustic_par_sum,
        "acoustic_par_n": acoustic_par_n,
        "text_sum": text_sum,
        "text_n": text_n,
        "content_sum": content_sum,
        "content_n": content_n,
        "pad_sum": pad_sum,
        "pad_n": pad_n,
    }
    result = {**sums, **loss_means(sums, weights)}
    if freeze_trunk:
        result["loss"] = weights.audio * (audio_sum / audio_n.clamp(min=1))
    return result
