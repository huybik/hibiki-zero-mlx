# Track B — Parallel Codebook Head (Phase 3 scaffold, smoke scale)

Self-distillation of a **parallel** codebook head to replace the AR depformer's 16
sequential slices (distill_plan.md §3–§9). This report documents the **working
scaffold + a smoke-scale distill on the M4** that proves the mechanism. It is
**not** a quality run — the head is trained on ~3 min of audio; audio quality is
rough by design. The phase gate is *mechanism works + speed measured*; the
**default runtime stays the AR head** (parallel is opt-in until trained at scale).

All numbers: M4 Pro, 3B q4, `weights/hibiki.q4.safetensors` frozen teacher.

## 1. Components & status

| Component | File | Status |
|---|---|---|
| Teacher dump | `distill/dump_teacher.py` | **working** — inference-only, resumable shards |
| Parallel head module | `distill/parallel_head.py` | **working** — vectorised delay-pattern head, `num_passes` knob |
| Distill trainer | `distill/train_head.py` | **working (smoke)** — KL+CE, head-only grads |
| Runtime integration | `moshi_mlx/models/lm.py`, `hibiki_mlx/pipeline.py`, `scripts/bench.py` | **working** — `HIBIKI_HEAD=parallel` switch |
| Iterative refinement (`num_passes` 2–4) | `parallel_head.py` | **placeholder** — runs, not tuned |
| q4 head + real-hours training | — | **follow-up** (see §7) |

## 2. Teacher dump format & size

`dump_teacher.py` runs the frozen model (reusing `LmGen`) over CoVoST2 fr wavs and
captures, per frame, via a capture hook added to `DepFormer.sample`:

- `transformer_out` — (main_dim=2048,) fp16 — the head's only conditioning input.
- `text_tokens` — int32 — the sampled text token (slice-0 input of the AR head).
- `teacher_tokens` — (N=16,) int32 — the teacher's sampled codebook tokens (CE target
  + the head's delay-1 conditioning).
- `teacher_logits` — (N=16, vocab=2048) fp16 — full teacher distribution (KL target).

Shards are `shard_*.safetensors` + `manifest.json` (resumable: records done clips /
shards / frames). Per-clip order is preserved via `clip_lengths` (needed to build the
delay-1 conditioning during training).

Smoke dump (`distill/teacher_3b/`): **30 clips / 2289 frames / 152 MB / 1 shard**,
~3.1 min audio, dumped in ~105 s. Disk ≈ **66 KB/frame** (dominated by the fp16
logits: 16×2048×2 = 64 KB). → **~2.9 GB/h**. Full logits are fine to ~5 h; past that,
switch to top-k logits (documented, not implemented — smoke uses full vocab for clean KL).

## 3. Head architecture

Delay-pattern, single parallel forward (distill_plan §3). Per-codebook projections /
embeddings / norms are stored as **stacked arrays** and applied with batched
einsum / grouped ops, so the whole head is a handful of GPU launches (vs the AR
head's ~370):

```
transformer_out (B,2048) --w_in[N,1024,2048] (einsum)--> (B,N,1024)
  + cb_emb[N,2049,1024] gather( prev-frame token per codebook )   # delay-pattern cond
  + text_emb(text_token) + pos_emb[N,1024]
  -> shared trunk: 6× bidirectional gated transformer layer over the N=16 positions
  -> grouped LayerNorm -> w_out[N,2048,1024] (einsum) -> logits (B,N,2048)
```

- **Warm start** (distill_plan §5): `w_in`, `w_out`, per-codebook norm, and the
  token/text embeddings are copied from the AR depformer's `linear_in` / `linear_out`
  / `norm` / `emb` (shapes match exactly). The trunk is trained from scratch.
- **`num_passes`**: 1 (default, fully parallel). >1 = MaskGIT-style refinement
  (re-condition on the current frame's provisional argmax) — placeholder.
- **Params: 227.0 M** (fp32 checkpoint 908 MB). Breakdown: w_in 33.5M, w_out 33.5M,
  cb_emb 33.5M, text_emb 49.2M, trunk ~76M. The text_emb (48001×1024) and the
  un-shared per-codebook w_in/w_out dominate — obvious shrink targets for the real run.

## 4. Smoke loss curves

Trainer: `loss = KL(teacher‖student, T=2) + CE(teacher tokens)`, Adam,
`nn.value_and_grad` on **head params only** (the head is a standalone module; the
frozen main is never in the graph), clip-clean train/val split.

**(a) Full smoke train** — 30 clips (train 2016 / val 273 frames), 500 steps, lr 3e-4,
warm-started. Train loss falls monotonically:

```
step    1  loss 21.40  kl 13.53  ce 7.87
step  100  loss  6.71  kl  3.40  ce 3.31
step  250  loss  4.59  kl  2.39  ce 2.20
step  500  loss  2.91  kl  1.72  ce 1.20
final val loss 11.05      (107 s)
```
Val stays high (single-shard, 30 clips → overfits) — expected at smoke scale; this
phase does not chase val quality.

**(b) Overfit sanity** — first 32 frames, 800 steps, lr 1e-3 (capacity + gradient flow):

```
step   1  loss 26.71
step  50  loss  1.20
step 200  loss  0.12
step 800  loss  0.044      final val loss 0.044  (136 s)
```
→ near-zero: the head has capacity and gradients flow end-to-end through the
einsum/gather/trunk.

## 5. Measured speed — AR vs parallel (the key deliverable)

`scripts/bench.py --model 3b`, 100 timed frames, clean (no background load):

| stage | AR head (default) | Parallel head (smoke, bf16) |
|---|---:|---:|
| mimi encode (CPU, hidden) | 19.1 ms | 19.3 ms |
| LM main transformer | 13.2 ms | 13.2 ms |
| **LM codebook head** | **12.70 ms** (16× 0.79) | **5.28 ms** (1 fwd) |
| mimi decode (CPU, hidden) | 17.7 ms | 17.7 ms |
| **LM total (critical path)** | **25.9 ms** | **18.5 ms** |
| pipelined live | 3.1× RT | **4.3× RT** |
| projected iPhone LM @0.5× | 51.9 ms | **37.0 ms** |

**Head: 12.70 → 5.28 ms (−58%). Frame LM total: 25.9 → 18.5 ms (−29%).** The AR head
was *launch-bound* (16 sequential slices); the parallel head is one forward, so it is
now **bandwidth-bound** on its 227M bf16 weights (~0.45 GB/frame read). Quantising the
head to q4 (§7) should read ~4× less and take it toward the 1–4 ms target.

## 6. Gates

| gate | AR (default — must pass) | Parallel (smoke — report only) |
|---|---|---|
| `verify_mlx_q4` text | **coherent** (leon Olympics translation) | **coherent** — identical text (text stream never routes through the head, distill_plan §1) |
| silence-in (rms<0.10, peak<1.1) | **PASS** rms 0.0002 peak 0.012 | **PASS** rms 0.0637 peak 0.739 |
| end-to-end run | ok | ok, 3.46× RT on leon |

The AR default path is unchanged and passes both gates. The parallel path — even at
3 min of training — clears silence-in and keeps text coherent; audio is rough (not
scored here).

## 7. Scale-up command list (real training run — follow-up, not this phase)

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python

# 1. Dump ~10-20 h of frozen-teacher targets (resumable; add more source wavs to the
#    manifest first). Past ~5 h, add a top-k-logits path to dump_teacher to cap disk.
$PY distill/dump_teacher.py --model 3b --hours 15 --out distill/teacher_3b

# 2. Distil the head. Recommended: warm start, more steps, LR decay, tune KL temp.
$PY distill/train_head.py --teacher distill/teacher_3b --steps 20000 --batch 128 \
    --lr 3e-4 --temp 2.0 --warm-start weights/hibiki.bf16.safetensors \
    --out distill/head_3b.safetensors
#   Then sweep num_passes 1→4 (--passes) for the quality/speed knee (distill_plan §3).

# 3. Run behind the flag / bench the drop.
HIBIKI_HEAD=parallel HIBIKI_HEAD_CKPT=distill/head_3b.safetensors \
    $PY scripts/bench.py --model 3b --frames 100
HIBIKI_HEAD=parallel HIBIKI_HEAD_CKPT=distill/head_3b.safetensors \
    $PY main.py assets/samples/leon.wav

# 4. (perf) Quantise the head to q4-gs32 to make it bandwidth-cheap (target 1-4 ms);
#    shrink params first (share w_in/w_out across codebooks, low-rank text_emb).
# 5. Score CoVoST2 fr->en ASR-BLEU vs the AR teacher; aim for a small BLEU drop.
```

Recommended hyperparameters for the real run: **~15 h** source audio, batch 128,
20k steps, lr 3e-4 with cosine decay, KL temp 2.0, KL:CE = 1:1, warm start on.
Expect the audio to become usable only at this scale; then quantise + BLEU-gate.
