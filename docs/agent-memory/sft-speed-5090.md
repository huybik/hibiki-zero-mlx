# SFT and distillation speed on the RTX 5090

Read when touching `training/finetune/bf16_adamw.py`, `training/finetune/fp8_linear.py`, `train.py
--bf16-adamw`, `--fp8` or `--fp8-teacher`, or measuring training on a 32 GB Blackwell card.
Back to [CONTEXT.md](../../CONTEXT.md) · [Levers](#levers) · [Measured](#measured) ·
[Slices distillation kernels](#slices-distillation-kernels). The trainer's fast path and the other
GPUs are in [sft-speed](sft-speed.md).

RTX 5090: 32 GB, sm_120, $0.511/h, ~515 W (2026-11-26, box 62513802). Full-model 3B SFT and 14L
16-codebook heads-student distillation both run on it at ~39–44% lower $/M frames than an H100 SXM
($0.0363/M at $2/h) and an RTX PRO 6000 ($0.0200/M at $1.217/h).

## Levers

- Ruled out (2026-11-23): 3B fp32 SFT (46.7 GiB of static states), host-offload AdamW (~94 GB host
  traffic a step, estimated 1,400–2,700 fr/s) and 8-bit blockwise moments (29 GiB static, broke a
  tensor, [sft-speed](sft-speed.md#measurements)).
- **`--bf16-adamw`** (`training/finetune/bf16_adamw.py`): keeps 2D+ weights, gradients, and AdamW
  moments (`m`, `v`) in bf16 (8 bytes/param = **23.3 GiB** on 3B, **15.4 GiB** on student + 3B
  bf16 teacher), keeping 1D norms/scales/biases in fp32. A single fused Triton kernel runs each
  AdamW update in fp32 registers with fused gradient clipping (no separate grad-scaling pass or
  host sync) and unbiased IEEE-754 stochastic rounding (`tl.randint`) back to bf16 for `p`, `m`,
  and `v`, preserving fp32's full 8-bit exponent range so `v` never underflows to 0.
- **50-step real-data parity** (PhoAudiobook `pho`, `lr=1e-5`, `batch_frames=4608`):
  - `fp32_adamw`: 17,790 fr/s, 24.48 GiB, `mean_last10=2.2132`;
  - `bf16_adamw`: 22,466 fr/s (+26%), 12.96 GiB, `mean_last10=2.2132` (`mean|diff|=0.00033`,
    `last10=+0.00001`);
  - `bf16_adamw + --fp8`: `mean_last10=2.2152` (`+0.00207`, slower on small student batches).
- **Selective checkpointing (`--grad-checkpoint --grad-checkpoint-layers N`) and `--fp8`**
  (`training/finetune/fp8_linear.py`, rowwise E4M3 `torch._scaled_mm` on the main transformer's
  linears, and a depformer's or its slices' with `depformer_in`, in training): whole-loop compiled
  checkpointing in bf16 fuses norms/residuals across layers; per-layer compile under `--fp8`
  avoids unrolling 28 quantize+scaled_mm blocks (~20× faster compile). Keeping Mimi on CPU during
  training saves ~0.4 GiB; detaching `step_losses` and avoiding long-lived gradient list
  references prevents a 5.8 GiB step-2 leak.
- The 3B's and the heads student's 5090 routes passed full preempt → resume (exact 4-digit loss
  match) → teacher-forced + free-running FLEURS validation (15.9 GiB peak with `--park-states`) →
  promotion end-to-end tests on box 62513802; the slices' `restart.sh student-slices*-5090` cases
  run on the card.

## Measured

On real variable-shape batches (31.4 GiB usable VRAM, `--park-states` for validation):

| Config | frames/s | s/step | Peak VRAM | $/M frames (@ $0.511/h) |
|---|---:|---:|---:|---:|
| 3B `--bf16-adamw --grad-checkpoint --grad-checkpoint-layers 14` (`4,608` whole) | 6,554 | 0.703 | 28.3 GiB | $0.0217 |
| 3B `--bf16-adamw --fp8 --grad-checkpoint --grad-checkpoint-layers 8` (`4,608` whole) | 7,242 | 0.636 | 28.1 GiB | $0.0196 |
| Student 14L 16cb + 3B teacher, exact fp32 + `--grad-checkpoint` (`12,288` in 4 accumulated slices) | 10,065 | 1.221 | 27.4 GiB | $0.0141 |
| Student 14L 16cb + 3B teacher, `--bf16-adamw` (`12,288` in 2 accumulated slices, no ckpt) | 12,417 | 0.902 | 26.0 GiB | $0.0114 |
| Slices student (`student_slices14`, joint) + 3B teacher, `--bf16-adamw --fp8 --fp8-teacher all` (`12,288` in 2 accumulated slices, chunked KL + Triton FP8) | 17,604 | 0.698 | 25.5 GiB | $0.0081 |
| Slices student (`student_slices14`, `--freeze-trunk`) + 3B teacher, `--bf16-adamw --fp8 --fp8-teacher all` (`12,288` whole) | 28,710 | 0.428 | 15.8 GiB | $0.0049 |
| Wide slices (`student_slices14_wide`, 9 of 1024 × 3, `--freeze-trunk`), the same arguments (`12,288` whole; 2026-11-28) | 21,826 | 0.563 | 27.0 GiB | $0.0065 |

The wide slices' row is after the 2026-11-28 kernels below (0.605 s before them); its breakdown is
in [student-slices](student-slices.md#wide-slices). `train.py` runs every batch whole (no gradient
accumulation): the sliced rows are measurements of a trainer option since removed.

## Slices distillation kernels

2026-11-27 and 11-28, box 62513802:

- `common.compute_distill_losses_chunked` computes linear projections + exact forward KL in
  chunks (`2048` audio / `1024` text) without materializing full-vocab logits (~5.5 GiB saved),
  and skips `text_linear` / `text_kl` when `--freeze-trunk` is set. A slices teacher (the
  6-bit finish distils the slices student from its own bf16 weights) feeds its last slice's
  output to every parallel codebook, as the student side does.
- `lm_training._per_step` batches contiguous duplicate schedule indices (8× fewer teacher
  slice-8 GEMMs/HBM reads), `project_depformer_in` fuses `depformer_in` into one GEMM,
  `slices.py` fuses `linears[8..15]`, and `_rope_tables` caches RoPE once per forward.
- `_depformer_sdpa` uses cuDNN for forward-only teacher attention (1.57× faster on Blackwell) and,
  with gradients, the explicit causal form compiled into the layers: the wide slices' K=9
  attention `[12,288, 16, 9, 64]` forward and backward 3.9 ms against memory-efficient's 10.7,
  cuDNN's 11.8 and flash's 16.7 (2026-11-28).
- `fp8_linear.py` uses single-pass/two-pass Triton rowwise E4M3 quantization, and a tiled
  transpose-quantize in two passes over `[128, 64]` tiles (the columns' amax by atomic max, then
  quantize and store transposed), so a tall activation fills the 170 SMs: 23.9 µs at
  `[12,288, 1,024]` against 51.9 for the previous single-axis `BLOCK_K=16` kernel, 361 against 768
  at `[110,592, 1,024]`, byte for byte the same output; ~8 µs slower on a small weight
  (`[3,072, 1,024]`). Each is a `torch.library.custom_op` (no graph breaks under whole-loop
  `torch.compile`), plus `FrozenFP8Linear` (`--fp8-teacher all`) and plain-attribute frozen
  student trunk FP8 caches (`--freeze-trunk --fp8`).
- FP8 against bf16 at the wide slices' shapes (one linear's forward and backward, `M` = 12,288):
  1.24–1.6× faster, 3.75% relative error against an fp32 reference (bf16's 0.01–0.03%), the
  rowwise E4M3 error the main transformer's `--fp8` already has.
- Full table and kernel profiles in [student-slices](student-slices.md#training).
