# The slices student's on-policy stage and phone finish

Read when running or judging the slices student's on-policy run (`both_slices_z`), its 6-bit
palettization and its export to the phone. Back to [CONTEXT.md](../../CONTEXT.md) ·
[Profile](#profile) · [The run](#the-run) · [Phone finish](#phone-finish). The student and its
off-policy stages are in [student-slices](student-slices.md); the method in
[on-policy-distillation](on-policy-distillation.md), GRPO in [grpo](grpo.md).

## Why on-policy

Training the trunk off-policy on the SFT data lost text on every set (the joint stage,
[student-slices](student-slices.md#runs)): the trunk's text came from `both_heads_z`'s on-policy
run. So the trunk changes only on-policy, with `both_heads_z`'s recipe (both terms, z-scored, the
3B teacher, 1e-6), which also distils the audio codebooks on the student's own rollouts.

## Profile

`grpo_profile` on box 62513802 (RTX 5090, 2026-11-27 16:42 UTC) from Stage 1's decay at 8,000
(`weights/student_slices14_decay8000.safetensors`, hard-linked), `both_heads_z`'s data, the
teacher, `--park-states`; raw report `runs/grpo_profile_slices/profile.json`, copied to
`/Volumes/data/hibiki-zero-evals/box-62513802/grpo_profile_slices/`:

- Generation: 192 / 320 / 384 rows at 14.5 / 20.2 / 23.2 ms a step, peaks 15.1 / 23.4 / 27.6 GiB;
  a step costs 5.7 ms + 0.045 ms a row. A window's generation is 6.72 s an update at 320 rows.
- π_old's pass and the update at micro-batch 1 / 2 / 4: 1.36 / 1.26 / 1.18 s and 8.14 / 7.72 /
  7.35 s an update, peaks 20.9 / 23.9 / 28.0 GiB. First update's ratio deviation 0 at each.
- Start: `kl_text` 0.13–0.14, `kl_audio` 0.76–0.90 (the heads student's 1.3–1.4), final BLEU of the
  profile's rollouts 38.1.
- Kernels (`torch.profiler` over the profile at 320 rows and micro-batch 2, `prof_grpo.py` beside
  the records): the narrowed grouped-KV step attention (`fmha_cutlassF`, a boolean mask) is 49% of
  the CUDA time, ~0.8 ms a layer a step, near the card's bandwidth for 320 rows' KV cache; the
  update's GEMMs (bf16 under fp32 master weights) most of the rest.
- Levers left out: FP8 in the update (train.py's `--fp8`, `--fp8-teacher`) would save ~12% of an
  update at the cost of the teacher's targets (`all` flips 12–17% of acoustic argmaxes) and the
  policy's log-probabilities; `--bf16-adamw` cannot hold a 1e-6 step in bf16 weights; 384 rows
  and micro-batch 4 save ~1% and ~2% but peak within 4 GiB of 32 (the heads run held more than its
  profile). Sizes chosen: `--gen-frames 384000 --micro-batch 2 --park-states`, ~16.5 s an update.

## The run

## Phone finish
