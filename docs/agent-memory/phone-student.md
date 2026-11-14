# Phone student

Read when touching `training/student/`, `streams.model_rows` or a `StreamDataset`'s `codebooks`, or
when cutting, training or scoring a smaller model for the phone or its parallel codebook heads.
Back to [CONTEXT.md](../../CONTEXT.md) · [Cut](#cut) · [Training a student](#training-a-student) ·
[Checks](#checks) · [Next](#next)

The 3B misses real time on the A15, so the phone model is cut from the two-way model and scored in
both directions ([plan](../vi_en_parity_plan.md#5-the-phone-model), stages 5.3 and 5.4; order steps
9 and 10): first the depformer student (8 codebooks a stream, ~10 main layers, a 3-layer
depformer), then a ~14-layer student with parallel codebook heads in a delay pattern, trained on
the same streams for the same steps, adopted only if it wins at real time. Both start once GRPO
has frozen the two-way model (user, 2026-11-11: "continue with the current plan, can prepare the
code now").

## Cut

`python -m training.student.cut --weight <parent> --out-dir <dir>` writes `<dir>/config.json` and
`<dir>/model.safetensors`, every kept tensor copied whole from the parent:

```text
parent (28 layers, 16+16 codebooks, depformer 6 layers x 9 step slices)
  main layers      --layers 0,3,... | --num-layers 10 (spread: 0,3,6,9,12,15,18,21,24,27)
  emb (inputs)     target 0..k-1, then source 0..k-1 (parent emb 0..k-1, 16..16+k-1)
  depformer        --depformer-layers | --num-depformer-layers 3 (0,2,5); in each, step slices 0..k-1
  per step         depformer_in, depformer_norms, linears: the first k steps (depformer_emb: steps 1..k-1)
  unchanged        text_emb, text_linear, out_norm, depformer_text_emb
```

- `--codebooks` (k, default 8; up to 16, for stage 5.3's EN→VI check at 16) keeps the target's and
  the source's first k: Mimi's codebooks are residual, so they are the cache's first k. The kept
  steps' weight slices must be the parent's 0..m without gaps (Hibiki-Zero's steps 8–15 share
  slice 8, which a cut past 9 keeps shared); the schedule is written only if the parent has one.
- The config is the parent's with `num_layers`, `n_q` (2k), `dep_q` (k), `delays` (text, then each
  stream's first k) and the step schedule cut alike, naming the student's own weight
  (`moshi_name`) and the tokenizer it is on (`tokenizer_name`). moshi hands every config key to the model, so
  the cut's record goes only into the weight's metadata (`student`, `student_of`), beside the
  parent's own (its `text_tokenizer`, which `run_tokenizer` follows).
- The cut builds the student on the meta device from its config and requires every tensor's name
  and shape to match its parent's.

## Training a student

`training.finetune.train` with `--config-path <dir>/config.json --model-weight
<dir>/model.safetensors`: `get_mimi` follows the config (k codebooks), and each stream is
assembled at the cache's layout (`[33, T]`, silence encoded at the cache's 16 codebooks), then cut
to the model's rows (`streams.model_rows`: text, target 0..k-1, source 0..k-1). `validate.py` does
the same; GRPO's datasets keep the 3B's full width.

A student run reads `<dir>/config.json` and its start weight at every load (eval, validate,
resume), and neither syncs to the run's Hub prefix: keep the student dir with the run, or recut it
on a new box (the cut is deterministic from the parent and the weight's `student` record).

## Checks

- The cut's mapping (2026-11-11, box CPU, the two-way start weight, a 120-frame validation
  stream): an 8-codebook student keeping all 28 layers and all 6 depformer layers against the
  parent with its codebooks 8–15's input embeddings zeroed. Text logits equal exactly, the first 8
  codebooks' audio logits within 2.4e-5 (946 positions; the depformer's steps batched 8 against
  16), masks equal. Masking those codes to -1 instead is no test: moshi still embeds each
  codebook's initial token (2048) at the first step and the delayed positions.
- The default cut (10 layers, 3 depformer layers, 8 codebooks): 1.393B parameters against the
  parent's 3.13B; untrained, a validation stream's teacher-forced loss is 11.6 (audio 8.2, text
  3.4; the parent's ~2.9).
- Generation on the CPU (`LMGen`, moshi's per-step depformer at k slices): the default student
  emits text and 8 codebooks a step from the 3rd step on, its Mimi at 8 codebooks; a 16-codebook
  cut (10 layers, 1.546B parameters) text and 16, its Mimi at 16.
- The trainer's forward pins the GPU's efficient attention and compiles: a CPU check runs eager
  with `lm_training.sdpa_kernel` replaced, and needs Hibiki's (empty) condition tensors.

## Next

- Parallel heads (stage 5.4): k linear heads on the main transformer's output instead of the
  depformer, codebook j delayed j frames (MusicGen's pattern; moshi's `delays` carry it), heads
  trained from scratch; the trainer's forward, generation's per-step sampling (moshi's
  `depformer_step`) and the model's loading need the variant. The removed scaffold at 6a48332
  (`student/parallel.py`, a shared-trunk head distilled on a frozen student's states) is a
  different design.
- Speech evaluation in both directions: each target language's ASR transcript against the
  reference and speaker similarity to the source (evaluation is text-only today).
- The MLX and Swift runtimes for the student's config (the Swift app already times untrained cuts)
  and for the heads.
