# Phone student

Read when touching `training/student/`, `streams.model_rows` or a `StreamDataset`'s `codebooks`, or
when cutting, training or scoring a smaller model for the phone or its parallel codebook heads.
Back to [CONTEXT.md](../../CONTEXT.md) · [Cut](#cut) · [Training a student](#training-a-student) ·
[Parallel heads](#parallel-heads) · [Checks](#checks) · [Next](#next)

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

A student run reads `<dir>/config.json` at every load (eval, validate, resume) and its start
weight on every resume, and neither syncs to the run's Hub prefix: keep the student dir with the
run, or recut it on a new box (the cut is deterministic from the parent and the weight's `student`
record, a heads student's seed among it).

## Parallel heads

`--heads` cuts the delay-pattern student (stage 5.4; the plan's is ~14 layers: `--num-layers 14`):
the same main layers and codebooks, no depformer, and `training/student/heads.py`'s
`ParallelHeads` (one linear head a codebook, initialized afresh with `--seed`) in the model's
`depformer` slot. The heads read the main transformer's normed output plus the model's own text
embedding of the step's text token (teacher-forced in training, the just-sampled token in
generation), as the depformer's first step reads that token: speech then follows the inner
monologue, and the comparison with the depformer student isolates the parallel heads. Text is
sampled before audio at each step, so this adds no latency and no parameters.

- Codebook j is delayed j frames (`heads.delays`, written into the config's target delays; the
  source's keep the parent's): a head predicting codebook j at a step has read codebook j-1 of the
  same frame in the step's input, as in MusicGen. moshi's own delay handling then carries it
  through training (logits undelayed, masks) and generation (`LMGen` withholds the first k-1
  steps: 7 at 8 codebooks, 15 at 16). The trunk also relearns its input timing: the parent delays
  target codebooks 1–15 by 2 frames, the heads student codebook j by j. Its larger `max_delay`
  delays its speech, not its text, so evaluation's `end_offset_s` takes the extra frames off and
  the two students' text latency compares directly.
- In the `depformer` slot because generation samples audio only when a depformer exists:
  `heads.py` replaces `LMGen.depformer_step` (all codebooks sampled at once from the heads; any
  other model runs moshi's steps) and, per model, `forward_depformer_training`. The depformer's
  tensors are dropped; the config's `depformer_num_layers` is 1, the throwaway moshi builds first.
- The config marks it with `"audio_heads": "parallel"`; `common.load_checkpoint_info` takes the
  key out before moshi reads the config, and every model is built through `common.build_lm`
  (a heads model: built without weights, heads installed, then loaded, the EOS row copied from
  PAD's as moshi's own load does).

## Checks

- End to end on the H100 (2026-11-14, both students cut from GRPO update 1,400, the two-way
  caches, 4,608-frame batches, 200 steps each): dep10 trains 11.0 → 6.3 at 0.107 s a step (30.9 GB
  peak), heads14 11.3 → 8.4 at 100 steps, 0.101 s (27.1 GB); both save, validate (`validate.py`
  equals the trainer's own to 1e-3) and resume exactly (losses equal to the 4th decimal); `eval.py`
  scores them and decodes their speech at 8 codebooks; a recut reproduces every tensor. The heads
  student's text at output index i is the token sampled at step i for all 1,852 words, so the
  end offset's `max_delay − 2` correction holds. On the 3B the branch's teacher-forced losses
  equal main's bit for bit, and SFT and GRPO run. At 2.25e-6 a student barely learns in 200
  steps (dep10's text argmax is always PAD), so its rate is probed first ([Next](#next)).
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
- The heads student (14 layers [0, 2, …, 27], 8 codebooks; 1.178B parameters, less than the
  10-layer depformer student): loaded through `load_policy`, logits [1, 8, 120, 2048] with each
  codebook's valid positions one fewer than the last (its delay), finite untrained losses (11.6);
  `LMGen` emits from the 8th step on, text and 8 codebooks a step, every token within the card.
- Tiny models on the CPU (dim 64, 4 codebooks; the review's script, rerun with the text token in
  the heads): no target codebook ≤ j of a frame moves when codebook j of that frame is perturbed
  (no leak), head j sees codebook j-1 of the same frame, and greedy `LMGen` output equals the
  teacher-forced argmax at every valid position, audio and text, for a heads and a depformer model.
- The trainer's forward pins the GPU's efficient attention and compiles: a CPU check runs eager
  with `lm_training.sdpa_kernel` replaced, and needs Hibiki's (empty) condition tensors.

## Next

- Train both students on the GPU once GRPO frees it: the end-to-end check of the cut, the heads
  and the codebook cut.
- Profile each student before its run (user, 2026-11-13): frames/s and peak memory at 4,608,
  8,192 and 12,288 frames per update (`--batch-frames`), each a 300-step run timed over steps
  151–300, as [sft-speed](sft-speed.md#measurements) measured the 3B. A ~10-layer student should
  fit 8,192 or more at once on 80 GB, so the slicing that cost the 3B 6.5% there does not apply;
  on the H200 the 3B ran 11% faster per frame at 8,192 and 17% at 12,288 than at 4,608. The
  trainer's peak, 2.25e-6, is set for 4,608 (Adam's square-root rule,
  [sft-training](sft-training.md#trainer)): another batch scales it by sqrt(frames / 4,608), and
  a cut model may need a higher one anyway.
- Speech evaluation in both directions: each target language's ASR transcript against the
  reference and speaker similarity to the source (evaluation is text-only today).
- Architecture changes wait for the trained student's scores (2026-11-13). The untrained cut
  already holds real time on the A15 (41 → 44.5 ms a frame, GPU-bound,
  [swift-runtime](swift-runtime.md#device-test)), so the budget has room; what costs quality is
  8 codebooks (Whisper's Vietnamese WER through Mimi 42.0% at 8 against 27.8% at 16) and the 10
  layers. In order, each only if the scores call for it:
  1. Vietnamese speech poor at 8 codebooks: a 16-codebook student with the same 3-layer
     depformer (`--codebooks 16`). Its steps 8–15 share one weight slice, so a frame reads ~4%
     more weights, but runs 24 more sequential passes (estimated +12 ms on the A15): time it on
     the phone first.
  2. The parallel heads ([Parallel heads](#parallel-heads)): no depformer passes, ~4 more main
     layers for the same time, +400 ms of heard speech lag; adopted only if they win.
  3. A grouped depformer (2 codebooks a step) if 16 codebooks are too slow and the heads lose.
  Two levers that change no layer:
  - Trim the text vocabulary to the pieces English and Vietnamese use: the text head reads all
    48,000 rows every frame (~0.1 GB at 8 bits, ~12% of a student frame's reads, estimated), and
    the three text tables are ~245M of the student's 1.39B parameters. Needs an id map through
    training, evaluation and both runtimes; count the pieces the two languages use first.
  - 1-frame steps again: 2-frame steps came in when the CPU also decoded Mimi; with decoding on
    the Neural Engine the CPU's encode (20–28 ms a frame) is under the GPU's step, so 1-frame
    steps may hold real time with up to 160 ms less latency. One phone run decides.
- The MLX and Swift runtimes for the student's config (the Swift app already times untrained cuts)
  and for the heads.
