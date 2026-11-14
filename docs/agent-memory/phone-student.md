# Phone student

Read when touching `training/student/`, `streams.model_rows` or a `StreamDataset`'s `codebooks`, or
when cutting, training, distilling or scoring a smaller model for the phone or its parallel heads.
Back to [CONTEXT.md](../../CONTEXT.md) · [Cut](#cut) · [Training a student](#training-a-student) ·
[Parallel heads](#parallel-heads) · [Distillation](#distillation) · [Checks](#checks) · [Next](#next)

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

## Distillation

The plan distils only if the student falls short of the 3B
([plan](../vi_en_parity_plan.md#5-the-phone-model), stage 5.3). Trained on the targets alone, the
depformer student does, by ~14 chrF, and gains slowly: FLEURS dev VI→EN / EN→VI chrF 28.7 / 24.5
at update 96,000, +1.6 / +0.7 over 64,000 ([Next](#next) has its validations), against the
teacher's 42.7 / 38.8; the end offset 4–8 s against 1.1–1.5 s. Teacher-forced on the VI→EN
validation cache, its word accuracy is 0.569 at 96,000 (0.544 at 64,000) against the two-way SFT
model's 0.664.

- `train.py --teacher-weight` ([sft-training](sft-training.md#objective)): each training token's
  loss is `common.token_kl`, KL(teacher ‖ student) of its next-token distribution, in place of
  its CE, under the same masks and weights (text PAD 0.5, codebook 0 100). The student learns
  the teacher's whole distribution at every position, and with GRPO update 1,400 as the teacher
  its timing too: where that model would write a word before the SFT target's, its distribution
  says so.
- The teacher (`--teacher-config`, default the 3B's) loads frozen in bf16 (`load_lm`, its own
  EOS row) and runs the compiled training forward without a backward, its depformer cut to the
  student's k steps (`dep_q`: the depformer is causal over steps, so the first k logits are the
  same; ~11% of a step saved). It reads the whole stream: the dataset is assembled at the cache's
  16 codebooks and each batch is cut to the student's rows (`model_rows`) on the GPU. Both must
  be on the same tokenizer. A token is scored where both models' masks hold: moshi returns each
  model's logits on the stream's frames, so a heads student (codebook j delayed j frames, the
  teacher's 2) loses only the last frames one of the two cannot score. Text and codebook 0 read
  the same frames in both; for codebooks 1–2 the teacher has read up to 2 frames more of the
  stream, for 4 and up the student has, so there it learns a blurrier distribution than it
  could (weight 1 each, against codebook 0's 100).
- Validation is unchanged (the student's CE against the targets on validation's fixed streams,
  free-running chrF), so a distilled run's teacher-forced validation compares with
  `student_dep10`'s exactly; its chrF is sampled at its own `--seed` (48 against 47), within the
  ~±0.7 a set moves between evaluations. Its train log's losses are KL values, not CE.
- A resume needs the start model and the teacher in `weights/`, neither on the run's Hub prefix:
  `student_dep10`'s pair at the stop (its `checkpoints/` on the Hub) and the recovery repo's
  `grpo_twoway/best/best_step001400.safetensors`, besides the student dir ([Training a
  student](#training-a-student)).
- `token_kl` (eager, bf16 logits with masked positions NaN-filled as moshi fills them) equals
  `torch.nn.functional.kl_div` on the valid positions, is 0 elsewhere and gives them no gradient.
  Tiny models on the CPU (4 of 6 codebooks): a heads student under its depformer parent is
  scored on exactly the positions both masks hold, its audio loss equals `kl_div`'s weighted
  mean there, and every parameter gets a finite gradient; a depformer student's positions and
  loss are as before.
- **End to end on the H100** (2026-11-14, the launcher's arguments, two FLEURS dev sets, from
  `student_dep10`'s pair 100,000): 0.381 s a step (plain SFT 0.23), 59.3 GB peak in training,
  generation 38 GB. In 300 updates the KL fell 0.45 → 0.37, the VI→EN cache's word accuracy rose
  0.569 → 0.607 (more than `student_dep10` gained from 48,000 to 96,000), FLEURS chrF 28.7 →
  30.6 VI→EN and 24.5 → 25.9 EN→VI, and the end offset fell from ~6 s to 1.8 / 2.3 s: the
  teacher's timing comes with its distribution. A resume at 150 matched the uninterrupted run
  within 1.2e-4 (two fresh runs differ by 8e-4); plain SFT matched the pre-PR trainer within
  9e-5; the 3B distilled into itself gives a KL of 5e-4 (bf16 against fp32 masters).
- **The run** (2026-11-14, user: "at this rate we never get 40+ chrf"): `restart.sh
  student-distill`, `student_dep10_distill`, from `student_dep10`'s newest model (stopped for it),
  on new streams of the same rows (seed 48), whole 12,288-frame batches at peak 3e-5 (where
  `student_dep10`'s cosine had come down to), validation every 8,000, 184,000 updates (~1.9
  epochs; 0.405 s an update with saves and validations, ~21 h, leaving the box ~14 h for a GRPO
  run on it, its test sets and publishing). The heads student distils from the start
  ([Next](#next)).

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

- **Training** (2026-11-14): `restart.sh student`, then `restart.sh student-heads`
  ([tranche-box](tranche-box.md)), each cut from GRPO update 1,400 when its directory is missing:
  2 epochs of the two-way run's rows both ways (190,126 updates, ~12 h plain, ~21 h distilled), the same streams
  (seed 47) for the same steps, whole 12,288-frame batches at peak 5e-5, validation every 16,000
  updates on the two-way run's 7 sets (FLEURS dev both ways selects). The depformer student first:
  it starts from the 3B's own depformer, both runtimes already run it, and the heads are the
  experiment, adopted only if they win. `student_dep10` ran at commit 09ef157; stopped at ~9,200
  for the direction trial ([two-way](two-way.md#next)), it resumed 2026-11-14 07:22 UTC from its
  pair 8,000, and stopped for good at ~100,150 (2026-11-14 20:28 UTC+7) for
  [distillation](#distillation), which starts from its pair 100,000 (on the Hub under
  `student_dep10/checkpoints/`; its best, 96,000, is the promoted one). Its validations (FLEURS
  dev chrF VI→EN / EN→VI; end offset 4–8 s throughout): 20.7 / 15.0 at 16,000, 21.6 / 18.6 at
  32,000, 24.5 / 18.6 at 48,000, 27.0 / 23.8 at 64,000, 26.2 / 23.2 at 80,000, 28.7 / 24.5 at
  96,000.
- **The heads student distils from the start** (user, 2026-11-14: "why we not do distillation
  from the start?"): plain SFT left the depformer student ~14 chrF short after 100,000 updates,
  and 300 distilled updates gained more than its last 32,000. It runs on this box only if
  `student_dep10_distill` plateaus early enough to leave ~14 h for it and ~6 h for the test sets,
  publishing and margin before the box ends (2026-11-16 08:00 UTC+7); otherwise it is the next
  box's first job. Profile it first: a 14-layer student and the teacher at 12,288 frames.
- **Profiled** (2026-11-14, the 700 W H100 SXM, the depformer student, the two-way caches, 300
  steps each, steps 151–300 by their real stream frames), each size one whole batch:

  | Frames an update | s a step | Frames/s |
  |---|---|---|
  | 4,608 | 0.106 | 38,700 |
  | 8,192 | 0.160 | 47,800 (+23%) |
  | 12,288 | 0.230 | 51,000 (+32%) |

  The 3B ran 15,300 at 4,608. The heads student steps ~5% faster at 4,608 (0.101 s).
- **Peak rate** (2026-11-14, 1,500 updates a rate at 12,288 frames, 5% warmup then cosine,
  validation on both directions' caches; `/workspace/students/lr/` on the box). Teacher-forced
  validation loss, VI→EN / EN→VI:

  | Peak | Depformer student | Heads student |
  |---|---|---|
  | 3.7e-6 (the trainer's default scaled to the batch) | 4.65 / 4.45 | |
  | 1e-5 | 4.30 / 4.09 | |
  | 3e-5 | 4.06 / 3.84 | 4.55 / 4.23 |
  | 1e-4 | 4.03 / 3.76 | 4.48 / 4.20 |
  | 3e-4 | 4.50 / 4.11 | 5.10 / 4.65 |

  Both peak near 1e-4 at this length, with 3e-5 close and 3e-4 already worse. A run 127x longer
  wants less than the short run's best and sits 3x from where it breaks down, so both train at
  5e-5, the same for the comparison. Free-running chrF at 1,500 updates (10–17, most sets failing
  the empty-row gate) is too early to rank them; the heads student leaves fewer rows empty.
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
- Both runtimes run the depformer student's config as it is (2026-11-14, the untrained cut at 6
  bits, 1.24 GB): MLX file mode, and the Swift CLI with an 8-codebook `mimi-decoder.mlmodelc`
  (`export_mimi_coreml.py --decoder-codebooks 8`), 107 frames in 2.0 s on the M4 Pro (Mimi's
  encode 12.8 ms a frame, the LM 2.9, the decode 3.3). The heads need support in both runtimes,
  built only if they win.
- **6 bits holds on the student** (2026-11-14, `student_dep10` update 48,000 staged by
  `convert_mlx.py`, the gate on every FLEURS dev row, `--seed 42`; bf16 minus 6 bits, paired
  bootstrap 95%): VI→EN 24.81 / 24.43 (+0.38, −0.41 to +1.24), EN→VI 19.34 / 19.39 (−0.05, −0.82
  to +0.72), as on the 3B. The MLX bf16 scores match the trainer's own validation at that update
  (24.46 / 18.62, 48 empty EN→VI rows against MLX's 50), so the runtime reads the student
  faithfully. 1.24 GB at 6 bits, 2.79 in bf16.
