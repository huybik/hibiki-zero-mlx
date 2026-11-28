# The LM on the Neural Engine

Read when touching `inference/tools/lm_step.py`, `inference/tools/export_lm_coreml.py` or
`inference/swift/Sources/HibikiCore/NeuralLM.swift`, or when running the phone LM on an NPU.
Mimi's own Core ML models are in [mimi-npu](mimi-npu.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Export](#export) · [Runtime](#runtime) ·
[Precision](#precision) · [Speed](#speed) · [Depformer slices](#depformer-slices) · [Next](#next)

```text
convert_mlx.py (bf16) ─► lm_step.py (PyTorch, fp32) ─► export_lm_coreml.py --bits 6 ─► model dir ─► NeuralLM (Swift, ANE)
                                                         lm-trunk.mlmodelc   main transformer + text head, KV cache as state
                                                         lm-heads.mlmodelc   16 codebook heads
                                                      or lm-depformer.mlmodelc  every depformer slice + top-k sampling
                                                         lm-depformer-embeddings.bin, lm-embeddings.bin
```

## Export

- Two audio heads (`audio_heads` in the config), one call a frame each: the parallel-heads
  student's ([student-heads](student-heads.md); user, 2026-11-16: "i will do parallel head then
  we can consider this"), and `depformer_parallel`, a depformer whose last slice emits the last
  `parallel_codebooks` at once ([Depformer slices](#depformer-slices)). The Swift runtime runs
  only these Core ML LMs: `HibikiConfig.validate` refuses a directory without `lm_coreml`.
- `lm_step.py` reads the bf16 staging (MLX weight names) and holds the step as fixed-shape PyTorch,
  one frame per call. `Trunk`: x [1, 2048] → hidden, text logits. Its KV cache is Core ML state
  (iOS 18), a ring of `context` (3,000) slots per layer, fp16, 172 MB (sizes here in 10^6 bytes). Frame t writes slot t mod
  3,000 and attends to the frames t − 2,999 .. t, as moshi trains. The host computes the RoPE
  tables (angles in Double, rounded straight to fp16), the slot and the mask (`Inputs.at`, as
  `NeuralLM` does), so positions never enter fp16 arithmetic.
  `Heads`: hidden + `text_emb[sampled text]` → [16, 2048] logits, the text sampled on the host
  between the two calls, as `moshi_mlx`'s `ParallelHeads`. `DepformerStep`: hidden + slice 0's
  embedding of the sampled text + Gumbel noise [16, 2048] → the 16 codebooks, every slice in one
  graph, each sampling its codebook and embedding it for the next, as `moshi_mlx`'s
  `DepFormer.sample` ([Depformer slices](#depformer-slices)).
- The heads are 16 separate matmuls: one [2048 → 32,768] matmul ran on the CPU (compute plan), the
  16 run on the Neural Engine. The embedding gathers stay on the host (465 MB fp16, memory-mapped,
  one row from each of 33 tables a frame), since gathers are CPU ops in Core ML.
- Fed the MLX runtime's own inputs, `lm_step` in fp32 matches MLX bf16's argmaxes ([Precision](#precision)).
- `export_lm_coreml.py MODEL_DIR OUT_DIR --bits 6` writes a model directory: both models compiled,
  the tables, the tokenizer and Mimi checkpoint, and the config with `lm_coreml: {bits}` in place
  of `moshi_name`. Mimi's decoder comes from `export_mimi_coreml.py`
  ([README](../../inference/swift/README.md#core-ml-lm)). A `depformer_parallel` model exports
  `lm-depformer.mlmodelc` (`DepformerStep`, `depformer_top_k` recorded in `lm_coreml`) and slice
  0's table in `lm-depformer-embeddings.bin`. `--bits` (4 or 6) palettizes weights over
  65,536 elements by k-means (every heads-student weight is over 1M; a 512-wide slice's are not, so
  the threshold was 1M until 2026-11-27), one table of 2^N entries per 16 output channels: trunk
  1,958 → 738 MB (491 MB at 4 bits), heads 134 → 51 MB. It takes ~15 min on one core. Compute plan
  of the 6-bit models: all trunk ops (2,754 with chunked attention), all 18 head ops, and all
  fused depformer ops (757 for 9 slices of 512 × 2; 952 for 8 slices of 1024 × 3) run on the Neural
  Engine with 0 CPU ops. A student trained against these tables is snapped onto them first
  ([student-palettize](student-palettize.md)).

## Runtime

- `NeuralLM` is what the delayed streams (`Generator.swift`) drive. It samples as
  `lm_gen_config` says, and `Translator.seed` seeds it.
- `NeuralLM.start` runs the frame on a serial queue and returns a wait, so the driving thread
  encodes and decodes Mimi meanwhile. The host part of a frame:
  - the 33 embedding rows summed in fp32 with Accelerate (scalar loops took 1.2 ms on the M4 Pro);
  - the text sampled on the CPU (`Sampler` on `[Float]`: exactly k kept, by quickselect, the first
    of any tied with the k-th, as MLX's argpartition keeps k; `SplitMix64`);
  - its `text_emb` row copied to the heads, and the 16 codebooks sampled;
  - or, for `depformer_parallel`, slice 0's table row copied in and the Gumbel noise filled from
    a 65,536-entry quantile table, pre-scaled by the temperature (4 lookups per `SplitMix64`
    call; zeros at temperature 0); the depformer's call is timed as `lmHeadsMs`.
  The same seed does not reproduce MLX's tokens: the generators differ.
- `Translator.step` encodes before it decodes: the Core ML LM holds the Neural Engine at the
  start of its frame, and a Mimi decode queued behind it left the CPU idle (M4 Pro: frame 26 →
  18.5 ms).
- `--lm-units ane|gpu|cpu|all` (CLI, app) places the Core ML LM; `ane` by default. `bench` and
  the app log its time per frame (`lmTrunkMs`, `lmHeadsMs`, `lmHostMs`).
- Mimi encodes on the GPU, which the LM leaves idle: the frame holds flat where the CPU's encode
  slows it ([Speed](#speed); user, 2026-11-21;
  [swift-runtime](swift-runtime.md#mimis-encoder) for its kernels).

## Precision

Teacher-forced on the MLX runtime's own steps: 695 frames of 4 FLEURS dev clips (2 each way,
`student_heads14_cb16` update 16,000, 2026-11-20; rebuilt 2026-11-25 as
`/Volumes/data/hibiki-zero-evals/mac/npu-agree/npu_agree.py`, with a model trained against the
6-bit tables: [student-palettize](student-palettize.md#neural-engine-agreement)). Each column is the share of
argmaxes equal to the MLX bf16 run's, given the same inputs and sampled text:

| Model | Text | Codebook 0 | 1–7 | 8–15 | All 16 |
|---|---|---|---|---|---|
| `lm_step`, PyTorch fp32 | 99.7% | 99.0% | 98.1% | 98.1% | 98.1% |
| MLX 6 bits (`convert_mlx.py --bits 6`, what ships) | 99.4% | 97.4% | 95.4% | 96.4% | 96.0% |
| Core ML fp16, M4 Pro Neural Engine | 99.3% | 97.0% | 93.7% | 95.1% | 94.6% |
| Core ML 6 bits (palettized), M4 Pro Neural Engine | 98.7% | 95.4% | 90.8% | 92.8% | 92.1% |

- The Neural Engine's fp16 costs about what MLX's 6 bits do; palettizing costs as much again.
- **Agreement drifts over long contexts — fixed by chunked online-softmax attention.** On a
  4.5-minute clip (FLEURS dev VI rows joined, 3,412–3,670 frames, the same replay), flat fp16
  attention over all 3,000 KV slots agreed on codebook 0 for 97–98% over frames 0–1,500, then fell
  to 95%, 87%, 81%, 83–84% over frames 1,500–3,670 (`z1200_ane16`; `student_heads14_cb16` s16k
  fell 96–97% → 79–80% in fp16 and 6 bits alike, while MLX 6 bits held 95–97%). There is no step
  at the ring's wrap (frame 3,000), so the loss was fp16 summing 3,000 keys in one accumulator.
  Splitting KV attention into 12 chunks of 250 keys (`_ATTN_CHUNK = 250` via `.split(250)`, since
  Core ML `MLState` rejects 4-D reshapes with error `-14`) and combining them with online-softmax
  log-sum-exp weights keeps all 2,754 trunk ops on the Neural Engine at **zero 6-bit latency cost**
  (M4 Pro 6-bit trunk 11.31 vs 11.55 ms) and cuts the long-context drift sharply on `z1200_ane16`
  (2026-11-27, `/Volumes/data/hibiki-zero-evals/mac/long-context/`):
  - frames 0–1,500: cb0 97.6–98.0% (vs 97.0–98.0%), All 16 95.0–96.5% (vs 94.0–94.8%);
  - frames 2,000–3,670: cb0 **91.4 / 89.2 / 91.4 / 95.3%** (vs 86.6 / 81.2 / 82.6 / 84.1%, **+4.8
    to +11.2 pts**), All 16 **93.0 / 92.2 / 94.6 / 94.9%** (vs 90.8 / 90.2 / 93.2 / 92.0%).
  Range is no issue: the student's activations peak at 50.5, 1,300x under fp16's limit (measured
  on the depformer student, 2026-11-16).
- The gate's first 20 FLEURS dev rows each way through the Swift CLI (`hibiki translate`, seed 42
  plus the row, the M4 Pro; `/Volumes/data/hibiki-zero-evals/mac/gate/heads_cb16_s16k_coreml6_ane_swift_20rows/`,
  scratch script; the CLI reports no end offset). chrF / BLEU / length ratio / EOS, health pass on
  both:

  | Set | Core ML 6 bits, Swift | MLX 6 bits, `gate.py` ([student-heads](student-heads.md#inference)) |
  |---|---|---|
  | fleurs-vi/dev (20) | 28.09 / 1.97 / 1.44 / 20 of 20 | 30.40 / 2.04 / 1.72 / 15 of 20 |
  | fleurs-en/dev (20) | 25.30 / 7.64 / 1.02 / 16 of 20 | 25.79 / 6.95 / 1.02 / 17 of 20 |

  Sampled with other generators and another Mimi encoder (MLX Swift against rustymimi), so the
  rows differ token by token; a full-set gate, scored on the same runtime, decides.

## Speed

`student_heads14_cb16` update 16,000, Core ML 6 bits on the Neural Engine, 2026-11-20 and 21. The
LM's own time per frame (trunk / heads / host) and the whole frame, medians:

| Device | LM | Frame (paced) |
|---|---|---|
| M4 Pro, back to back (Python, `coremltools`) | 11.2 / 0.9 / – ms | |
| A15 from cool, minutes 1–10 | 42.6–44.3 / 3.2–3.5 / 2.2–5.9 ms | 52.2–53.9 ms |

- **From cool it holds the frame flat for 10 minutes** (the A15, iPhone 13 Pro Max, wired; a
  FLEURS dev clip looped, `--model heads6 --clip clip-vi.wav`; Mimi on the GPU, 2026-11-21):
  52.2–53.9 ms a frame every minute, p90 ≤ 55.3 after minute 1, lag ≤ 0.08 s, thermal "serious"
  from 324 s, footprint 541 MB. The LM held 50.6–52.2 ms a frame, with no throttle step; on the GPU in
  MLX, a 14-layer depformer cut of the 3B (4 bits, 8 codebooks) had run 64 ms a frame in minute 1
  and 94–98 from minute 3, GPU-bound ([swift-runtime](swift-runtime.md#device-test)). The trained
  6-bit release ran the same ([student-palettize](student-palettize.md#release-and-phone)).
- **On the GPU the encode is off the frame's critical path**: the driving thread builds and submits
  it compiled (9.5–10.6 ms; 11.8 uncompiled), waits on the decoder while the GPU encodes
  (42.7–45.6 ms: the LM, then the decode, on the Neural Engine) and finds the codes ready (encode
  wait 0.0). The frame is the LM plus the decode, and the GPU costs the LM 3–5 ms: its host part
  1.2–2.0 → 5.0–5.9 ms (2.2–2.8 in minutes 5–6; the CPU sits idle while the GPU works, likely its
  clock drops) and the trunk 42.3 → 43.0. 2-frame steps would shrink only the encode, so they were
  not run on the GPU.
- **In Speech Live the screen shares the GPU with the encode** (iPhone 13 Pro Max, `ab-vi2en.wav`,
  2026-11-26, [onevoice PR #9](https://github.com/anthoai97/onevoice-ai-challenge/pull/9)). Liquid
  Glass under moving views (the wave strip, the orb, a card growing per word) had the render server
  (`backboardd`) take 192 ms of GPU a second, beside the app's own 357–377 (the encode, mostly);
  flat panels cut it to 23 (Metal System Trace, 10 s each). From cool the frame stays 52 ms, set by
  the Neural Engine. Hot, the encode outgrows the decode's window: three sessions back to back on
  the glass UI ran 51 → 54 → 66 ms p50, the third with encode wait p90 69 ms and 7.2 s of lag; on
  flat panels 52 → 54 → 56 ms, encode wait 0, lag 0.1 s. The UI split also took the main thread
  27–29% → 17–21% of a core. Raw: `/Volumes/data/hibiki-zero-evals/iphone/speechlive-ui/`.
- **Mimi on the CPU (a removed option) set the margin** (from cool, 2026-11-20): 4–6 ms faster
  than the GPU for 7 minutes, but its encode went 36.7 ms (minute 1) → 64.4 (10), so minute 10 ran
  at 74.6 ms; started hot, the frame passed 80 ms in minute 6 and the lag reached 17.6 s. The
  Neural Engine streams 738 MB of weights a frame (~17.5 GB/s), which likely slows the CPU's
  memory-bound encode.
- Run with the CPU's kernels and waited on before the decode, the GPU encode had taken 42 ms
  a frame and the frame 55–59 ms, then 70–74 from minute 7 (started hot); the fused kernels
  and the overlap are what took it off the critical path.
- 2-frame steps (a removed option; Mimi on the CPU, started hot) did worse: 45 ms a frame in
  minute 1, then 78–96 ms and 14–20 s of lag. Per frame, the encode went 25 → 49–64 ms, and the
  decode rose to 30 ms: a step's two decodes queue behind the LM on the Neural Engine. The LM
  itself held 49 ms.
- fp16 weights: the trunk took 61.7 ms a frame on the M4 Pro's Neural Engine against 11.2 at 6
  bits. It streams its weights every frame, so their size sets its speed.
- Raw results: `/Volumes/data/hibiki-zero-evals/iphone/ane/` (`heads6_ane_gpuenc_compiled_cool_vi_10min`,
  the headline, Mimi on the GPU from cool; `heads6_ane_default_vi_1min`, the same with the
  default options, minute 1 alike (53.3 ms p50); `heads6_ane_gpuenc_fused_cool_vi_10min`, the same
  uncompiled; `heads6_ane_cool_vi_10min`, Mimi on the CPU from cool; started hot:
  `heads6_ane_vi_{1,10}min`, `heads6_ane_gpuenc_vi_10min` (the GPU with the CPU's kernels),
  `heads6_ane_chunk2_vi_10min`), each `.jsonl` (a sample a step), `.txt` (summary and text) and
  `.wav`.

## Depformer slices

- **Why:** the heads student speaks in a voice unlike the speaker's, from codebook 1 on, at every
  checkpoint and precision, where the depformer student keeps it
  ([voice-scores](voice-scores.md)). `depformer_parallel`: a depformer for codebooks 0–7, one
  slice each, whose 9th slice reads codebook 7 and emits 8–15 at once (`parallel_codebooks: 8`;
  moshi's step schedule, codebook j → slice min(j, 8)), so the delays go back to the teacher's
  (max 2). Trained as the slices student ([student-slices](student-slices.md)), shipped as
  `release/student_slices_coreml6` ([slices-phone](slices-phone.md)).
- **One Core ML model holds every slice** (`DepformerStep`, `lm-depformer.mlmodelc`): top-k
  Gumbel-max sampling and the embedding of each sampled token for the next slice run inside the
  graph. `ios18.topk`, `greater_equal`, `select` and `reduce_max` stay on the Neural Engine,
  where `reduce_argmax` and `gather` split the graph onto the CPU; exact fp16 integers `1..card`
  (exact for `card <= 2048`) break ties into a one-hot vector, `oh @ emb.T` embeds it, and the
  tokens come out as exact integers in fp16. All ops run on the Neural Engine: 757 for 9 slices of
  512 × 2 (80 MB at 6 bits, slice 0's host table 47 MB), 952 for 8 slices of 1024 × 3 (325 MB).
- **The math is moshi's:** `lm_step`'s slices, one at a time, against `moshi_mlx`'s
  `DepFormerSlice._step` on the depformer student (update 2,000 bf16, 8 slices, read as
  `parallel_codebooks: 1`), fp32, the same hidden states and tokens: logits within 5.4e-7 of the
  largest, argmax 32 of 32; `DepformerStep` greedy matched those slices on 80 of 80 codebooks.
- **The Core ML path keeps the voice:** that student exported at 6 bits
  (`/Volumes/data/models/hibiki/weights/student_dep10/g2000_ane6`, one Core ML model a slice),
  through the Swift CLI on the M4 Pro's Neural Engine, on the gate's 5 FLEURS dev rows each way,
  against the same model in MLX 6 bits: VI→EN chrF 44.8 (49.6), `speaker_sim` 0.27 (0.27),
  `hnr_db` 4.4 (4.7); EN→VI 39.8 (40.8), 0.18 (0.19), 7.6 (7.8). In one model, on the same 5×2
  rows in another run: VI→EN 38.61 chrF / sim 0.29 / HNR 5.7 dB, EN→VI 47.27 / 0.12 / 7.4, health
  pass (one model a slice: 40.65 / 0.30 / 5.3 and 36.28 / 0.20 / 8.4). The CLI feeds no direction
  primer, so on the user's mic clips the two-way model guessed EN→VI from the voice, where MLX
  primed with `speaking="vi"` translated to English.
- **One call, not one a slice:** a Core ML call costs the A15 ~1.8 ms even for a 512 × 1 slice
  (the Mac 0.2), so 9 slices in 9 models took 16–28 ms a frame (frame p50 66–78 ms; 1024 × 3
  missed the 80 ms frame), mostly in calls: doubling the layers added 1.2 ms. That export is
  removed. On the M4 Pro (`hibiki bench`) one model cut 512 × 2's frame 17.58 → 14.72 ms p50 (LM
  17.54 → 14.48, slices 2.65 → 1.80, host 2.50 → 1.35; the heads' LM 16.06) and 1024 × 3's
  15.83 → 14.63 (slices 5.86 → 3.62). Raw results: `/Volumes/data/hibiki-zero-evals/iphone/depformer/`,
  the Mac's in `/Volumes/data/hibiki-zero-evals/mac/depformer/`.
- The trained depformer student (10-layer trunk, 8 slices of 1024 × 3) was not timed on the A15:
  plain SFT left it ~14 chrF behind the 3B teacher
  ([phone-student](phone-student.md#distillation)), so a new student was trained either way.

## Next

- Time `lm-depformer.mlmodelc` (`d512x2_fused`, staged in `Documents/d512x2_fused/`) on the A15
  in `HibikiBench`; the slices release's LM took 58–60 ms a frame in Speech Live
  ([slices-phone](slices-phone.md#speech-live)).
- A full-set gate of the Core ML LM against MLX 6 bits, in one runtime.
- Not a lever: Mimi GPU encode in fp16 or mixed fp16 failed the 20×2 FLEURS dev gate
  ([swift-runtime](swift-runtime.md#mimis-encoder)); the LM's host part beside the GPU encode
  (5–6 ms; CPU clock bound, 2026-11-26).
- 4-bit trunk (~491 MB) needs QAT (`--palettize-bits 4`): plain 4-bit PTQ lost −4.1 chrF and
  −9.6% All-16 NPU agreement ([student-palettize](student-palettize.md#neural-engine-agreement)).
