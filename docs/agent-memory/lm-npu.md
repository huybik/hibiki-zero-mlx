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
                                                      or lm-depformer-k.mlmodelc, lm-depformer-embeddings.bin
                                                         lm-embeddings.bin   input tables, summed on the host
```

## Export

- Two audio heads (`audio_heads` in the config): the parallel-heads student's
  ([student-heads](student-heads.md)), one call a frame (user, 2026-11-16: "i will do parallel
  head then we can consider this"), and `depformer_parallel`, a depformer whose last slice emits
  the last `parallel_codebooks` at once, one call a slice ([Depformer slices](#depformer-slices)).
  The Swift runtime runs both only this way; its MLX LM has a plain depformer only, and
  `HibikiConfig.validate` refuses either without `lm_coreml`.
- `lm_step.py` reads the bf16 staging (MLX weight names) and holds the step as fixed-shape PyTorch,
  one frame per call. `Trunk`: x [1, 2048] → hidden, text logits. Its KV cache is Core ML state
  (iOS 18), a ring of `context` (3,000) slots per layer, fp16, 172 MB (sizes here in 10^6 bytes). Frame t writes slot t mod
  3,000 and attends to the frames t − 2,999 .. t, as moshi trains. The host computes the RoPE
  tables (angles in Double, rounded straight to fp16), the slot and the mask (`Inputs.at`, as
  `NeuralLM` does), so positions never enter fp16 arithmetic.
  `Heads`: hidden + `text_emb[sampled text]` → [16, 2048] logits, the text sampled on the host
  between the two calls, as `moshi_mlx`'s `ParallelHeads`. `DepformerSlice` k: hidden + the
  slice's embedding of the token before it (the text, else codebook k − 1) and the keys and values
  of the frame's slices before it → logits (one codebook, or `parallel_codebooks` for the last)
  and its own key and value, as `moshi_mlx`'s `DepFormerSlice._step`.
- The heads are 16 separate matmuls: one [2048 → 32,768] matmul ran on the CPU (compute plan), the
  16 run on the Neural Engine. The embedding gathers stay on the host (465 MB fp16, memory-mapped,
  one row from each of 33 tables a frame), since gathers are CPU ops in Core ML.
- Fed the MLX runtime's own inputs, `lm_step` in fp32 matches MLX bf16's argmaxes ([Precision](#precision)).
- `export_lm_coreml.py MODEL_DIR OUT_DIR --bits 6` writes a model directory: both models compiled,
  the tables, the tokenizer and Mimi checkpoint, and the config with `lm_coreml: {bits}` in place
  of `moshi_name`. Mimi's decoder comes from `export_mimi_coreml.py`
  ([README](../../inference/swift/README.md#core-ml-lm)). A `depformer_parallel` model gets
  `lm-depformer-{k}.mlmodelc` and the slices' input tables, `lm-depformer-embeddings.bin` (fp16,
  memory-mapped), in place of the heads. `--bits` palettizes weights over 65,536
  elements by k-means (every heads-student weight is over 1M; a 512-wide slice's are not, so the
  threshold was 1M until 2026-11-27), one table of 2^N entries per 16 output channels: trunk 1,958 → 738 MB, heads 134
  → 51 MB. It takes ~15 min on one core. Compute plan of the 6-bit models: all 892 trunk ops and
  all 18 head ops on the Neural Engine. A student trained against these tables is snapped onto
  them first ([student-palettize](student-palettize.md)).

## Runtime

- `FrameLM` (`Generator.swift`) is what the delayed streams drive: `HibikiLM` (MLX, GPU) or
  `NeuralLM` (Core ML). Each samples as `lm_gen_config` says, and `Translator.seed` seeds it.
- `NeuralLM.start` runs the frame on a serial queue and returns a wait, so the driving thread
  encodes and decodes Mimi meanwhile, as with the MLX LM on the GPU. The host part of a frame:
  - the 33 embedding rows summed in fp32 with Accelerate (scalar loops took 1.2 ms on the M4 Pro);
  - the text sampled on the CPU (`Sampler` on `[Float]`: exactly k kept, by quickselect, the first
    of any tied with the k-th, as MLX's argpartition keeps k; `SplitMix64`);
  - its `text_emb` row copied to the heads;
  - the 16 codebooks sampled;
  - or, for `depformer_parallel`, between the slices: each slice's table row copied in, its key
    and value copied into the later slices' inputs, its codebooks sampled (timed as `lmHeadsMs`
    for the slices' calls and `lmHostMs` for the rest).
  The same seed does not reproduce MLX's tokens: the generators differ.
- `Translator.step` encodes before it decodes: the Core ML LM holds the Neural Engine at the
  start of its frame, and a Mimi decode queued behind it left the CPU idle (M4 Pro: frame 26 →
  18.5 ms).
- `--lm-units ane|gpu|cpu|all` (CLI, app) places the Core ML LM; `ane` by default. `bench` and
  the app log its time per frame (`lmTrunkMs`, `lmHeadsMs`, `lmHostMs`).
- Beside the Core ML LM on the Neural Engine, Mimi encodes on the GPU by default, which the LM
  leaves idle: the frame holds flat where the CPU's encode slows it ([Speed](#speed); user,
  2026-11-21). Beside any other LM it encodes on the CPU; `--encoder cpu|gpu` (CLI, app) chooses
  ([swift-runtime](swift-runtime.md#mimi-on-the-gpu) for its kernels).

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
- **Agreement drifts over long contexts.** On a 4.5-minute clip (FLEURS dev VI rows joined, 3,412
  frames, the same replay), codebook 0 agreed 96–97% over frames 0–1,500, then fell to 92%, 84–85%
  and 79–80% by frames 3,000–3,412, in fp16 and at 6 bits alike. MLX 6 bits held 95–97% throughout.
  There is no step at the ring's wrap (frame 3,000), so this is the Neural Engine's fp16 summing
  over thousands of keys, as with Mimi's encoder ([mimi-npu](mimi-npu.md#precision)), not
  palettizing or the ring. The short-clip numbers and the gate do not see it. The model
  itself degenerates on long input anyway ([swift-runtime](swift-runtime.md#device-test)); the
  reference's text stopped after frame 2,250.
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

| Device | LM | Frame (paced, 1-frame steps) |
|---|---|---|
| M4 Pro, back to back (Python, `coremltools`) | 11.2 / 0.9 / – ms | |
| M4 Pro, `hibiki bench`, Mimi on the CPU | 13.1 / 0.9 / 1.2 ms | 18.5–19.0 ms |
| A15 from cool, Mimi on the GPU (the default), minutes 1–10 | 42.6–44.3 / 3.2–3.5 / 2.2–5.9 ms | 52.2–53.9 ms |
| A15 from cool, Mimi on the CPU, minutes 1–7 | 41.9–42.3 / 3.6–4.6 / 1.1–2.0 ms | 47.2–48.6 ms |
| A15 from cool, Mimi on the CPU, minutes 8–10 | 39.9–41.2 / 3.1–3.5 / 1.8–2.3 ms | 51.9, 55.3, 74.6 ms |
| A15 starting hot, Mimi on the CPU, minutes 3–10 | 42.9–43.1 / 3.6 / 1.8–2.7 ms | 73.7–84.6 ms |

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
- **On the CPU, Mimi's encode sets the margin** (`--encoder cpu`, from cool, 2026-11-20: thermal
  "serious" from 41 s, lag ≤ 0.36 s, footprint 653 MB). The frame is the larger of the LM and the
  driving thread's encode plus decode: 4–6 ms faster than on the GPU for 7 minutes, but the encode
  went 36.7 ms (minute 1) → 45 (8) → 64.4 (10), so minute 10 ran at 74.6 ms, near the 80 ms
  budget. Started hot (after earlier runs), it reached 63.5 ms at minute 3 and 73.7 by minute 10:
  the frame passed 80 ms in minute 6 and the lag reached 17.6 s.
- The CPU encode took 37 ms cold, against 20–28 beside the MLX LM on the GPU
  ([swift-runtime](swift-runtime.md#mimi-on-the-cpu)). The Neural Engine streams 738 MB of weights
  a frame (~17.5 GB/s), which likely slows the CPU's memory-bound encode.
- Run with the CPU's kernels and waited on before the decode, the GPU encode had taken 42 ms
  a frame and the frame 55–59 ms, then 70–74 from minute 7 (started hot); the fused kernels
  and the overlap are what took it off the critical path.
- 2-frame steps (`--chunk 2`, Mimi on the CPU, started hot) did worse: 45 ms a frame in minute 1, then 78–96 ms and
  14–20 s of lag. Per frame, the encode went 25 → 49–64 ms, and the decode rose to 30 ms: a step's
  two decodes queue behind the LM on the Neural Engine. The LM itself held 49 ms.
- fp16 weights: the trunk took 61.7 ms a frame on the M4 Pro's Neural Engine against 11.2 at 6
  bits. It streams its weights every frame, so their size sets its speed.
- Raw results: `/Volumes/data/hibiki-zero-evals/iphone/ane/` (`heads6_ane_gpuenc_compiled_cool_vi_10min`,
  the headline, Mimi on the GPU from cool; `heads6_ane_default_vi_1min`, the same with no
  `--encoder`, minute 1 alike (53.3 ms p50); `heads6_ane_gpuenc_fused_cool_vi_10min`, the same
  uncompiled; `heads6_ane_cool_vi_10min`, Mimi on the CPU from cool; started hot:
  `heads6_ane_vi_{1,10}min`, `heads6_ane_gpuenc_vi_10min` (the GPU with the CPU's kernels),
  `heads6_ane_chunk2_vi_10min`), each `.jsonl` (a sample a step), `.txt` (summary and text) and
  `.wav`.

## Depformer slices

- **Why:** the heads student speaks in a voice unlike the speaker's, from codebook 1 on, at every
  checkpoint and precision, where the depformer student keeps it
  ([voice-scores](voice-scores.md)). `depformer_parallel` (2026-11-27, not yet
  trained): a depformer for codebooks 0–7, one slice each, whose 9th slice reads codebook 7 and
  emits 8–15 at once (`parallel_codebooks: 8`), so the delays go back to the teacher's (max 2).
- **The math is moshi's:** `lm_step`'s slices against `moshi_mlx`'s `DepFormerSlice._step` on the
  depformer student (update 2,000 bf16, 8 slices, read as `parallel_codebooks: 1`), fp32, the same
  hidden states and tokens: logits within 5.4e-7 of the largest, argmax 32 of 32.
- **The Core ML path keeps the voice:** that student exported at 6 bits as `depformer_parallel`
  with `parallel_codebooks: 1` (`/Volumes/data/models/hibiki/weights/student_dep10/g2000_ane6`),
  through the Swift CLI on the M4 Pro's Neural Engine, on the gate's 5 FLEURS dev rows each way,
  against the same model in MLX 6 bits: VI→EN chrF 44.8 (49.6), `speaker_sim` 0.27 (0.27),
  `hnr_db` 4.4 (4.7); EN→VI 39.8 (40.8), 0.18 (0.19), 7.6 (7.8). Its LM: trunk 8.6, slices 5.9,
  host 1.2 ms a frame. The CLI feeds no direction primer, so on the user's mic clips the two-way
  model guessed EN→VI from the voice, where MLX primed with `speaking="vi"` translated to English.
- **Untrained sizes** (random slices on the heads release's trunk, 6 bits; the slices' tables fp16
  on the host):

  | Variant (dim × layers, 9 slices) | Slices | Tables | M4 Pro: slices / host ms a frame |
  |---|---|---|---|
  | heads release (for reference) | 51 MB heads | – | 1.2 / 1.1 |
  | 512 × 1 | 47 MB | 63 MB | 1.8 / 3.1 |
  | 512 × 2 | 74 MB | 63 MB | 2.6 / 2.4 |
  | 1024 × 1 | 147 MB | 126 MB | 4.1 / 1.9 |
  | 1024 × 3 (the depformer student's shape) | 365 MB | 126 MB | 6.3 / 3.1 |

  The trunk held 11.6–12.2 ms in each (`hibiki bench`, 1 minute, Mimi on the GPU). The trained
  depformer student (10-layer trunk, 8 slices of 1024 × 3): trunk 8.8, slices 5.9, host 0.9 ms,
  the LM 15.7 against the heads release's 14.8.
- **On the A15** (the app, 2 minutes each from the same Vietnamese clip, Mimi on the GPU, one after
  another, so warm; ms a frame, frame times from minute 2):

  | Variant | Trunk | Slices | Host | Frame p50 / p99 |
  |---|---|---|---|---|
  | heads release | 43.0 | 3.3 heads | 5.0 | 53.2 / 55.1 |
  | 512 × 1 | 42.4 | 16.0 | 7.8 | 66.1 / 69.5 |
  | 512 × 2 | 42.4 | 17.2 | 7.9 | 67.7 / 70.3 |
  | 1024 × 1 | 42.2 | 19.6 | 7.9 | 69.7 / 73.2 |
  | 1024 × 3 | 40.9 | 28.0 | 9.0 | 78.0 / 82.0 |

  The slices cost mostly their 9 calls: doubling the layers added 1.2 ms, where the phone takes
  ~1.8 ms a call even for 512 × 1 (the Mac 0.2). The 9 samplings add ~3 ms of host. Each reached
  thermal state 2 (fair) within the 2 minutes. 1024 × 3, the depformer student's shape, misses
  the frame's 80 ms (p90 80.0). **512 × 2 holds for 10 minutes** (run on its own after these):
  frame p50 66.8–68.8 ms every minute, p99 69.3–74.5 but 79.4 in minute 10, thermal 2 throughout,
  lag at most 0.09 s; LM 66.8 (trunk 42.9, slices 17.4, host 6.3). The trained depformer student
  was not timed: it trails the heads student by ~14 chrF ([phone-student](phone-student.md)), so
  a new student is trained either way. Raw results:
  `/Volumes/data/hibiki-zero-evals/iphone/depformer/` (`*_vi_2min.jsonl` and
  `d512x2_vi_10min.jsonl`, a sample a step, and `.txt`, the console), the Mac's in
  `/Volumes/data/hibiki-zero-evals/mac/depformer/`.

## Next

- Fewer Core ML calls for the slices, whose cost on the A15 is ~1.8 ms a call: a larger
  `parallel_codebooks` (fewer slices), or every slice in one model with the sampling inside it.
- A full-set gate of the Core ML LM against MLX 6 bits, in one runtime.
- Long sessions: the fp16 drift past ~2 minutes of context. Untried: attention over the cache in
  chunks with their partial sums combined (more fp16 headroom per sum), or a shorter context.
- The encode is the GPU's main load in Speech Live; MLX in fp16 would halve its bytes, with codes
  99.8 / 98 / 95% equal to fp32's ([mimi-npu](mimi-npu.md)), after a chrF gate.
- Not a lever: the LM's host part beside the GPU encode (5–6 ms). Its logit conversion and
  sampling take 0.7 ms a frame on the M4 Pro, the same with reused buffers and vImage (0.73
  against 0.67 ms; the conversion 0.02 → 0.006), so the CPU's clock sets it (2026-11-26).
- Fewer bytes on the Neural Engine may free bandwidth for the encode and the LM: 4-bit
  palettization (~490 MB), after a quality gate; training against the tables is built
  ([student-palettize](student-palettize.md)).
