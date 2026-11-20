# The LM on the Neural Engine

Read when touching `inference/tools/lm_step.py`, `inference/tools/export_lm_coreml.py` or
`inference/swift/Sources/HibikiCore/NeuralLM.swift`, or when running the phone LM on an NPU.
Mimi's own Core ML models are in [mimi-npu](mimi-npu.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Export](#export) · [Runtime](#runtime) ·
[Precision](#precision) · [Speed](#speed) · [Next](#next)

```text
convert_mlx.py (bf16) ─► lm_step.py (PyTorch, fp32) ─► export_lm_coreml.py --bits 6 ─► model dir ─► NeuralLM (Swift, ANE)
                                                         lm-trunk.mlmodelc   main transformer + text head, KV cache as state
                                                         lm-heads.mlmodelc   16 codebook heads
                                                         lm-embeddings.bin   input tables, summed on the host
```

## Export

- Only the parallel-heads student ([student-heads](student-heads.md)): its audio is one call a
  frame, where the depformer would take 8 sequential round trips (user, 2026-11-16: "i will do
  parallel head then we can consider this"). The Swift runtime runs the heads only this way; its
  MLX LM has a depformer only, and `HibikiConfig.validate` refuses a heads config without
  `lm_coreml`.
- `lm_step.py` reads the bf16 staging (MLX weight names) and holds the step as fixed-shape PyTorch,
  one frame per call. `Trunk`: x [1, 2048] → hidden, text logits. Its KV cache is Core ML state
  (iOS 18), a ring of `context` (3,000) slots per layer, fp16, 172 MB (sizes here in 10^6 bytes). Frame t writes slot t mod
  3,000 and attends to the frames t − 2,999 .. t, as moshi trains. The host computes the RoPE
  tables (angles in Double, rounded straight to fp16), the slot and the mask (`Inputs.at`, as
  `NeuralLM` does), so positions never enter fp16 arithmetic.
  `Heads`: hidden + `text_emb[sampled text]` → [16, 2048] logits, the text sampled on the host
  between the two calls, as `moshi_mlx`'s `ParallelHeads`.
- The heads are 16 separate matmuls: one [2048 → 32,768] matmul ran on the CPU (compute plan), the
  16 run on the Neural Engine. The embedding gathers stay on the host (465 MB fp16, memory-mapped,
  one row from each of 33 tables a frame), since gathers are CPU ops in Core ML.
- Fed the MLX runtime's own inputs, `lm_step` in fp32 matches MLX bf16's argmaxes ([Precision](#precision)).
- `export_lm_coreml.py MODEL_DIR OUT_DIR --bits 6` writes a model directory: both models compiled,
  the tables, the tokenizer and Mimi checkpoint, and the config with `lm_coreml: {bits}` in place
  of `moshi_name`. Mimi's decoder comes from `export_mimi_coreml.py`
  ([README](../../inference/swift/README.md#core-ml-lm)). `--bits` palettizes weights over 1M
  elements by k-means, one table of 2^N entries per 16 output channels: trunk 1,958 → 738 MB, heads 134
  → 51 MB. It takes ~15 min on one core. Compute plan of the 6-bit models: all 892 trunk ops and
  all 18 head ops on the Neural Engine.

## Runtime

- `FrameLM` (`Generator.swift`) is what the delayed streams drive: `HibikiLM` (MLX, GPU) or
  `NeuralLM` (Core ML). Each samples as `lm_gen_config` says, and `Translator.seed` seeds it.
- `NeuralLM.start` runs the frame on a serial queue and returns a wait, so the driving thread
  encodes and decodes Mimi meanwhile, as with the MLX LM on the GPU. The host part of a frame:
  - the 33 embedding rows summed in fp32 with Accelerate (scalar loops took 1.2 ms on the M4 Pro);
  - the text sampled on the CPU (`Sampler` on `[Float]`: exactly k kept, by quickselect, the first
    of any tied with the k-th, as MLX's argpartition keeps k; `SplitMix64`);
  - its `text_emb` row copied to the heads;
  - the 16 codebooks sampled.
  The same seed does not reproduce MLX's tokens: the generators differ.
- `Translator.step` encodes before it decodes: the Core ML LM holds the Neural Engine at the
  start of its frame, and a Mimi decode queued behind it left the CPU idle (M4 Pro: frame 26 →
  18.5 ms).
- `--lm-units ane|gpu|cpu|all` (CLI, app) places the Core ML LM; `ane` by default. `bench` and
  the app log its time per frame (`lmTrunkMs`, `lmHeadsMs`, `lmHostMs`).
- Tried and dropped: Mimi's encoder on the GPU beside the Core ML LM (2026-11-20, the A15,
  10 paced minutes). It took 42 ms a frame there, against 37 on the CPU, and the GPU's traffic
  slowed the LM (trunk 43 → 47 ms) and the decoder (11 → 15–17 ms). Hot, it reached 55–60 ms
  (p90 109) against the CPU's 64–74; the frame still went over 80 ms by minute 7.

## Precision

Teacher-forced on the MLX runtime's own steps: 695 frames of 4 FLEURS dev clips (2 each way,
`student_heads14_cb16` update 16,000, 2026-11-20, scratch scripts). Each column is the share of
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

`student_heads14_cb16` update 16,000, Core ML 6 bits on the Neural Engine, 2026-11-20. The LM's
own time per frame (trunk / heads / host) and the whole frame, medians:

| Device | LM | Frame (paced, 1-frame steps) |
|---|---|---|
| M4 Pro, back to back (Python, `coremltools`) | 11.2 / 0.9 / – ms | |
| M4 Pro, `hibiki bench` | 13.1 / 0.9 / 1.2 ms | 18.5–19.0 ms |
| A15, the app, minutes 1–2 | 42.3 / 3.8 / 1.1–2.0 ms | 47.7–48.3 ms |
| A15, minutes 3–10 | 42.9–43.1 / 3.6 / 1.8–2.7 ms | 73.7–84.6 ms |

- On the A15 (iPhone 13 Pro Max, wired, `clip-vi.wav` a FLEURS dev clip looped, thermal state
  "serious" from the start), the LM held 48–49 ms a frame for all 10 minutes, with no throttle step.
  On the GPU in MLX, a 14-layer depformer cut of the 3B (4 bits, 8 codebooks) had run 64 ms a
  frame in minute 1 and 94–98 from minute 3, GPU-bound
  ([swift-runtime](swift-runtime.md#device-test)). The footprint was 652 MB, flat.
- The frame is the larger of the LM and the driving thread's Mimi encode (CPU) plus decode
  (Neural Engine). Encode / decode went 36.8 / 11.4 ms in minute 1; at minute 3 the CPU throttled
  and the encode jumped to 63.5 ms, 73.7 by minute 10. The frame passed 80 ms in minute 6, and the
  lag reached 17.6 s at minute 10. The LM fits real time; Mimi's CPU encode at 1-frame steps does
  not, once hot.
- 2-frame steps (`--chunk 2`) did worse: 45 ms a frame in minute 1, then 78–96 ms and 14–20 s of
  lag. Per frame, the encode went 25 → 49–64 ms, and the decode rose to 30 ms: a step's two
  decodes queue behind the LM on the Neural Engine. The LM itself held 49 ms.
- Every A15 run here began "serious" from the one before; the CPU encode at 1-frame steps took
  37 ms cold, against 20–28 beside the MLX LM on the GPU ([swift-runtime](swift-runtime.md#mimi-on-the-cpu)).
  The Neural Engine streams 738 MB of weights a frame (~17.5 GB/s), which likely slows the CPU's
  memory-bound encode.
- fp16 weights: the trunk took 61.7 ms a frame on the M4 Pro's Neural Engine against 11.2 at 6
  bits. It streams its weights every frame, so their size sets its speed.
- Raw results: `/Volumes/data/hibiki-zero-evals/iphone/ane/` (`heads6_ane_vi_{1,10}min`,
  `heads6_ane_gpuenc_vi_10min`, `heads6_ane_chunk2_vi_10min`), each `.jsonl` (a sample a step), `.txt` (summary and text) and
  `.wav`.

## Next

- A full-set gate of the Core ML LM against MLX 6 bits, in one runtime.
- Long sessions: the fp16 drift past ~2 minutes of context. Untried: attention over the cache in
  chunks with their partial sums combined (more fp16 headroom per sum), or a shorter context.
- Mimi's CPU encode is now the phone's bottleneck. Its fp16 Neural Engine codes drift
  ([mimi-npu](mimi-npu.md#precision)), so it stays on the CPU. Fewer bytes on the Neural Engine
  may free bandwidth for it: 4-bit palettization (~490 MB), after a quality gate.
