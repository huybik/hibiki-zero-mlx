# Mimi on neural accelerators

Read when touching `inference/tools/mimi_step.py`, `inference/tools/export_mimi_coreml.py` or
`inference/swift/Sources/HibikiCore/NeuralDecoder.swift`, or when moving Mimi (or later the LM)
onto an NPU: the Apple Neural Engine now, Qualcomm's later.
Back to [CONTEXT.md](../../CONTEXT.md) · [Step module](#step-module) · [Core ML](#core-ml) ·
[Precision](#precision) · [Qualcomm](#qualcomm)

```text
mimi_step.py (PyTorch, fp32 exact) ─► export_mimi_coreml.py ─► mimi-decoder.mlpackage ─► NeuralDecoder (Swift, ANE)
                                   │                        └► mimi-encoder.mlpackage (not used: fp16 codes drift)
                                   └► later: ONNX ─► QNN (Qualcomm HTP)
```

## Step module

- `mimi_step.py` holds Mimi's streaming encoder and decoder steps as pure functions of explicit
  state, k frames per call fixed when built. The state is three tensors the host feeds back
  unchanged: `conv` (every convolution's left context and every transposed convolution's
  overlap, packed) and `keys` / `values` (each transformer layer's last 249 steps). The host
  computes the attention mask, the RoPE tables and the first-call flag from the step count
  (`Inputs.at`), with positions in float64, so they never enter a half-precision graph.
- Every op is a convolution, matmul, layer norm, softmax or elementwise op on fixed 4-D or
  smaller shapes: the query and key features of each head are reordered from RoPE's
  interleaved pairs to two halves through `in_proj`'s rows, which leaves attention unchanged.
- Exact in fp32: at 2 frames per call its codes equal the training caches' encoder
  (`training/prep/mimi_mlx.py`, whole clip) on all 16 codebooks over 400 frames of `leon.wav`,
  and its audio is within 1.3e-5 (relative RMS) of moshi's whole-sequence decode.
- moshi 0.2.13's own streaming Mimi is no reference for calls of several steps: its
  `RingKVCache` holds `context` (250) steps, so the first queries of a call lose their oldest
  keys, as moshi-swift's ring did ([swift-runtime](swift-runtime.md#chunks)). Its whole-sequence
  encode and decode use the exact window.

## Core ML

- `export_mimi_coreml.py CHECKPOINT MODEL_DIR --frames K --decoder-codebooks N` traces both
  steps and converts them to fp16 ML programs (iOS 18, CPU and Neural Engine), in ~17 s. N is
  the LM's `dep_q`; the Swift runtime checks it.
- `NeuralDecoder` compiles the package when the translator loads, reads k and the codebook
  count from its input shapes, and keeps the state arrays the model returns as the next call's
  inputs. `Translator` decodes the queued target frames in whole groups of k, so audio waits
  until k frames are ready; `flush` fills the last group with copies of the last frame and
  resets the decoder.
- Compute plan (`coremltools.models.compute_plan.MLComputePlan`): the decoder runs 453 ops on
  the Neural Engine and 73 on the CPU (the codebook gathers, the last two convolutions and
  transposed convolutions, three ELUs); the encoder 460 and 122 (its quantizer: argmin,
  codebook matmuls, gathers).
- M4 Pro Neural Engine, 2 frames per call: decoder 5.2–5.8 ms, encoder 11 ms. In `translate`
  the decoder costs 2.1 ms a frame, against 14–22 ms for the MLX decoder on the CPU; the text
  is unchanged (the LM never sees decoded audio).
- A15 (iPhone 13 Pro Max), 10 paced minutes: the decoder costs 4.2 ms a frame from cold to
  throttled, against 15–17 ms (2-frame calls) for the MLX CPU decoder, and the phone's step
  becomes GPU-bound throughout ([swift-runtime](swift-runtime.md#device-test)).
- Core ML returns autoreleased objects; on the bench's thread, which has no run loop, they piled
  up until it ended (1.6 MB a minute on the phone). `NeuralDecoder.decode` runs each call in an
  `autoreleasepool`.

## Precision

- Decoder in fp16 on the Neural Engine: 37 dB SNR against moshi's fp32 decode, 40 dB against
  the former MLX decoder (which used the tanh GELU); far below Mimi's own coding error.
- Encoder in fp16 on the Neural Engine: codes equal fp32 on 94% of frames in codebook 0, 59% in
  1–7 and 17% in 8–15. MLX in fp16 (fp32 accumulation) keeps 99.8, 98 and 95%, so the encoder
  stays on the CPU in fp32. The error enters inside SEANet (1.7% at its output, 4.4% at the
  latent): at convolutions whose output is a small difference of large terms (the first
  residual convolution sums ~20x its output's magnitude and comes out 6% off; block 3 too).
- Tried and dropped: centering each convolution's input on calibrated per-channel means (bias
  folded, exact in fp32) cut one layer's error 4x when tested alone, but changed nothing in the
  full model. Small single-op Core ML models may run on the CPU rather than the Neural Engine,
  so test ops inside the full model and check the compute plan.
- To run the encoder on an NPU: train the LM on the NPU encoder's codes, or find an
  accelerator mode that keeps more precision in its sums (untested: Qualcomm's HTP with 16-bit
  integer activations, which it sums in 32-bit integers).

## Qualcomm

- Android has no MLX, so a Qualcomm phone needs another runtime for the LM too (ExecuTorch with
  the QNN backend, or ONNX Runtime with the QNN execution provider). `mimi_step.py` is the
  pilot of that path: the same module goes to ONNX and then QNN. Check every layer against fp32
  on the target before trusting it, as the Neural Engine's encoder showed; Qualcomm AI Hub
  compiles and profiles on hosted Snapdragon phones.
