# Qwen3-TTS MLX recurrent compilation v4

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 `a6eb4f68…` · fixed same-speaker B8 · unchanged RNG and sampling.

| Candidate | Scope | Rows/min | Gain | Generation-stage gain | Code/WAV exact | Decision |
|---|---:|---:|---:|---:|---:|---|
| eager bf16 | 16 | 53.826 | baseline | — | baseline | control |
| functional compiled code predictor | 16 | 61.862 | +14.9% | — | 16/16 | advance |
| + main-talker compiled pre/post split | 16 | 57.658 | -6.8% vs predictor | — | 16/16 | reject |
| eager bf16 | 64 | 38.807 | baseline | baseline | baseline | control |
| functional compiled code predictor | 64 | 39.742 | +2.4% | +5.3% | 64/64 | retain, no ≥3% total claim |

The five-layer code predictor now threads every layer K/V state as explicit arrays through 15 fixed-position closures; sampling remains outside. Functional eager and fixed-B8 compiled execution both have zero maximum logit and cache delta and exact top-1 across two frozen B8 prefills. Process-first calls across all positions cost 0.080 s in the isolated trace with the machine compiler cache warm; the warm fixed trace improved +23.1%.

The 28-layer talker split was also array-exact, but regressed end-to-end throughput -6.8% and was rejected. Shapeless compilation failed at position 0 with `[Primitive::output_shapes] Slice cannot infer output shapes.`; fixed B8 is the supported boundary.

Pinned 16-row QA on the compiled candidate gave WER 0.3966, median speaker cosine 0.9004, and zero prompt leaks. Paired quality deltas are exactly zero because every generated code array and PCM hash matches eager. Absolute QA remains no-go because the unchanged baseline WER exceeds 0.08; no campaign or upload was launched.

Active-lane compaction remains the next phase. This phase deliberately did not change RNG, compact lanes, quantize weights, fuse weights, or launch a full campaign.
