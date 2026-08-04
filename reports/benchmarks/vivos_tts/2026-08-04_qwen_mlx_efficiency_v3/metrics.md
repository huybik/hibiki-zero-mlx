# Qwen3-TTS MLX Apple-efficiency v3

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 `a6eb4f68…` · B8 same-speaker generation · fresh 16-row quantization screen with pinned Whisper/WavLM QA.

| Candidate | Rows/min | Gain vs paired bf16 | Active-memory reduction | WER (Δ) | Median cosine (Δ) | Decision |
|---|---:|---:|---:|---:|---:|---|
| bf16 | 62.631 | baseline | — | 0.3966 | 0.9004 | absolute no-go |
| bf16 + full reference cache | 62.355 | −0.4% | 0% | exact bf16 | exact bf16 | no fresh speed win |
| main MLP q8 g64 | 51.610 | −17.6% | 21.5% | 0.3966 (+0.0000) | 0.8794 (−0.0210) | reject |
| all main-transformer linears q8 g64 | 65.474 | +4.5% | 28.7% | 0.4655 (+0.0690) | 0.8787 (−0.0217) | reject |
| code-predictor transformer q8 g64 | 66.393 | +6.0% | 1.6% | 0.5690 (+0.1724) | 0.8873 (−0.0131) | reject |
| combined safe q8 g64 | 89.726 | +43.3% | 30.3% | 0.4483 (+0.0517) | 0.8827 (−0.0177) | reject |

The paired non-inferiority limits were WER Δ ≤ +0.01 and median speaker-cosine Δ ≥ −0.01, in addition to the existing absolute gates. No q8 candidate passed, so q6/q8 and selective q4/q8 were not justified, no candidate advanced to fresh 64-row QA, and no campaign launched. Quantized embeddings, codec/output heads, projections, norms, speaker encoder, tokenizer, and speech tokenizer were excluded; exact module lists and parameter counts are in `selection_report.json`.

The reusable full reference/prefix cache preserves **64/64 code arrays and 64/64 PCM WAV hashes** and leaves mlx-audio's allocator clearing unchanged. Its earlier v2 run was +4.6%, but fresh v3 timing was unstable: paired n=16 was −0.4%, while a drifted n=64 run was −39.7%. It therefore remains an exact implementation candidate, not a demonstrated production speed win.

Right-padding one preserved row to 16-token decoder buckets and trimming exactly was waveform-exact on 8/8 rows; the single-pass median speedup was 1.12× but shape-compilation noise is visible in the raw timings, so it is not a production selection. A dedicated MLX CPU decoder/NumPy-queue pipeline was attempted twice: the first run deadlocked after consumer failure, and the repaired run produced only 2/32 rows in more than 264 seconds versus 56.37 seconds for all 32 bf16-Metal rows. Its two scored outputs had WER 1.0 and median cosine 0.796; the path was terminated and rejected. A second Metal stream was not run because shared-model allocator clearing and stream-bound lazy arrays did not provide a safe ownership boundary.

No throughput-per-watt claim is made: authoritative `powermetrics` was unavailable without privileges. Wall time, MLX/RSS memory, macOS thermal status, raw timings, code/WAV hashes, QA, exact commands, failures, model/source/config/cohort hashes, and external audio/code paths are archived here.
