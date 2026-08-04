# Qwen3-TTS MLX production memory diagnosis

Date: 2026-08-04 · observations 20:19:02–20:28:29 ICT (UTC+07:00) · Apple M4 Pro / 48 GiB · live retry-v6 attempt 0.

This is a derived, read-only diagnosis of the running production campaign. It did not load a model, signal or pause a process, mutate production artifacts, run QA, build a cache, or contact Hugging Face.

## Result

**Confirmed: the production loop retains MLX allocator cache across atomic groups.** The live process reached a 34.7 GiB physical footprint (34.8 GiB peak), dominated by 34.2–34.3 GiB of nonvolatile `IOAccelerator (graphics)` memory across about 5,932–5,954 regions. Repeated 15-second samples stayed near 35 GiB while completed groups advanced from 576 to 578. This is a high-water allocator-cache plateau, not evidence that active model tensors grow per group.

| Observation | Result |
|---|---:|
| Live process footprint | 34.7 GiB; 34.8 GiB peak |
| `IOAccelerator (graphics)` | 34.2–34.3 GiB resident; ~5.9k regions |
| Graphics memory swapped in the 20:19 `vmmap` sample | 17.9 GiB |
| Total process memory swapped in that sample | 18.3 GiB |
| MLX per-group peak, first 581 completed groups | 5.754 / 5.934 / 5.957 GiB min/median/max |
| Python-retained completed `group.json` payload, 581 groups | 16,413,542 bytes (15.65 MiB) |
| File descriptors | 84, stable |
| Global swap at initial sample | 24,540.19 / 25,600 MiB used |
| Repeated-sample state | 35 GiB flat while groups 576 → 578 |
| Prior v2 normal-clear MLX cache | 0.085 GiB median after group |
| Prior v2 retained-cache MLX cache | 25.250–25.594 GiB after group |
| Prior v2 retained-cache throughput effect | 33.903 vs 36.008 rows/min, **12.2% slower** |

The production group records bound the active workload: all predictor widths 1–8 had appeared, 19 speakers were covered, the per-group maximum generated-frame statistic was 34 / 69 / 156 min/median/max, and maximum output duration was 12.48 seconds. These distributions coexist with a nearly flat 5.75–5.96 GiB MLX per-group peak, contradicting active-tensor accumulation as the main explanation.

## Owning boundary

The defect is in `training-data/benchmark_vivos_qwen_mlx_retry_v6.py`:

- `run_group` line 408 calls private `model._decode_generated_codes`, bypassing the installed `mlx-audio` public non-streaming cleanup at `qwen3_tts.py` line 2062.
- `execute` line 532 clears MLX cache once after warmup.
- The production loop at lines 534–548 commits and records every later group without a per-group `mx.clear_cache()`.

The prior controlled v2 experiment independently reproduces the signature: retaining allocator cache left 25.25–25.59 GiB cached and was 12.2% slower than normal clearing. That archived result and the live `IOAccelerator` plateau jointly confirm the diagnosis.

## Hypotheses bounded by evidence

- **Confirmed:** MLX allocator-cache retention across group boundaries is the primary owner of the high physical footprint.
- **Contradicted as primary causes:** active model tensors, the completed-record Python list, open file descriptors, the 19-entry speaker/reference cache, predictor active-width specialization, and bounded vocoder/output shapes.
- **Not isolated:** main-talker prefill compilation may contribute to the retained high-water mark, but its independent share cannot be measured without changing or instrumenting the running process.
- **Terminology:** users experience this as a memory leak, but the observed mechanism is retained allocator/cache high-water memory. The short repeated sample proves a plateau at diagnosis time, not indefinite monotonic growth.

## Separate metadata defect

Line 439 records `token_count = len(codes)`. Stored code arrays have shape `[1, T, 16]`, so the value is always `1`; the intended frame count is `codes.shape[1]`. WAV and code-array contents are unaffected. This metadata bug is independent of allocator retention and must not be cited as its cause.

## Risk and repair procedure

Continuing unchanged risks swap exhaustion, severe system paging, throughput loss, and a host-level failure despite stable per-group MLX peaks. The postprocess supervisor must not interpret an intentional generator stop as completion.

Recommended procedure, **not executed by this diagnosis**:

1. Stop the postprocess supervisor first and record its terminal state so it cannot validate or launch QA after the generator exits without a completion sentinel.
2. Stop the generator at an atomic group boundary. Preserve every committed group; record and remove only an incomplete hidden temporary group after verifying it has no committed counterpart.
3. Run the CPU production validator against the completed prefix and record group/row/hash counts before editing.
4. Add `mx.clear_cache()` at the shared committed-group boundary in `execute`, after `run_group` has atomically returned. Fix `token_count` to `codes.shape[1]` separately.
5. Create new script hashes and an immutable repair/supersession attestation. Do not rewrite the original plan, launch record, groups, or v2 evidence.
6. Re-test the repaired loop on a history-disjoint multi-speaker B8 cohort. Require exact code/WAV equivalence for the clear-cache-only change, stable QA, cache returning near the normal-clear regime, no `IOAccelerator` region growth across repeated groups, and no throughput regression.
7. Resume the exact 10,950-row source scope in its existing resumable namespace. Validation must hash-check and skip every committed group. Launch a new supervisor bound to the new generator PID and repair attestation.

The concrete read-only commands, raw observations, hypotheses, environment, and immutable bindings are preserved alongside this report. Audio remains external.
