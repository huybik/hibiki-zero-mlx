# iPhone Budget Report — projected per-frame time & memory vs the 80 ms budget

**Everything below the M4 column is a projection. No iPhone was in the loop.** The
M4 Pro numbers are measured with `scripts/bench.py` (150 timed frames of `leon.wav`,
15 warmup, fixed seed, `mx.eval` barriers between stages). The iPhone columns apply a
*documented, assumed* GPU-throughput scale to those numbers — they are an engineering
estimate to size the problem, **not** a device measurement.

Hibiki emits one frame every **80 ms** (12.5 Hz). To be realtime, all per-frame work
on the live critical path (the LM step) must fit in 80 ms; the Mimi codec runs on CPU
threads and is hidden by the 3-thread pipeline (`hibiki_mlx/pipeline.py`).

## Scale assumption (stated explicitly)

`iphone_ms = m4_ms / scale`, where `scale` = assumed **A18(-Pro) GPU / M4 Pro GPU
throughput ratio for these kernels**.

- **Default `scale = 0.5`; plausible band `0.4–0.6`.** Rationale: the A18 Pro GPU
  (iPhone 16 Pro, 6 cores) has on the order of half the shader throughput and memory
  bandwidth of an M4 Pro GPU (16–20 cores). The main transformer and the quantized
  head are memory-bandwidth-bound, so a bandwidth-fraction scale is the right shape.
- **Caveat that the scale does *not* model:** the AR depformer is *launch-bound*, not
  compute-bound (~0.6–0.9 ms/slice, 8/16 sequential kernel launches). Per-dispatch
  overhead need not scale like bandwidth; on a phone GPU it can be relatively *worse*,
  which would push the AR numbers up. This is exactly why the parallel head (one
  forward, bandwidth-bound) is the safer on-device bet.
- These are **projections, not measurements.** Real A-series silicon, Metal driver
  dispatch cost, ANE-vs-GPU placement, and thermals can each move them materially.

## Per-stage M4 Pro measurements (`scripts/bench.py`, this run)

| stage | 1B q4 (ms) | 3B q4 (ms) |
|---|---:|---:|
| mimi encode (CPU, hidden) | 17.5 | 17.7 |
| **LM main transformer** (GPU) | 8.05 | 12.32 |
| **LM codebook head — AR depformer** (GPU) | 7.02 (8× 0.88) | 9.90 (16× 0.62) |
| mimi decode (CPU, hidden) | 16.3 | 16.4 |
| **LM total (live critical path)** | **15.06** | **22.21** |

Parallel head (Phase 3 smoke, measured on 3B, `reports/parallel_head_smoke.md`):
codebook head **5.28 ms** (one bf16 forward, 227 M params, bandwidth-bound) replacing
the AR depformer.

## Projected iPhone frame time vs the 80 ms budget

LM step only (codec is off the critical path). Three configs the task asks for:

| config | LM main (M4) | codebook head (M4) | LM total M4 | iPhone @0.6 | **iPhone @0.5** | iPhone @0.4 | vs 80 ms |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **1B, AR head** (shipped) | 8.05 | 7.02 (AR) | **15.06** | 25.1 | **30.1** | 37.7 | **FITS** |
| **1B, parallel head** † | 8.05 | ~5.3 (par, bf16) | **~13.4** | 22.3 | **26.7** | 33.4 | **FITS** |
| **3B, AR head** | 12.32 | 9.90 (AR) | **22.21** | 37.0 | **44.4** | 55.5 | **FITS** |

† Parallel-head row substitutes Phase 3's **measured 5.3 ms bf16 head** (measured on
the 3B's 16-codebook 227 M head) for the depformer — a **conservative upper bound** for
a 1B head, which has only 8 codebooks and would be smaller/faster. `reports/
inference_matrix.md` projects an optimized (q4 + param-shrunk) 1B parallel head at
**~8.5–11 ms LM total on M4 → ~17–22 ms iPhone @0.5**. The head is a *documented
follow-up* — the shipped artifact runs the AR head. On the 1B the AR depformer is
already only 7 ms, so the parallel head's win is modest on the 1B; its real value is
(a) margin that survives a real device and thermals, and (b) making the **3B**
phone-viable (3B: LM 22.2 → ~13–16 ms M4 → ~26–40 ms iPhone).

**All three configs project inside the 80 ms budget at every scale in the 0.4–0.6
band.** The 1B AR head — the shipped artifact — lands at **~30 ms @0.5 (worst case
~38 ms @0.4)**, roughly **2.6×–2.1× headroom**.

### Codec (CPU) sanity
Mimi encode ~17.5 / decode ~16.4 ms per frame on M4. At a pessimistic 0.5× CPU scale
each is ~35 ms — still under one 80 ms frame, and both are hidden behind the LM by the
3-thread pipeline. They stay off the critical path *provided the phone app keeps the
encoder/LM/decoder pipeline*; collapsing to a single thread would serialize
~35+15+35 ≈ 85 ms and blow the budget.

## Memory footprint vs a 3rd-party iOS app budget

Resident set = LM weights + KV cache (capped live window, Phase 2 S4) + Mimi codec +
runtime/activation working set.

| component | 1B q4 | 1B q4-depq3 | 3B q4 |
|---|---:|---:|---:|
| LM weights (q4 gs32) | 1.13 GB | 1.05 GB | 2.41 GB |
| KV cache at cap (bf16) | 0.07 GB (500 fr / 40 s) | 0.07 GB | 0.34 GB (3000 fr / 4 min) |
| Mimi codec (bf16) | 0.38 GB | 0.38 GB | 0.38 GB |
| activations + runtime (est.) | ~0.2 GB | ~0.2 GB | ~0.3 GB |
| **total (AR head)** | **~1.8 GB** | **~1.7 GB** | **~3.4 GB** |
| + parallel head (bf16 / q4) | +0.45 / +0.11 GB | — | +0.45 / +0.11 GB |

**iPhone app RAM budgets (approximate, jetsam-limited):**
- iPhone 15 / 16 (base, **6 GB** RAM): a 3rd-party app can safely resident ~**2.5–3 GB**
  before the OS jetsams it.
- iPhone 15 Pro / 16 Pro (**8 GB** RAM): ~**4–5 GB** with the *increased-memory-limit*
  entitlement (the one Apple-Intelligence-class apps use).

→ The **1B (~1.8 GB, or ~1.7 GB depq3)** fits comfortably even on a **base 6 GB**
device, with room for a bf16 parallel head (~2.3 GB). The **3B (~3.4 GB)** needs an
**8 GB Pro** device plus the extended-memory entitlement — another reason the 1B is the
phone artifact and the 3B stays the Mac/teacher.

## What still blocks *true* on-device realtime (be honest)

These are the reasons the table above is a projection, ranked by risk:

1. **No device build in the loop.** moshi-swift has not run *these exact weights* on an
   iPhone. `scripts/check_swift_compat.py` proves the artifact loads the way moshi-swift's
   loader does (gs32 q4 naming/shapes, config keys, tokenizer+mimi), but loading ≠
   measured realtime. A device build + on-device timing is the one thing that converts
   these projections into a result.
2. **ANE vs GPU execution.** The product vision targets the **Apple Neural Engine**; the
   current MLX / moshi-swift path executes on the **Metal GPU**. ANE could help the dense
   matmuls but maps *poorly* to the launch-bound AR depformer — which strengthens the case
   for the parallel head before an ANE port. Numbers on ANE are unknown.
3. **Real thermals / sustained clocks.** The 0.5× scale models a burst, not a phone
   throttling over a multi-minute conversation. Sustained realtime needs thermal headroom
   the projection doesn't capture — the 1B's 2× margin is the buffer here.
4. **Launch-bound dispatch on a phone GPU.** Per-kernel dispatch overhead may not scale
   like bandwidth; the AR depformer's 8/16 serial launches are the most fragile part of
   the estimate. The parallel head removes this dependency entirely.
5. **Pipeline + memory pressure.** The budget assumes the 3-thread codec pipeline holds
   on iOS and that the OS doesn't jetsam under real memory pressure (esp. the 3B on base
   devices).

## Bottom line

**In projection, the shipped 1B q4 AR artifact meets the 80 ms iPhone budget with
comfortable headroom (~30 ms @0.5×, ~38 ms worst-case @0.4×) and fits in a base-6 GB
app RAM budget (~1.8 GB).** The parallel head is a documented follow-up that widens the
margin and is what would make the 3B phone-viable. The remaining unknowns are all
device-side — a moshi-swift on-device build, ANE-vs-GPU placement, and sustained
thermals — none of which can be resolved without an iPhone in the loop.
