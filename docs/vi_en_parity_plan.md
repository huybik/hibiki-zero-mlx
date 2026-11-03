# VI→EN plan: from Hibiki-Zero to a real-time phone translator

Goal: Vietnamese speech in, English speech (in the speaker's voice) and text out, simultaneously,
on a phone, offline. The base is Hibiki-Zero 3B (user's choice, 2026-11-01): its tokenizer, Mimi
codec and runtime stay the same from training to the phone.

```text
VI TTS (done) ─► Mimi cache (finish) ─► SFT 3B ─► GRPO 3B ─► teacher
                 + noisy VI re-encode     streams,              │
                 + real-VI eval sets      fixed loss            ▼
                                            on-device test of upstream Hibiki-Zero 3B q4
                                            ├─ real time on target iPhones ─► ship 3B q4
                                            └─ too slow / hot ─► shrink in the family + distill
                                                                 (8 codebooks, ~10–14 layers)
                                                     ─► q4 + Swift runtime (4 Hibiki-Zero deltas)
```

## Where we are

- Data: the `en_speech` corpus (Common Voice 27 EN + Libriheavy, 10.5k h of 5–22.4 s clips) is
  translated and voiced in each row's own English voice
  ([en-vi-speech-dataset](agent-memory/en-vi-speech-dataset.md)). The grounded Mimi cache is
  built ([sft-data-caches](agent-memory/sft-data-caches.md#builds)), and the full 3B run trains
  on it ([order](#order)).
- The last run (`grounded_v2_full_direct_voice_5epoch`, PhoMT + FLEURS) improved teacher-forced
  loss but failed free-running translation ([analysis](analysis/validation_collapse_analysis.md),
  [examples](analysis/step135k-check.md)).
- No Vietnamese checkpoint has run on a phone; every iPhone number in `docs/vision/reports/` is a
  projection.

## What the current cache teaches (measured 2026-11-01)

300 rows each of `cv27-00010` and `libriheavy-medium-00050`, speech edges from a −40 dB energy
gate:

| | CV | Libriheavy |
|---|---|---|
| VI heard before the first English word, median (share under 1 s) | 1.8 s (19%) | 2.2 s (17%) |
| English speech ends after VI speech ends, median / p90 | 2.4 / 4.0 s | 3.8 / 6.8 s |
| English speech ÷ VI speech duration, median | 1.27 | 1.15 |

A ~3 s lag is the realistic target; the paper's RL-tuned Italian adaptation ends ~3 s after the
source. English runs longer than the VieNeu Vietnamese, so streams need source pauses or the lag
grows sentence by sentence.

## Decisions (2026-11-01, later dates marked)

| Decision | Why |
|---|---|
| Stay on Hibiki-Zero 3B; the phone model is the 3B or a smaller copy of it | One tokenizer, codec and runtime. Hibiki-M 1B shares Mimi and the tokenizer but only knows FR→EN |
| A tenth of the streams hold one sentence (2026-11-04) | Trained on 30–120 s streams only, the first run never saw the source end after one sentence, and never ended a third of FLEURS dev's single sentences (EOS on 55 of 361 rows at step 9,000, against 124 of 128 for the earlier single-sentence run) |
| Train on multi-sentence streams of 30–120 s, not single sentences (120 s from 2026-11-02) | Upstream's coarse stage trained on 30–75 s utterances and its RL stage on 120 s; upstream 3B degrades after ~1.6 min of input; a phone hears a continuous stream |
| Gaps between sentences are silence codes on both sides, sometimes long | Teach "stay quiet while nobody speaks"; the MLX runtime already has to cut the model off in silence (`PAD_STOP`) |
| Source after its EOS is silence codes, not `-1` | Upstream inference (`add_input_eos`) feeds encoded silence there; `-1` is an empty input the model never sees at inference |
| Keep the source EOS and text EOS at the end of each stream | Upstream models termination this way and forces the input EOS in offline use; a live stream simply never sends it |
| Audio codebook 0 weighted ×100; every text PAD supervised at 0.5 | Kyutai's `moshi-finetune` defaults. Before, codebook 0 was 1/16 of the audio loss and PADs after the first word had no loss, so nothing taught when not to emit a word; the last run's length ratio grew 1.06 → 1.28 → 2.05 |
| Batches by frame budget; warmup + cosine LR | Attention is ~2.5% of compute at 1,500 frames, so cost tracks frames, not rows; the constant 1e-6 LR kept moving the model after its optimum |
| Choose checkpoints by free-running chrF on real Vietnamese speech, inside training (2026-11-02) | All training and cache-validation speech is VieNeu TTS, and the last run's teacher-forced loss kept improving while its translations got worse |
| Prepare the phone model for NPUs, Qualcomm's next (2026-11-02) | Phones with stronger NPUs are the later target. Android has no MLX, so each model piece is written once as a PyTorch step with explicit state and fixed shapes and exported per backend: Core ML now, ONNX/QNN later. Mimi is first ([mimi-npu](agent-memory/mimi-npu.md)) |

## Stages

### 1. Data

1. The cache, built unchanged and finished on 2026-11-02
   ([sft-data-caches](agent-memory/sft-data-caches.md#builds)). The trainer builds streams from its
   rows; nothing needs re-encoding.
2. Noisy Vietnamese re-encode (built 2026-11-04, the second epoch's data):
   `training.prep.noisy_cache` copies the cache with the Vietnamese source of a seeded 60% of
   the rows re-encoded under room reverb, background noise, a narrowed band (16 kHz as FLEURS,
   8 kHz as MultiMed, a phone band) and a level change, reusing each row's English codes and
   text timing; it runs on the pod's GPU after a run, where the Mac would take days
   ([noisy-source](agent-memory/noisy-source.md)).
   - `en-vi-val-noisy` measures the channel. At step 63,000 of the first run it cost 4.8 chrF
     (41.5 → 36.7) of the ~18 between clean TTS and FLEURS dev (23.5)
     ([noisy-source](agent-memory/noisy-source.md#evaluation)): the noise makes the model
     robust to rooms and phones, but most of the FLEURS gap is real speech, for stage 3.
   - Left out: speed changes, since a source slowed or sped against the cached delays moves
     Vietnamese words past the English that translates them; and low-bitrate Opus.
3. Real Vietnamese source tranche (next after the noisy epoch; the larger lever, see 1.2): real Vietnamese speech (GigaSpeech 2 vi, Bud500; check
   licences) with LLM-translated text and voice-cloned English TTS targets, timed causally. This
   is the paper's stage-4 analogue and the main defence against the TTS-only source.
4. First run pool: `en-vi-grounded-v2/train`. `cv-anquach` (16 kHz English targets) and the PhoMT
   cache (half its rows not voice-matched) stay out of the first run and come back as ablations.

### 2. Evaluation sets

Built by `training.prep.eval_sets` into `data/eval/`; evaluation only, never training
([sft-training](agent-memory/sft-training.md#evaluation-sets)).

| Set | Speech | Role |
|---|---|---|
| FLEURS vi_vn dev (361) | real read speech, 16 kHz, human EN references | checkpoint selection |
| FLEURS vi_vn test (857) | same | final report |
| MultiMed-ST vi `select` (half of 3,437) | real conversational medical speech, **8 kHz** telephone band; references LLM-drafted, then corrected by 5 annotators | non-regression gate |
| MultiMed-ST vi `final` (other half) | same | final report |
| `en-vi-val` Common Voice dev and Libriheavy dev (180 rows each) | the training data's own TTS Vietnamese, held out | in-domain check at each validation, scored apart from FLEURS |
| Own phone recordings (to record) | conversational VI on a phone mic, incl. 2–3 min monologues | final gate before shipping |

MultiMed segments are cut mid-phrase and its references are clean sentences: read chrF, not
BLEU, and compare against upstream Hibiki-Zero rather than in absolute terms.

### 3. SFT (the stream recipe)

`training/finetune/train.py` ([sft-training](agent-memory/sft-training.md)):

1. Streams of 30–120 s: rows at random (consecutive PhoMT rows of one source chunk stay in
   order; Libriheavy segment numbers are not reading order and overlap); each row's English
   placed at its cached delay after its Vietnamese, never before the previous English ends;
   gaps drawn so the previous English overhangs the next Vietnamese by at most 2 s. That cap
   binds at the median join, so the source "waits" for the interpreter; watch lag growth on
   long real speech.
2. Loss as above; LR peak 1e-6, 5% warmup, cosine to 0; one pass over the pool.
3. Each validation logs teacher-forced losses on streams of the cache's `validation/` split and
   translates FLEURS dev and the held-out `en-vi-val` rows free-running, scoring each set apart.
   The trainer promotes the checkpoint whose every set passes the health gates (nonempty, EOS,
   repeated 4-grams, length ratio) with the highest FLEURS dev chrF, and keeps it beside the
   newest two. `eval.py` then checks the promoted checkpoint on MultiMed `select`:
   chrF no more than 1 point below the previously promoted checkpoint's, and the same gates.

### 4. GRPO

[docs/grpo_plan.md](grpo_plan.md), on streams with silent stretches, plus a penalty on speech
while the source is silent. RL the teacher before any distillation; a short GRPO on a distilled
student afterwards.

### 5. The phone model

1. Done (2026-11-01): `inference/swift/` runs Hibiki-Zero in MLX Swift with the four deltas
   (`hidden_scale`, grouped-query `kv_repeat`, `rope_concat`, per-slice depformer norms),
   matching the Python runtime ([swift-runtime](agent-memory/swift-runtime.md)).
2. Measured on an iPhone 13 Pro Max (A15, 6 GB), upstream 3B q4, 10 minutes of audio back to
   back: it fits in memory (2.8 GB, 1.3 GB to spare) but runs at 135 ms/frame cold and
   ~200 ms/frame once throttled, against an 80 ms budget; the LM alone is 100 → 153 ms
   ([swift-runtime](agent-memory/swift-runtime.md#device-test)). Still to measure: a current
   phone (A17 Pro / A18, 8 GB) before settling the target.
3. The 3B does not hold real time on the A15, so the shrink branch is on: drop depformer steps
   8–15 (8 codebooks per stream; the depformer's sequential steps halve) and main layers 28 →
   ~10–14 initialised from the 3B's own, and fine-tune it on the streams directly. Only if it
   falls short of the 3B, distil from the teacher on its free-running outputs over the cached
   Vietnamese codes and unlabeled real Vietnamese audio.
   Timed on the A15 with untrained cuts and Mimi on the CPU beside the GPU LM
   ([swift-runtime](agent-memory/swift-runtime.md#device-test)): 10 layers run at 58 ms/frame
   cold and hold real time for ~3 minutes, then throttle to ~90 ms; 14 layers 64 → 98 ms; the
   8-codebook 3B 84 → 137 ms. The phone reaches thermal state "serious" within a minute.
   Mimi encoding and decoding 2 frames per call halves its CPU time (hot: 80 → 36 ms a frame);
   the 10-layer step then turns GPU-bound once the GPU throttles (~4 minutes). Halving the
   depformer (each of its 8 slices keeps 3 of 6 layers) holds real time through 10 minutes on
   the A15 (42 → 79 ms a frame), and with Mimi's decoder on the Neural Engine the step is
   GPU-bound at 41 → 44.5 ms for all 10 minutes (an earlier, warmer run throttled at ~9). So
   the student is 8 codebooks, ~10 main layers and a 3-layer depformer; measure a current
   phone (A17 Pro / A18) before settling it.

### 6. Quantise and package

q4 group 32 on transformer linears; embeddings, text head and depformer input/output projections
at 8 bits (q3 there made the 3B babble, [inference_matrix](vision/reports/inference_matrix.md)).
Gate q4 against bf16 on the real-Vietnamese sets. q3 is out everywhere: teacher-forced on
`leon.wav` it keeps 82% of bf16's text tokens and 71–78% of codebook 0, against q4's 93% and 88%
([swift-runtime](agent-memory/swift-runtime.md#device-test)). Ship on MLX Swift: the LM on the
GPU; beside it Mimi encodes on the CPU in fp32 and decodes on the Neural Engine
(`export_mimi_coreml.py`; fp16 on the Neural Engine shifts the encoder's codes, codebook 0 on 6%
of frames, so the encoder stays on the CPU, [mimi-npu](agent-memory/mimi-npu.md)). File mode
should send the source EOS at the end of the input, as upstream does; live mode never sends it.
Sample the Vietnamese model as the paper does: temperature 0.8, top-k 250 on every token. The MLX
runtime's text sampling (0.4, top-k 25) suits upstream, but at 0.4 the fine-tuned model lags and
seldom ends ([analysis](analysis/eos_and_lag.md)).

### 7. English to Vietnamese (later)

A possible next step once the VI→EN phone model works (user, 2026-11-03), not started. Every
dataset row holds real English audio, Vietnamese audio voiced in the same speaker's voice and
both texts, so the data serves both directions; EN→VI is the paper's own setup (real source,
synthetic target).

```text
VI→EN (now):   source = TTS Vietnamese audio  ─►  target = real English audio + English text
EN→VI (later): source = real English audio    ─►  target = TTS Vietnamese audio + Vietnamese text
```

The tokenizer blocks it first. Measured on the FLEURS dev sentences (361 rows, 2026-11-03), it
spends 4.2 tokens per Vietnamese syllable, 44% of them byte fallbacks, against 1.3 per English
word. Spoken at FLEURS pace, Vietnamese text needs 10.3 tokens/s on average, 13.2 at p90 and 17
at most. The text stream carries one token per 12.5 Hz frame and needs room for PAD, so
Vietnamese text cannot keep up with its speech.

1. Check Mimi on Vietnamese first (cheap go/no-go): encode and decode real Vietnamese speech and
   compare a Vietnamese ASR's error before and after, to see whether the tones survive.
2. Add ~5–8k Vietnamese syllable tokens (to about one token per syllable): grow the text
   embedding and head past `text_card` 48,000; the MLX and Swift runtimes take the new size.
3. Build an EN→VI cache: Vietnamese text word-timed by a Vietnamese CTC aligner
   (`wav2vec2-base-960h` is English-only), the random delay on the Vietnamese target, and
   source and target codebooks swapped in the `[33, T]` rows.
4. Evaluation: FLEURS `en_us` speech with the `vi_vn` sentences as references (paired by
   sentence id, as the VI→EN sets are), held-out Common Voice dev and Libriheavy dev rows, and
   a Vietnamese ASR on the output audio.
5. Train EN→VI as its own fine-tune, since the model must learn to speak a new language. If the
   phone should ship one model, a joint fine-tune can then mix both caches (the trainer already
   mixes caches). Hibiki-Zero has no target-language switch, so the output language follows the
   input language; keep the joint model only if it matches both single-direction models.

## Order

1. Done: the evaluation sets and the stream recipe; the Swift port and the A15 device test;
   checkpoint selection by free-running chrF inside training, and 30–120 s streams; the full
   cache build ([sft-data-caches](agent-memory/sft-data-caches.md#builds)).
2. Done (2026-11-03, H100 NVL): the trainer 1.5x faster per frame, startup in seconds, cheap
   frequent checkpoints, and runs that survive preemption
   ([sft-speed](agent-memory/sft-speed.md#by-gpu)).
3. Running (started 2026-11-03): the full 3B run `en_speech_streams_epoch1b`, one epoch of
   `en-vi-grounded-v2/train` on a 700 W H100 SXM (~10.5 h with validations). The first validation
   (step 9,000, ~45 min in) replaces a separate smoke run. On the first attempt
   (`en_speech_streams_epoch1`, which then ran out of memory) the training loss fell from 10.1
   to 3.05 and teacher-forced content accuracy reached 0.51 (PAD 0.947).
   - By step 36,000 content accuracy was 0.62 and FLEURS dev chrF 15.9, but only 38 of 361
     rows found EOS.
   - Most of that came from the decode. The run's validation samples text at 0.4 with
     an 8 s tail; at the paper's 0.8 and a longer tail, the step-18,000 model ended 33 of 48
     FLEURS rows instead of 11, and chrF rose from 18 to 22
     ([analysis](analysis/eos_and_lag.md)).
   - The run validates with its commit's old settings, so its gates cannot pass and it promotes
     nothing. Even at the new settings a third of FLEURS rows never end, short of the gate's 91%.
     Its checkpoints are scored with `eval.py`'s new defaults as each reaches the recovery repo,
     which keeps the newest two pairs; each validation step's model is copied to the run's
     `offline_best/` there, pruned to the newest and the best by FLEURS dev chrF.
   - To step 63,000 (of 131,553) teacher-forced validation loss kept falling but slowed
     (3.17, 2.94, 2.85, 2.79, 2.76, 2.73, 2.72 every 9,000 steps), with content accuracy flat
     at 0.63 since 54,000. It sits ~0.2 above the training loss, a gap nearly unchanged since
     18,000, so no overfitting. FLEURS dev chrF at the run's settings rose 12.3 → 17.8; on the
     Mac at the new ones, 20.7, 22.6, 23.1 at 18k, 36k, 45k; on held-out TTS (`en-vi-val`) at
     18k, 31.4 (Common Voice) and 36.3 (Libriheavy). Its final model (step 131,553) has
     FLEURS dev 24.9 and `en-vi-val` 43.4 on the Mac, 18.8 and val loss 2.673 at the run's
     settings; the channel still costs 5.0 (`en-vi-val-noisy` 38.4). More passes over clean TTS speech cannot
     fix the real-speech gap, so the second epoch trains on the noisy cache (stage 1.2) as a
     new run from this run's final model, with new streams (seed 43) and validation every
     18,000 steps ([finetune](finetune.md#a-second-epoch-on-the-noisy-cache)). The Mac scores
     only chosen checkpoints: one FLEURS or `en-vi-val` pass takes it ~30 min, a checkpoint
     comes every ~43.
4. GRPO on the best checkpoint of that run.
5. The phone student (8 codebooks, ~10 layers, a 3-layer depformer) trained on the streams
   directly, distilled only if it falls short; measure a current phone (A17 Pro / A18) before
   settling it.
6. Possibly, once the VI→EN phone model works: EN→VI
   ([stage 7](#7-english-to-vietnamese-later)).

## Risks

| Risk | Mitigation |
|---|---|
| TTS-only Vietnamese source | Real-speech evaluation from day one; noisy re-encode; real-VI tranche |
| Lag grows over long input (English 1.15–1.27× longer) | Stream gaps bound the overhang; measure lag growth on long real recordings; GRPO latency rewards |
| 3B too slow or hot on phones | Measured on the A15: the 3B runs at 0.4x real time sustained. 10 layers, a 3-layer depformer, 2-frame Mimi encoder calls and Mimi's decoder on the Neural Engine hold real time for 10 minutes at 41–45 ms a frame (a warmer start throttled at ~9 minutes). Next: the encoder off the CPU, a current phone |
| Translation degrades over long input: upstream 3B on a 64 s clip looped repeats itself after ~1.6 min and emits garbage after ~2.4 min, in Python and Swift alike | Test on long distinct speech; SFT streams of up to 120 s; reset the session in silence if needed |
| Code-domain splicing at stream joins: each clip was encoded from a fresh Mimi state | Streams open with the encoder's start-up codes as inference does, and replace a mid-stream clip's start-up codes (Vietnamese frame 0, sometimes a syllable onset; English silence before the first word) with silence. English speech in a clip's first 16 frames and residual codebooks near joins still differ. Full fix: re-encode each clip after a silence pre-roll |
