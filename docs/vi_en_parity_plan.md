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
| GRPO comes last, after every SFT stage (user, 2026-11-04) | SFT on new data retrains the timing its data teaches and would undo most of what RL tuned; the paper's order is base, fine-tune, then RL |
| The real-Vietnamese tranche is research and includes non-commercial corpora; their weights are research-only, and each source has its own cache (user, 2026-11-04) | The non-commercial corpora (PhoAudiobook, viVoice, Bud500, GigaSpeech 2) have cleaner transcripts, more speakers and long-form audio than YODAS; a commercial model is the same SFT without their caches ([real-vi-tranche](agent-memory/real-vi-tranche.md#decision)) |
| Two-way speech in one model whose output language follows the input language, prepared now beside the tranche (user, 2026-11-04) | A conversation needs both directions with no language-ID step; Hibiki-Zero has no conditioning input; two 3B models do not fit a 6 GB phone ([stage 7](#7-two-way-speech)) |
| Prepare the phone model for NPUs, Qualcomm's next (2026-11-02) | Phones with stronger NPUs are the later target. Android has no MLX, so each model piece is written once as a PyTorch step with explicit state and fixed shapes and exported per backend: Core ML now, ONNX/QNN later. Mimi is first ([mimi-npu](agent-memory/mimi-npu.md)) |

## Stages

### 1. Data

1. The cache, built unchanged and finished on 2026-11-02
   ([sft-data-caches](agent-memory/sft-data-caches.md#builds)). The trainer builds streams from its
   rows; nothing needs re-encoding.
2. Noisy Vietnamese re-encode (built 2026-11-04, the second epoch's data):
   `training.prep.mimi_cache_cuda` copies the cache with the Vietnamese source of a seeded 60% of
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
3. Real Vietnamese source tranche (in preparation; the larger lever, see 1.2): a first ~2,000 h of
   real Vietnamese speech (PhoAudiobook, viVoice, Bud500, ~700 h of YouTube video marked CC BY
   from YODAS, Common Voice, FOSD, VIVOS), its transcript translated by an LLM and the English
   voiced in each speaker's own voice by a TTS chosen in a bake-off (VoxCPM2, Qwen3-TTS,
   CosyVoice3) ([real-vi-tranche](agent-memory/real-vi-tranche.md)). This is the paper's recipe
   for a new language and the main defence against the TTS-only source. The paper adapted
   Italian with 850 h, but from related languages; Vietnamese starts from TTS only, so the lift
   per 1,000 h on FLEURS and MultiMed decides the scale-up toward ~10k h (GigaSpeech 2, more
   YODAS). Its rows serve both directions ([stage 7](#7-two-way-speech)).
4. First run pool: `en-vi-grounded-v2/train`. `cv-anquach` (16 kHz English targets) and the PhoMT
   cache (half its rows not voice-matched) stay out of the first run and come back as ablations.

### 2. Evaluation sets

Built by `training.prep.eval_sets` into `data/eval/`; evaluation only, never training
([sft-evaluation](agent-memory/sft-evaluation.md#evaluation-sets)).

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
   The trainer promotes the checkpoint whose every set passes the health gates (nonempty,
   repeated 4-grams, length ratio; EOS is reported, not gated, since 2026-11-04) with the highest
   FLEURS dev chrF, and keeps it beside the newest two. `eval.py` then checks the promoted checkpoint on MultiMed `select`:
   chrF no more than 1 point below the previously promoted checkpoint's, and the same gates.

### 4. GRPO

[docs/grpo_plan.md](grpo_plan.md): the paper's GRPO with BLEU process rewards on 30–75 s streams,
the last training stage (user, 2026-11-04): from the final SFT model, on both directions, after the
real-Vietnamese tranche. Its trainer is built (PR #59) and is profiled on the H100 before its run.
RL the teacher before any distillation. A penalty on speech while the source is silent, and a short
GRPO on a distilled student, come later if needed.

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

### 7. Two-way speech

One model translates a conversation both ways, each side in its speaker's own voice: Vietnamese
in, English out; English in, Vietnamese out (user, 2026-11-04). Hibiki-Zero has no conditioning
input, so the output language follows the input language. It is prepared now, beside the
tranche, and a pilot gates it before the final SFT. Every dataset row holds real English audio,
Vietnamese audio voiced in the same speaker's voice and both texts, so the data serves both
directions; EN→VI is the paper's own setup (real source, synthetic target). The tranche's rows
flip the other way and give EN→VI real Vietnamese targets.

```text
VI→EN:  en_speech  TTS Vietnamese  ─►  real English + English text
        tranche    real Vietnamese ─►  TTS English  + English text
EN→VI:  en_speech  real English    ─►  TTS Vietnamese  + Vietnamese text
        tranche    TTS English     ─►  real Vietnamese + Vietnamese text
```

The tokenizer blocks it first. Measured on the FLEURS dev sentences (361 rows, 2026-11-03), it
spends 4.2 tokens per Vietnamese syllable, 44% of them byte fallbacks, against 1.3 per English
word. Spoken at FLEURS pace, Vietnamese text needs 10.3 tokens/s on average, 13.2 at p90 and 17
at most. The text stream carries one token per 12.5 Hz frame and needs room for PAD, so
Vietnamese text cannot keep up with its speech.

1. Mimi on Vietnamese (measured 2026-11-04, Whisper large-v3-turbo, 80 FLEURS dev clips against
   their Vietnamese transcripts and 80 real English dataset clips against theirs):

   | Audio | English WER | Vietnamese WER (CER) |
   |---|---:|---:|
   | Original | 6.8% | 14.3% (10.5%) |
   | Mimi, 16 codebooks | 6.3% | 27.8% (20.4%) |
   | Mimi, 8 codebooks | 8.9% | 42.0% (32.2%) |

   Mimi leaves English intact but doubles the Vietnamese errors, and stripping the diacritics
   barely lowers them, so the loss is not only tones. It matters for both directions: EN→VI
   speaks through Mimi, and VI→EN hears Vietnamese only through its 16 codebooks, which may cap
   how much of the real-speech gap any data closes. The 8-codebook phone student loses more.
   Still to check on the box: 32 codebooks, and a control degradation of similar strength on
   both languages, to tell Mimi's loss from Whisper's weaker Vietnamese robustness.
2. Give Vietnamese syllables their own tokens by replacing pieces English never uses, so the text
   embedding, the head and every runtime keep `text_card` 48,000 and only the tokenizer file
   changes. Built 2026-11-04 by `training/vi_tokenizer`
   ([vi-tokenizer](agent-memory/vi-tokenizer.md)): 6,000 slots English never uses now hold
   Vietnamese letters and words. Vietnamese text drops from 4.10 to 1.28 tokens per word and
   from 17.2 to 7.7 tokens/s of speech with one PAD per word (held-out rows), and English is
   identical on all 3.63M rows. New pieces start from the mean of the rows of the pieces they
   used to split into. Still to build with the pilot: the config, the init on the checkpoint,
   the runtimes' UTF-8 assembly from byte pieces, and a Vietnamese aligner (step 3), since
   `text_timing._spoken_group` folds text to ASCII for the English aligner.
3. Build an EN→VI cache: Vietnamese text word-timed by a Vietnamese CTC aligner
   (`wav2vec2-base-960h` is English-only), the random delay on the Vietnamese target, and
   source and target codebooks swapped in the `[33, T]` rows.
4. Evaluation: FLEURS `en_us` speech with the `vi_vn` sentences as references (paired by
   sentence id, as the VI→EN sets are), held-out Common Voice dev and Libriheavy dev rows, and
   a Vietnamese ASR on the output audio.
5. A pilot of ~10k steps from the VI→EN model, both directions mixed (the trainer already mixes
   caches), with the new tokenizer. It passes if EN→VI chrF rises with healthy outputs and VI→EN
   FLEURS dev stays within 1 chrF. Then the final SFT trains both directions on `en_speech` and
   the tranche, and GRPO covers both. An EN→VI-only fine-tune is the fallback if the mixed pilot
   learns EN→VI too slowly.
6. The app must keep the model's own speech out of the microphone, or the model translates its
   own translation.

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
     settings; the channel still costs 5.0 (`en-vi-val-noisy` 38.4). Those Mac numbers are at
     batch 16; as `eval.py` now batches, the same model scores 25.1, 43.5 and 38.0, the second
     run's baseline ([noisy-source](agent-memory/noisy-source.md#evaluation)). More passes over
     clean TTS speech cannot fix the real-speech gap, so the second epoch trains on the noisy
     cache (stage 1.2) as a new run from this run's final model, with new streams (seed 43) and
     validation every 18,000 steps ([finetune](finetune.md#a-second-epoch-on-the-noisy-cache)), which translate
     all five sets as `eval.py` does; one Mac call over them takes 21.5 min
     ([sft-evaluation](agent-memory/sft-evaluation.md#side-tools)).
   - That run, `en_speech_noisy_epoch2`, moved to a cheaper 700 W H100 SXM at step 18,000
     ([finetune](finetune.md#move-a-running-run-to-another-pod)) and ended at step 131,703 at
     13:06 UTC on 2026-11-04.
     - FLEURS dev chrF by validation (every 18,000 steps, then the last step): 27.5, 27.4, 27.5,
       28.8, 29.1, 29.0, 29.0, 29.5, with 237–291 of 361 rows ended. The final model is the best
       and is kept in `offline_best/` on the Hub (every validation fails the health gates). From
       step 18,000 to the end, `en-vi-val` went 43.0 → 44.7 (Common Voice) and 43.7 → 45.2
       (Libriheavy), `en-vi-val-noisy` 38.4 → 40.8 and 41.5 → 43.4.
     - Test sets, in one `eval.py` call over FLEURS test and both MultiMed-ST halves (batch 64,
       `--seed 43`, text only), its final model against the first run's. Predictions are on the
       Hub under `<run>/eval/test_step*/`.

       | Set (rows) | chrF | BLEU | Rows ended |
       |---|---:|---:|---:|
       | FLEURS test (857) | 29.5 vs 24.7 | 5.4 vs 4.0 | 653 vs 378 |
       | MultiMed-ST select (1,718) | 24.9 vs 23.3 | 3.0 vs 2.9 | 893 vs 479 |
       | MultiMed-ST final (1,719) | 24.7 vs 23.2 | 3.1 vs 3.1 | 829 vs 514 |

     - FLEURS BLEU stays near 5 while the held-out TTS rows score 15–20. At step 90,000 FLEURS
       dev's n-gram precisions were 28, 6, 2 and 1%, 45 of 361 hypotheses shared a 4-gram with
       their reference, and hypotheses ran 19% long; lowercasing and dropping punctuation moved
       BLEU only from 4.2 to 4.5. On real speech the model drifts from the source (paraphrase,
       invented content), which only real Vietnamese speech in training (stage 1.3) addresses.
4. Done: a learning-rate test on the box (2026-11-04, 13:31–16:28 UTC): 10k steps from run 2's
   final model at peak LR 1e-6 (the control), 3e-6 and 1e-5 (5% warmup, cosine to 0, `--seed
   44`, the noisy cache), validated every 2,500 steps. The paper's language stage trained at
   3e-5 with ~15× more audio per step; an earlier recipe collapsed into silence at 3e-5 to 5e-5,
   so the test stopped at 1e-5. No rate beats run 2's final model (FLEURS dev chrF 29.47,
   teacher-forced loss 2.6108):

   | Peak LR | FLEURS dev chrF at 2.5k / 5k / 7.5k / 10k | Rows ended at 10k | TF loss at 10k |
   |---|---|---:|---:|
   | 1e-6 | 30.1 / 29.2 / 29.9 / 29.0 | 289 | 2.614 |
   | 3e-6 | 28.3 / 27.6 / 29.2 / 29.2 | 291 | 2.621 |
   | 1e-5 | 26.0 / 27.0 / 27.2 / 28.9 | 314 | 2.647 |

   A higher rate first knocks the model back (−2 chrF at 3e-6, −3.4 at 1e-5) and only recovers
   as the cosine reaches 0; 1e-6 stays within FLEURS dev's ±0.6 noise. The model has converged
   on `en_speech`, so there is no third epoch on it. The tranche SFT, on new data, takes 3e-6:
   the highest rate whose dip recovers within the test.
5. Skipped: a third VI→EN epoch on `en_speech` (it needed ~1.5 chrF from a higher rate).
6. In parallel: the real-Vietnamese tranche (stage 1.3) and the two-way preparation (stage 7),
   then the two-way pilot.
   - Done (2026-11-06, the box): `tranche_epoch1b`, one epoch of the first tranche's three caches
     (777,090 rows, 1,760 h of real Vietnamese; 26,176 steps at 0.277 s, 16:37–18:43 UTC+7) from
     run 2's final model, peak LR 3e-6, `--seed 45`, validated every 8,000 steps (5–6 min each
     with MultiMed select). Every validation passed the health gates and promoted; the final
     model is the best. FLEURS dev chrF by validation: 37.40, 38.50, 39.14, 39.58 (run 2: 29.47).
     Held-out TTS (`en-vi-val`) stayed within ~2 of run 2 (final 44.5 and 43.1 against 44.7 and
     45.2; noisy 39.9 and 41.0 against 40.8 and 43.4). Fewer FLEURS dev rows end inside the tail
     as it trains (300, 252, 226, 217 of 361), the median end offset ~4.5 s.
     - Test sets, in one `eval.py` call (batch 64, `--seed 43`, text only), against run 2's final
       model; predictions and metrics on the Hub under `tranche_epoch1b/eval/test_step026176/`:

       | Set (rows) | chrF | BLEU | Rows ended |
       |---|---:|---:|---:|
       | FLEURS test (857) | 38.6 vs 29.5 | 10.5 vs 5.4 | 504 vs 653 |
       | MultiMed-ST select (1,718) | 34.6 vs 24.9 | 9.6 vs 3.0 | 985 vs 893 |
       | MultiMed-ST final (1,719) | 34.8 vs 24.7 | 9.8 vs 3.1 | 981 vs 829 |

     - The first ~1,760 h of real speech lifts every real-speech set by 9–10 chrF; the lift per
       1,000 h decides the scale-up toward ~10k h (stage 1.3). GRPO comes next, its rollouts on
       the tranche caches (already on the box) rather than the noisy `en_speech` cache (73 GB, ~5 h
       at the box's ~4 MB/s Hub download).
7. The final SFT: both directions on `en_speech` and the tranche.
8. GRPO, last, on the final SFT model and both directions
   ([plan](grpo_plan.md#5-phases-each-profiled-on-the-box-before-the-next)).
9. The phone student (8 codebooks, ~10 layers, a 3-layer depformer) trained on the streams
   directly, distilled only if it falls short; measure a current phone (A17 Pro / A18) before
   settling it.

## Risks

| Risk | Mitigation |
|---|---|
| TTS-only Vietnamese source | Real-speech evaluation from day one; noisy re-encode; real-VI tranche |
| The tranche's licences: non-commercial corpora; YouTube uploaders' own CC BY; English voices cloned from speakers who never consented | Research-only weights, marked CC BY-NC-SA on the public recovery repo; one cache per source, so a commercial build drops the non-commercial ones; legal sign-off before anything ships ([real-vi-tranche](agent-memory/real-vi-tranche.md#decision)) |
| EN→VI training costs VI→EN quality | The two-way pilot's gate: VI→EN FLEURS dev within 1 chrF |
| Lag grows over long input (English 1.15–1.27× longer) | Stream gaps bound the overhang; measure lag growth on long real recordings; GRPO latency rewards |
| 3B too slow or hot on phones | Measured on the A15: the 3B runs at 0.4x real time sustained. 10 layers, a 3-layer depformer, 2-frame Mimi encoder calls and Mimi's decoder on the Neural Engine hold real time for 10 minutes at 41–45 ms a frame (a warmer start throttled at ~9 minutes). Next: the encoder off the CPU, a current phone |
| Translation degrades over long input: upstream 3B on a 64 s clip looped repeats itself after ~1.6 min and emits garbage after ~2.4 min, in Python and Swift alike | Test on long distinct speech; SFT streams of up to 120 s; reset the session in silence if needed |
| Code-domain splicing at stream joins: each clip was encoded from a fresh Mimi state | Streams open with the encoder's start-up codes as inference does, and replace a mid-stream clip's start-up codes (Vietnamese frame 0, sometimes a syllable onset; English silence before the first word) with silence. English speech in a clip's first 16 frames and residual codebooks near joins still differ. Full fix: re-encode each clip after a silence pre-roll |
