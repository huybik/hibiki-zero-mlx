# Vietnamese data generation plan

This plan replaces the first PhoMT campaign as the data recipe for the next
VI→EN model. It is intentionally paired with the [training plan](training_plan.md)
and [validation plan](validation_plan.md): data does not ship to training until
the gates here pass, and no model ships on synthetic-data metrics alone.

## Goal and non-goals

The goal is a reproducible training mixture that teaches Vietnamese source
grounding, realistic speech, long-form timing, and English target speech without
recreating the synthetic-domain and short-utterance ceiling of the first run.

This campaign does **not** generate another thousand hours of the same
PhoMT/VieNeu/Kokoro distribution, retrofit old English voices by default, or use
training data as validation data. It also does not treat TTS speaker matching as
a substitute for real Vietnamese source speech.

## What the first campaign established

| Evidence | Consequence |
|---|---|
| 696,243 uploaded pairs, about 1,228 VI hours; the usable Mimi cache contains 694,422 rows / 1,114 hours after the 25 s filter | Raw synthetic scale is no longer the missing variable. |
| Mean usable source duration is about 5.78 s | Sentence-level data does not exercise long context, punctuation pauses, or recovery after silence. |
| Almost all source and target speech is synthetic; real FLEURS is only a small auxiliary set | The phase-1 and phase-2 train/real-val divergence is a domain problem. More passes over the same TTS distribution made it worse. |
| VI voice QA reduced the source pool to 40 voices; only the newer 51% of rows has embedding-matched EN timbre | Voice diversity improved, but half the corpus still has the old independent EN assignment. This is a voice-consistency issue, not the root translation failure. |
| Upload rejects duration ratios outside 0.4–1.8 and the generators rescue non-finite/all-zero audio | Catastrophic waveform failures are controlled. Row-level transcript correctness is not. |
| Cache construction samples one deterministic clip-level delay per id in `[0, 0.5 · source_duration]` | The corpus has timing variation, but not per-sentence timing, punctuation pauses, or long-form alignment. |
| Prefix timing creates many supervised text PAD tokens; one measured effective split is about 45% prefix PAD / 55% content+EOS before weighting | Every cache/alignment variant must report its own prefix-PAD/content/EOS counts. A loss mitigation cannot be evaluated without knowing the data-induced balance. |
| No recorded normalized-text leakage audit exists across training and real validation/test manifests | Split integrity must become a generated artifact, not an assumption. |

These are project measurements. They are not paper claims.

## Target mixture

The next corpus has three explicit strata. Every row records its stratum, source
corpus, license, speaker/voice id, synthesis model, seed, and alignment recipe.

| Stratum | Purpose | Decision |
|---|---|---|
| Real-source ST core | Learn Vietnamese acoustics, accents, microphones, and source grounding | Use every approved public training split before generating more synthetic VI. The first mixed run requires at least 20 accepted hours, 100 speakers, and two recording domains; 100 hours is the next acquisition milestone, not a paper floor. |
| Long-form aligned slice | Teach multi-sentence timing, punctuation pauses, and context | Build 30–120 s examples only from native contiguous recordings or preserved document/speaker segments. The 120 s ceiling matches the [released Hibiki training range](https://github.com/kyutai-labs/hibiki#inference). Never make "long-form" by concatenating unrelated short rows, even from the same speaker. |
| Existing PhoMT synthetic supplement | Preserve text/topic coverage cheaply | Reuse the validated 1,114 h cache as a reservoir, not an epoch that must be exhausted. Keep it separately addressable and cap it at 80% of optimizer exposure; source-aware sampling is required. |

The 20-hour, 100-speaker, 80%, and 100-hour values are project controls. The
first gate is deliberately reachable with vetted public data; the larger target
exists because neither Hibiki paper establishes a universal Vietnamese floor.
At a 20% real-source mixture, stop after at most two real-data passes and rotate
the synthetic reservoir rather than repeating the small real set until all
1,114 synthetic hours have been consumed.

## Real-source acquisition ledger

Availability and terms were checked on 2026-08-03. A corpus name is not approval
to ingest it; the frozen data card records the exact revision and terms accepted.

| Source | Current evidence | Use decision |
|---|---|---|
| FLEURS `vi_vn` | The project already has 6.19 train hours. The [dataset card](https://huggingface.co/datasets/google/fleurs/blob/main/README.md) reports CC-BY-4.0 and speaker-disjoint train/dev/test. | Keep train in the real core. Preserve validation/test exclusively for model evaluation. |
| Common Voice Vietnamese | The live [Common Voice page](https://commonvoice.mozilla.org/vi/languages) reports 24 recorded hours, 422 speakers, and 32% validation progress; current releases are CC0. | Ingest only validated train clips. Preserve `client_id` splits and obey the download agreement, including its re-hosting restriction; publish a fetch manifest, not the audio. |
| VIVOS | The [original catalogue record](https://live.european-language-grid.eu/catalogue/corpus/22131) reports 15 hours and CC-BY-NC-SA-4.0. | Eligible for this non-commercial research project after confirming compatibility with the model/data release. Split by speaker before target generation. Do not rely on mirrors with conflicting license metadata. |
| VLSP/VinBigData 100 h, ViMD 102.56 h, newer conversational corpora | Reported scale is attractive, but access, redistribution, speaker metadata, and license are not yet verified in this repository. | Blocked discovery items. Promote one only after a recorded license and provenance review; do not count its advertised hours toward a launch gate. |

The clearly vetted public pool is therefore measured in tens, not hundreds, of
hours. If it cannot satisfy the first gate after filtering and deduplication,
the correct next action is a licensed acquisition/recording effort, not relaxed
QA or another VieNeu campaign.

### VIVOS source preparation — complete 2026-08-03

The first real-source acquisition phase is reproducible with:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/prepare_vivos.py \
  --root /Volumes/data/datasets/hibiki_vi_v2
```

The command pins `AILAB-VNUHCM/vivos` revision
`3cbfb2502e5e84776b4b778b020a09759f723f52`, verifies the 1,474,408,300-byte
archive against SHA-256
`147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca`, safely
extracts it, and is resumable through the Hugging Face local-download cache.

Dev is drawn only from official train. The exact policy is to sort official
train speaker ids by `SHA-256("hibiki-vi-v2-vivos-dev-v1" + NUL + speaker_id)`
and select the first five: `VIVOSSPK18`, `VIVOSSPK33`, `VIVOSSPK37`,
`VIVOSSPK39`, and `VIVOSSPK41`. Official test has priority over dev, and dev has
priority over train for normalized-transcript leakage. Conflicting official
train rows remain in the source and excluded manifests with
`eligibility_split=excluded_text_overlap`; they are never optimization rows.

| Split | Rows | Hours | Speakers |
|---|---:|---:|---:|
| Eligible train | 9,845 | 12.909465 | 41 |
| Eligible dev | 1,106 | 1.453915 | 5 |
| Preserved official test | 760 | 0.745985 | 19 |
| Excluded transcript overlap | 709 | 0.557707 | 32 |
| Full official source | 12,420 | 15.667071 | 65 |

The generated audit reports zero duplicate ids, missing or out-of-root audio
paths, train/dev/test speaker overlap, normalized-text overlap, and official
split misuse. All 12,420 files are 16 kHz mono 16-bit PCM. Versioned JSONL
manifests live in `/Volumes/data/datasets/hibiki_vi_v2/manifests/` and the full
audit, including manifest hashes, lives at
`/Volumes/data/datasets/hibiki_vi_v2/audits/vivos_3cbfb250_source_v1_audit.json`.
No English translations, target audio, or Mimi caches have been created yet,
and nothing from this phase was uploaded.

## Alignment and target construction

Two paper-derived recipes serve different stages:

1. **Coarse ST alignment (Hibiki-Zero):** for each source sentence, delay the
   corresponding target start by `δᵢ ~ U(0, 0.5 · dᵢ)` and add pauses sampled from
   `U(0, 2 s)` at punctuation. Apply this per sentence, not once per joined clip.
2. **Alignment-aware long-form targets (original Hibiki):** generate 6–8 TTS
   candidates, select primarily by ASR WER and then speaker similarity, and
   constrain target timing from contextual word alignment. The paper's padding
   logit penalty belongs to target generation; it is not a model-loss weight.

The first recipe is the default for the coarse corpus. The second is reserved
for the smaller long-form continuation slice because it is substantially more
expensive. We do not invent the papers' undisclosed thresholds or loss weights.

For real-source rows, the canonical source waveform stays untouched. Because
the current trainer consumes cached Mimi codes, source-noise augmentation must
be materialized as a versioned waveform/cache variant before training; it cannot
be added honestly inside the current cached-code loop. English text comes from a versioned translation step
with its input transcript and model/version retained. English audio is generated
from that text with a source-conditioned or embedding-matched speaker. Direct
source conditioning is preferred; nearest-voice matching must be labeled as an
approximation.

## Build phases and gates

### Phase D0 — freeze splits and provenance

- Normalize VI and EN text, compute content hashes, and remove exact duplicates.
- Reject any training row whose normalized VI or EN text hash overlaps the real
  development or test sets.
- Freeze immutable train/dev/test id lists before target generation.
- Record corpus revision, license, source URL, speaker id where available, and
  the code commit used to build the manifest.

**Gate:** zero known train↔dev/test text overlap; every row has a stable id,
source stratum, corpus revision, and license record.

### Phase D1 — real-source pilot

- Ingest a small representative pilot before the full download/conversion.
- Validate resampling, channels, transcript encoding, speaker metadata, and
  redistribution constraints.
- Run source ASR against the supplied transcript and inspect the error
  distribution by corpus, duration, and speaker. Thresholds are frozen from
  this pilot before bulk processing; they are not chosen after model training.
- Human-audit random rows plus every catastrophic ASR, duration, or waveform
  outlier.

**Gate:** the pilot is intelligible, legally usable, and its transcript-error
tail can be filtered without removing a major speaker or duration slice.

### Phase D2 — translate, synthesize, and align

- Produce versioned EN translations and retain translation confidence or
  review status.
- Generate target candidates, ASR-score them, and keep the selected candidate
  plus selection metrics.
- Build coarse per-sentence timing for the core and alignment-aware timing for
  the long-form slice.
- Keep source and target audio before Mimi encoding so failed rows can be
  regenerated without reconstructing provenance.

**Gate:** every accepted row has transcript, translation, source audio, target
audio, sentence boundaries, durations, and alignment metadata.

### Phase D3 — row QA and corpus audit

Retain the existing hard gates: readable waveform, finite samples, non-zero
signal, and EN/VI duration ratio 0.4–1.8. Add:

- source-ASR error against VI transcript;
- target-ASR error against EN text;
- clipping, DC offset, RMS/speech-duration, and excessive-silence summaries;
- per-speaker, per-corpus, per-duration, and per-voice acceptance rates;
- speaker-similarity score where target voice transfer is claimed;
- manual review of a seeded random sample and every rejected failure class.

**Gate:** zero non-finite/all-zero accepted rows; no unexplained quality collapse
for a corpus, speaker, or length bucket; all thresholds and rejection counts are
published with the dataset version.

### Phase D4 — Mimi cache and release

- Encode every cache in the same PyTorch Mimi backend. The project measured only
  about 42% code agreement for the MLX alternative, so backends must never be
  mixed within a run.
- Preserve real, long-form, FLEURS, and PhoMT caches as separate directories so
  the training sampler can enforce the declared mixture.
- For each cache and alignment variant, report supervised prefix-PAD, content,
  EOS, and ignored tail/batch-pad token counts. Also report the effective
  PAD/content+EOS loss mass at prefix-PAD weights 1.0 and 0.5. The current code
  estimates about 45/55 at 1.0 and 29/71 at 0.5 for the measured variant; that
  accounting is not evidence that 0.5 improves a model.
- Run the existing degenerate-code scan and produce cache indexes with row ids,
  frames, data stratum, alignment version, and source manifest hash.
- Publish immutable cache chunks and verify archive hashes after download and
  extraction.

**Gate:** zero degenerate cache rows, 100% accepted-manifest↔cache id agreement,
and reproducible archive hashes. The first mixed run remains blocked until the
real core passes the 20-hour/100-speaker/two-domain gate. Long-form continuation
also remains blocked until its dedicated validation slice is available.

## What exists and what must be built

Existing commands are useful but do not implement the new campaign by
themselves:

```bash
python training-data/build_voice_bank.py
python training-data/qa_vi_voices.py
python training-data/match_voices.py
python training-data/pipeline.py
python training-data/upload.py
python finetune/cache_phomt_stream.py --device mps --keep-parquet
```

`training-data/pipeline.py` is configured through module constants, not CLI
flags. It generates paired PhoMT TTS; it does not ingest real speech or build
long-form alignment.

Required work, at the owning boundaries:

1. A manifest builder for real speech that emits the existing pair fields plus
   provenance, speaker, corpus, license, and stratum fields.
2. A long-form/alignment builder that owns sentence boundaries, target candidate
   selection, punctuation pauses, and alignment metadata.
3. A row-level QA command that writes machine-readable accept/reject reasons and
   aggregate slice reports. `qa_vi_voices.py` only scores the voice pool.
4. A cache-format revision that retains stratum/alignment provenance in each
   sample and index. Current `hibiki_vn_lora_cache_v1` drops it at training load.
5. Source-aware sampling in `finetune/common.py`; passing multiple `--cache-dir`
   values currently pools every row uniformly and cannot enforce the 80/20 cap.

## Reproducibility artifacts

Each released data version contains:

- immutable train/dev/test manifests and normalized-text overlap report;
- source licenses and corpus revisions;
- translation, TTS, ASR, speaker-embedding, and Mimi model revisions;
- random seeds, voice map, blocklist, timing recipe, and generator config;
- row-level QA table and aggregate acceptance report;
- raw/accepted/rejected counts and hours by stratum, corpus, speaker, and duration;
- supervised prefix-PAD/content/EOS counts and alignment-variant id;
- cache indexes, archive SHA-256 hashes, and the repository commit;
- a short dataset card that states how much source speech is real versus TTS.

## Go/no-go decisions

- **No-go:** another bulk all-synthetic VI campaign before real-source coverage.
- **No-go:** training on data that has not passed split leakage, row QA, and cache
  agreement gates.
- **No-go:** automatic retrofit of the older unmatched EN voices. Execute
  `training-data/RETROFIT_PLAN.md` only if the audio validation suite identifies
  speaker-consistency as a material failure.
- **Go:** coarse SFT only after D0–D4 pass; long-form continuation only after its
  dedicated real-speech validation slice is frozen.
