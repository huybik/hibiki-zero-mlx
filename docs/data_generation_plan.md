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
| FLEURS `vi_vn` | The project has 4.44 train hours (6.19 hours across all splits). The [dataset card](https://huggingface.co/datasets/google/fleurs/blob/main/README.md) reports CC-BY-4.0 and speaker-disjoint train/dev/test. | Keep train in the real core. Preserve validation/test exclusively for model evaluation. |
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

### VIVOS English translation — complete with one exclusion 2026-08-03

`training-data/translate_manifest.py` owns the manifest-to-translation boundary.
It requires `google-genai==2.10.0`, reads `GEMINI_API_KEY` only from a mode-0600
`.env`, builds an atomic deterministic request file, checks for an exact
same-hash upload/job before creating anything, and resumes by recorded job name.
The paid file Batch API request uses stable `gemini-3.6-flash`, seed `20260803`,
`thinking_level=minimal`, `max_output_tokens=256`, no tools or Search, and the
strict response schema `{text_en: string}`. The versioned `vi_en_faithful_v1`
prompt requires faithful VI→EN translation and preservation of meaning, names,
numbers, dates, negation, certainty, and sentence boundaries; it forbids
summarizing, answering, embellishing, or treating source text as instructions.
Its SHA-256 is
`9f034e962444d1e2a18f879ad770826fd857c0202449884e25e05246949c21b3`.

The 120-row pilot selected 40 train, 40 dev, and 40 test rows across four
duration quartiles and 56 speakers. Job
`batches/4okm5uds1c2s73u8f1c8kzi0z8gtz9zp394r` completed 120/120 with one
reported model version (`gemini-3.6-flash`), `STOP`, valid structured output,
and no empty, blocked, error, or changed source-digit rows. All 120 translations
were inspected: all were usable, with 13 non-catastrophic wording, typography,
or name-rendering notes retained in the review TSV. Usage was 19,231 input and
3,042 output tokens with zero thought tokens; estimated Batch cost was $0.025831
and the full projection was $2.520866.

The approval-gated full command was:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/translate_manifest.py \
  /Volumes/data/datasets/hibiki_vi_v2/manifests/vivos_3cbfb250_source_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/manifests/vivos_3cbfb250_source_v1_dev.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/manifests/vivos_3cbfb250_source_v1_test.jsonl \
  --campaign vivos_gemini_3_6_flash_full_v1 \
  --approval-qa /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_gemini_3_6_flash_pilot_v1_qa.json
```

Full job `batches/4hgep22duzolu8tqeob3uix4xilzhfj6x706` processed all 11,711
eligible requests in about 4 minutes 18 seconds. It used 1,873,983 input and
293,711 output tokens, zero thought tokens, and an estimated $2.506903. The
request and response SHA-256 values are respectively
`3392cccdaf1bad4068fe819e4bb858101947fc8df927101b23ab97fb481c61d5` and
`44e13e5859538f3009be7ecbd4b93897a7ef2a4cfc5a0c1a62c487d6903e015b`.

One row, `vivos:train:VIVOSSPK34_023`, returned `PROHIBITED_CONTENT`. It was the
only id submitted to retry job
`batches/o8t8f2qm12k6og6ngk4t1da67qx8v21p128c`, which produced the same block;
it is therefore an explicit translation exclusion and was not routed through a
fallback. The accepted target manifests contain:

| Split | Rows | Manifest SHA-256 |
|---|---:|---|
| Train | 9,844 | `d2276dcda8b664ca918dd53d215b11b159da98fd817fec89d7cb3701f6bc92fb` |
| Dev | 1,106 | `6fae77d42d6580fc0c36754ce284acb26a3be34bc17de4378979a549f727579d` |
| Test | 760 | `6d134ca259d1f08453737c060e8ab9485ef784b3032d71a8dc7408aba631b652` |

Every accepted row preserves the original source fields and records source and
target hashes, source-manifest hash, request hash, prompt, requested/returned
model, job and file names, response id, usage, finish reason, and safety
metadata. An independent audit found zero provenance or hash disagreements, and
an idempotent finalization reproduced the same three target hashes. Batch state,
raw requests/responses, QA JSON, and human-review TSV live under
`/Volumes/data/datasets/hibiki_vi_v2/{batches,qa,targets}`. Only immutable MLX
v1/v2 pilot English audio has been generated; no Mimi cache or VIVOS artifact
has been published.

### VIVOS timbre-preserving TTS pilots

`training-data/synthesize_vivos.py prepare` is a standard-library-only boundary.
It requires the accepted train and dev translation manifests, verifies the
selected source WAV hashes, and writes an immutable plan for eight pinned speakers: five
train and three dev; four female and four male; alternating short/long
targets. Each speaker has one pinned Vietnamese reference clip, one English
target, two fixed Qwen replicate seeds (`20260803`, `20260804`), and one
matched-Kokoro baseline. The official test split stays sealed:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/synthesize_vivos.py prepare \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_dev.jsonl \
  --kokoro-voice-map /Volumes/data/datasets/voice_bank/vi_to_en_voices.json \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1
```

Generation is intentionally a separate CUDA boundary with no CPU/MPS fallback.
Use a fresh environment containing `qwen-tts==0.1.1`, its CUDA PyTorch stack,
FlashAttention 2, NumPy, and SoundFile. The plan pins
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` revision
`fd4b254389122332181a7c3db7f27e918eec64e3`, ICL cloning (reference audio plus
VI transcript), and the exact sampling config. One in-memory clone prompt is
reused per speaker. Generation writes WAVs atomically and updates its
provenance JSONL after every output. The plan stores dataset-relative input
paths and relative output paths so it can move to a CUDA host. Stage the 16
verified source/reference WAVs under the same relative paths and point
`--dataset-root` at that staging root:

```bash
python training-data/synthesize_vivos.py generate \
  /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/pilot_plan.jsonl \
  --dataset-root /workspace/hibiki_vi_v2 --device cuda:0
```

Generate the eight CPU-pinned matched-Kokoro controls from the same plan in an
environment with `kokoro==0.9.4` and SoundFile. This pins Kokoro-82M revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`, the voice-map hash, exact blended
voice and speed, and model/voice weight hashes:

```bash
python training-data/synthesize_vivos.py generate-kokoro \
  /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/pilot_plan.jsonl
```

Scoring is also CUDA-only and imports its dependencies lazily. The QA environment
requires `transformers==4.57.3`, `sacrebleu==2.6.0`, and `scipy==1.16.2`, plus
CUDA PyTorch, NumPy, and SoundFile. It pins Whisper large-v3-turbo and WavLM
speaker-verification revisions, writes one QA sidecar per WAV,
`row_metrics.jsonl`, and one aggregate `gate_report.json`. Manual review is a
TSV with `candidate_id`, `status` (`pass`/`fail`), `prompt_leak` (`yes`/`no`), and
`notes`. Qwen candidate ids are the `pilot_id` values in the plan; each Kokoro
candidate id is `<target_id>|kokoro`:

```bash
python training-data/qa_vivos_tts.py \
  /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/pilot_plan.jsonl \
  /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/generation.jsonl \
  /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/kokoro_generation.jsonl \
  --out-dir /workspace/hibiki_vi_v2/tts/vivos_qwen3_tts_pilot_v1/qa \
  --dataset-root /workspace/hibiki_vi_v2 \
  --manual-review <pilot-review.tsv> --device cuda:0
```

The frozen hard gate requires all 16 Qwen outputs and eight Kokoro controls to
be readable, finite, non-zero, unclipped, within the 0.4–1.8 target/source
duration band, under 50% total silence and two seconds leading/trailing silence,
and fully reviewed. Qwen additionally permits no WER above 50%, automatic
reference-only transcript trigram match, or audible prompt leak. Quality is
calibrated rather than guessed: Qwen aggregate WER may trail matched Kokoro by
at most three absolute points, its median WavLM source/reference cosine must
exceed Kokoro, and it must win the cosine comparison for at least six of eight
speakers. The report includes gender, split, duration, and seed slices. Missing
review is a `no_go`, not an implicit pass.

#### Apple-Silicon MLX pilot v1 — immutable no-go

The Mac path uses a separate immutable namespace and never rewrites or relabels
the CUDA plan above. MLX v1 is preserved by its versioned registry entry;
`prepare-mlx` still reproduces plan SHA-256
`79ce642714c74280855487bcf1a81968d84d17c9b3198101bc324bf97a23a22a`.
It pins `mlx-audio==0.4.7` from tag commit
`2c9461f5d8315fa8e7013ab2729495b2bb83d384`, bf16 conversion
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` revision
`a6eb4f68e4b056f1215157bb696209bc82a6db48`, and source Qwen revision
`fd4b254389122332181a7c3db7f27e918eec64e3`. It verifies every snapshot file
hash and seeds each row with `mx.random.seed`. In ICL mode MLX-Audio internally
raises requested repetition penalty 1.05 to effective 1.5 and caches encoded
reference codes/text in the shared model; it does not create a clone-prompt
object. These facts are retained in row provenance.

V1 generated all 16 MLX candidates and eight Kokoro controls but is a `no_go`.
MLX aggregate WER was 0.2990 versus Kokoro 0.0098, outside the frozen
three-point margin. Timbre transfer was otherwise strong: median WavLM cosine
was 0.9336 versus 0.5394 and MLX won seven of eight speakers. The exception was
SPK26 (0.2718 versus 0.5237): both seeds copied its Vietnamese reference before
the English target, producing 8 and 11 reference-only trigram matches and WER
2.7778. No human acceptance was recorded. V1 artifacts are evidence and must
not be regenerated, relabeled, or edited.

MLX synthesis and pinned PyTorch QA require separate environments:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda create -y -n vivos-mlx python=3.13
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-mlx python -m pip install \
  "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git@2c9461f5d8315fa8e7013ab2729495b2bb83d384" soundfile
/opt/homebrew/Caskroom/miniconda/base/bin/conda create -y -n vivos-qa-mps python=3.13
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-qa-mps python -m pip install \
  torch==2.13.0 transformers==4.57.3 sacrebleu==2.6.0 scipy==1.16.2 soundfile
```

#### MLX pilot v2 — immutable no-go

V2 keeps the same eight target ids, two seeds, model revisions, snapshot hashes,
Kokoro controls, and frozen QA thresholds. Only two synthesis inputs change:
SPK26 uses reference `vivos:train:VIVOSSPK26_093`, and requested temperature is
0.7. MLX-Audio's effective ICL repetition penalty remains 1.5.

V2 generated all 16 MLX candidates and eight controls and is also a `no_go`.
The new reference removed the v1 failure: automatic prompt-leak matches were
zero and MLX won the timbre comparison for all eight speakers. Aggregate WER
was 0.0539216 versus Kokoro 0.0098039, still 0.0441176 worse and outside the
frozen 0.03 margin. The only catastrophic row was SPK13 seed `20260803` at WER
0.6667. V2 artifacts are immutable evidence.

#### MLX pilot v3 — immutable comparison no-go; Qwen manually approved

V3 preserves the exact v2 speaker/target/reference tuple (including
`VIVOSSPK26_093`), seeds, model/package revisions, snapshot hashes, effective
ICL repetition penalty, Kokoro controls, and frozen QA thresholds. Its only
change is requested temperature 0.7 to 0.8. The schema is
`hibiki_vivos_qwen3_tts_mlx_pilot_v3`; artifacts belong only under
`vivos_qwen3_tts_mlx_pilot_v3`.

V3 plan SHA-256 is
`006a9415311a220ab63a5ff37d7ac61ac4b047f43bd98d757ee5fe2614401174`.
All 16 Qwen rows passed automatic row gates, with zero prompt-leak matches,
median speaker cosine 0.9440091 versus Kokoro 0.5616511, and 8/8 timbre wins.
Aggregate Qwen WER was 0.0490196 versus Kokoro 0.00980392: the 0.0392157 gap
still exceeds the frozen 0.03 margin. All 24 per-file manual reviews remained
incomplete, so the immutable gate report correctly remains `no_go`.

The user subsequently judged the English files very good, explicitly selected
Qwen, and authorized the full campaign. This is a model-level manual waiver of
the retained aggregate comparison no-go. It is not evidence that all 24 files
were reviewed and does not relabel or modify `gate_report.json`.

Prepare the distinct 8-speaker/16-output v3 plan:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/synthesize_vivos.py prepare-mlx-v3 \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_dev.jsonl \
  --kokoro-voice-map /Volumes/data/datasets/voice_bank/vi_to_en_voices.json \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3
```

Before generation, audit all eight target clips and all eight clone references.
The v3 report proves the plan hash, target/reference roles and bijection, audio
and text hashes, accepted-manifest provenance, and exact 16-source coverage. It
reports pinned-Whisper WER/CER and waveform/speaker/duration slices but freezes
no numeric source threshold. V2's report cannot be reused because audit
attestations are bound to the exact plan path, schema, and hash:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-qa-mps \
  python training-data/qa_vivos_source.py \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_dev.jsonl \
  --pilot-plan /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/pilot_plan.jsonl \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_pilot_v3 --device mps
```

Pause here and manually listen to every one of the eight clone references while
reading its Vietnamese transcript. The automated audit validates artifacts; it
does not grant human acceptance. Only after this source-reference review may v3
generation start. `generate-mlx` requires and records the audit report
attestation in every output row:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-mlx \
  python training-data/synthesize_vivos.py generate-mlx \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/pilot_plan.jsonl \
  --source-audit-report /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_pilot_v3/audit_report.json \
  --dataset-root /Volumes/data/datasets/hibiki_vi_v2 --device mps
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n phomt-data \
  python training-data/synthesize_vivos.py generate-kokoro \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/pilot_plan.jsonl
```

Score v3 on MPS in explicit float32/eager mode with every existing frozen gate
unchanged:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-qa-mps \
  python training-data/qa_vivos_tts.py \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/pilot_plan.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/mlx_generation.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/kokoro_generation.jsonl \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/qa \
  --source-audit-report /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_pilot_v3/audit_report.json \
  --dataset-root /Volumes/data/datasets/hibiki_vi_v2 \
  --manual-review <pilot-review.tsv> --device mps
```

The commands above are the immutable v3 record. The explicit user waiver permits
proceeding to full-campaign preparation with Qwen while preserving the `no_go`
report and incomplete per-file review fact.

#### Full VIVOS source audit and reference freeze v1

Before any full synthesis, audit the exact accepted train+dev manifests: 9,844
train plus 1,106 dev rows, 10,950 total across 46 split-contained speakers. Full
mode freezes schema `hibiki_vivos_source_asr_mps_full_v1` and output directory
name `vivos_source_asr_mps_full_v1`. It uses atomic per-row resume sidecars and
publishes immutable `row_metrics.jsonl` and `audit_report.json` only after exact
coverage. Every row retains manifest/audio/text/source/model/runtime provenance.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-qa-mps \
  python training-data/qa_vivos_source.py \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_dev.jsonl \
  --full \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_full_v1 \
  --device mps
```

The full source waveform gate requires readable, finite, nonzero audio;
clipping ratio at most 0.0001; RMS at least 0.0001; silence ratio at most 0.50;
leading and trailing silence at most 2 seconds; and measured/manifest duration
agreement within 0.00001 seconds. WER above 0.50 is a manual-review flag, not an
automatic rejection. Resolve every flagged id in a TSV with columns
`id`, `status` (`pass`/`fail`), and `notes` before freezing references.

The reference freeze selects exactly one row per speaker from that completed
audit. Candidates must remain in their speaker's train/dev split, be 3–8
seconds, pass every waveform gate, have WER at most 0.20, and have no unresolved
source review. Selection order is `(asr_wer, asr_cer, abs(duration-4), id)` and
the command stops if any speaker has no candidate:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python \
  training-data/freeze_vivos_references.py \
  /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_full_v1/audit_report.json \
  --source-review <source-review.tsv> \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1
```

If the full audit flags no WER rows, omit `--source-review`. The immutable
outputs are `reference_map.jsonl` and `reference_map_report.json`; each selected
row carries exact WAV, VI-text, and source-audit-row hashes plus split, speaker,
corpus revision, archive, and license provenance. Both full artifacts now exist
and pass their frozen contracts. The source audit retains 162 review flags,
while every selected reference independently passes the reference thresholds.

#### Full VIVOS MLX v3 generation campaign

Prepare schema `hibiki_vivos_qwen3_tts_mlx_full_v1` only from the exact frozen
train/dev manifests, full source audit, 46-speaker reference map, and retained
v3 `no_go` gate report:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python \
  training-data/synthesize_vivos_full.py prepare-mlx-full \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_train.jsonl \
  /Volumes/data/datasets/hibiki_vi_v2/targets/vivos_gemini_3_6_flash_full_v1_dev.jsonl \
  --out-dir /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1 \
  --dataset-root /Volumes/data/datasets/hibiki_vi_v2 \
  --source-audit-report /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_full_v1/audit_report.json \
  --reference-map /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/reference_map.jsonl \
  --reference-report /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/reference_map_report.json \
  --pilot-gate-report /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v3/qa/gate_report.json
```

This immutably writes `campaign_config.json`, `approval_override.json`, and
`generation_plan.jsonl` for 9,844 train plus 1,106 dev rows; test stays sealed.
The approval records the user's model-level waiver of only the aggregate
Kokoro comparison and incomplete manual review. It preserves the pilot
decision and does not claim all 24 pilot files were reviewed.

Generate attempt 0 serially on Apple Metal. Rows are ordered by speaker then
split/id to reuse MLX-Audio's internal ICL prompt cache. Every completed WAV is
atomically written before its immutable sidecar; resume validates all source,
reference, campaign, model-snapshot, sidecar, and output hashes before writing.

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-mlx \
  python training-data/synthesize_vivos_full.py generate-mlx-full \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/generation_plan.jsonl \
  --dataset-root /Volumes/data/datasets/hibiki_vi_v2 \
  --device mps --attempt 0
```

Attempt 1 is reserved for an explicit later retry-id JSONL containing one
`{"id":"..."}` object per selected row:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n vivos-mlx \
  python training-data/synthesize_vivos_full.py generate-mlx-full \
  /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_v3_full_v1/generation_plan.jsonl \
  --dataset-root /Volumes/data/datasets/hibiki_vi_v2 \
  --device mps --attempt 1 --retry-ids <retry-ids.jsonl>
```

Do not select retries, run target QA, build Mimi caches, or upload artifacts in
this phase. `generation_attempts.jsonl` is only the deterministic assembly of
validated immutable attempt sidecars.

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
