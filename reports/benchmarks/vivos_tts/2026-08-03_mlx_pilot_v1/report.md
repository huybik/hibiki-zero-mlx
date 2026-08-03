# VIVOS MLX TTS pilot v1

## Outcome

Decision: **no-go**.

The Apple-Silicon path completed 16 Qwen3-TTS MLX voice-clone outputs and eight
matched Kokoro controls for eight train/dev speakers. The plan SHA-256 is
`79ce642714c74280855487bcf1a81968d84d17c9b3198101bc324bf97a23a22a`.

Qwen preserved source timbre substantially better than Kokoro, but it failed
the frozen intelligibility gate. Both `VIVOSSPK26` replicates copied Vietnamese
reference speech before the English target, exceeded the duration-ratio bound,
and had catastrophic WER. Manual review was not performed because the automated
gate had already failed.

## Aggregate metrics

| Metric | Qwen MLX | Kokoro | Gate/result |
|---|---:|---:|---|
| English ASR WER | 0.299020 | 0.009804 | Qwen trails by 0.289216; maximum 0.03 |
| Median speaker cosine | 0.933584 | 0.539382 | pass |
| Speaker cosine wins | 7/8 | — | pass; minimum 6/8 |
| Automatic prompt-leak trigrams | 19 | — | fail |

Excluding the failed `VIVOSSPK26` pair for diagnosis only, Qwen WER is
0.059140 versus Kokoro 0.010753. That diagnostic still misses the frozen
three-point margin, so removing only the failed row would not make v1 pass.

## Runtime

| Backend | Rows | Output audio | Generation time | Real-time factor |
|---|---:|---:|---:|---:|
| Qwen3-TTS MLX bf16 | 16 | 97.200 s | 175.237 s | 1.803 |
| Kokoro CPU | 8 | 41.775 s | 4.293 s | 0.103 |

The first model download and load time are excluded from per-row generation
timings. Qwen used `mlx-audio==0.4.7` at commit
`2c9461f5d8315fa8e7013ab2729495b2bb83d384` and model revision
`a6eb4f68e4b056f1215157bb696209bc82a6db48`.

## Reproduction artifacts

- Pilot directory: `/Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_pilot_v1`
- Gate report: `qa/gate_report.json`
- Row metrics: `qa/row_metrics.jsonl`
- Source audit: `/Volumes/data/datasets/hibiki_vi_v2/qa/vivos_source_asr_mps_pilot_v1/audit_report.json`
- Implementation commits: `2fbce15`, `490a66c`

V2 is a separately preregistered remediation. It keeps every target, seed,
model revision, and frozen QA gate, changes requested temperature from 0.9 to
0.7, and replaces only the pathological `VIVOSSPK26_300` clone reference with
source-ASR-clean `VIVOSSPK26_093`.
