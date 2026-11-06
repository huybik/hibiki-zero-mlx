# Tranche cache

Read when touching `training/tranche/cache.py`: the CUDA build of the real-Vietnamese tranche's
grounded caches, its row gates and their thresholds. The tranche and its other steps are in
[real-vi-tranche](real-vi-tranche.md); the row layout in
[sft-data-caches](sft-data-caches.md#row-layout).
Back to [CONTEXT.md](../../CONTEXT.md) · [Gates](#gates) · [Codes and timing](#codes-and-timing) ·
[Running](#running)

`training/tranche/cache.py build <sources>` (CUDA, the training `.venv`) turns each segments shard
whose `text_en`, `en_audio` and `asr_vi` are on the Hub into `cache/<source>/shard_<shard>.pt` of
its tranche repo, in `cache_codes`'s format ([sft-data-caches](sft-data-caches.md#row-layout)),
one cache per source (see [Decision](real-vi-tranche.md#decision)). A sample also holds `doc` and
`seq`, so the trainer joins a doc's rows in order ([sft-training](sft-training.md)); the delay
draws use split `train`.

## Gates

- A row is dropped, with the first reason in the shard's `rejected`, when `en_broken`; when
  `en_asr` has Vietnamese letters `text_en` lacks; when its language is not Vietnamese; when its
  CER is over `--max-cer`; when the ASR hears more nghìn/triệu/tỷ than the transcript holds; when
  it overlaps an evaluation sentence; then on `cache_codes`'s own
  gates (decoded duration, frame cap, CTC alignment score 0.5, token timing).
- `--max-cer` 0.15 for every source (2026-11-06), from the CER of the first shards:

  | Source | Rows | p90 | p99 | Over 0.10 | Over 0.15 |
  |---|---:|---:|---:|---:|---:|
  | PhoAudiobook (16 shards) | 20,784 | 0.033 | 0.102 | 1.04% | 0.56% |
  | viVoice (30 shards) | 23,040 | 0.061 | 0.151 | 3.58% | 1.02% |
  | GigaSpeech 2 (2 shards) | 42,255 | 0.083 | 0.288 | 7.92% | 4.00% |

  Over 0.15 the transcript and the clip differ by phrases: PhoAudiobook transcripts that repeat
  the clip's last clause (0.17–0.34), punctuation spelled out ("chấm ba chấm"). The English
  translates the transcript, so such a row teaches the model to say what was not spoken. From
  0.10 to 0.15 the corpus is mostly right: on viVoice the ASR mangles English names and loanwords
  ("Jujutsu Kaisen" heard as "Shikai Sen", "Berserk" as "bơm sơn"), and percentages and year
  ranges are spelled otherwise ("30%" against "ba mươi phần trăm"); on PhoAudiobook a word at a
  cut edge or a number's spelling. Of 16 viVoice rows sampled there, one had a wrong fact ("dài
  200 tấn" where the speaker says "dài 282 mét"). 0.10 would drop 2.6% more of viVoice for that.
  From 0.05 to 0.10 the two differ by homophones (sinh/xin) and edge syllables. GigaSpeech 2's
  machine transcripts are the worst: over 0.15 its clips hold whole sentences more than their
  transcript (CER 1–1.9) or numbers lose their magnitude; from 0.10 to 0.15, of 20 rows sampled,
  5 write "một mươi" for "mười" (the same number), 5 leave out fillers ("cái"), and 5 have a wrong
  number.
- Magnitudes: GigaSpeech 2's transcripts drop nghìn, triệu and tỷ from numbers ("hơn một chuyến
  tàu" where the speaker says "hơn một nghìn chuyến", "ba ba trăm linh" for 3,300), at any CER. A
  row is dropped when the ASR (over `spoken`, so digits count) has more of them than the
  transcript: 2.06% of GigaSpeech 2's first 42,255 rows, 1.45% under the CER cut; 0.2% of
  PhoAudiobook's and viVoice's, mostly the ASR mishearing ("tết tỷ").
- Over the whole first tranche (`asr_vi` complete on 2026-11-06, 840,520 rows in 9.13 h), the
  gates that need no English drop, in their order:

  | Source | Rows | Language | CER over 0.15 | Magnitudes | Total |
  |---|---:|---:|---:|---:|---:|
  | PhoAudiobook | 290,037 | 0.01% | 0.48% | 0.17% | 0.66% |
  | viVoice | 269,350 | 0.03% | 0.99% | 0.17% | 1.20% |
  | GigaSpeech 2 | 281,133 | 0.03% | 3.91% | 1.39% | 5.34% |
- The Vietnamese-letter rule drops 118 of 10,392 rows on PhoAudiobook shards 0–7 (1.1%), 49 of
  them broken anyway. Of the rest, about half are lead-ins the TTS spoke in Vietnamese ("Không.
  Being rejected…", "Bước ngoặt khác, managing the farm") and half Vietnamese names the ASR spelled
  otherwise ("Tào Chí Viễn" for "Tào Trí Viễn"). Comparing words with their diacritics folded
  would keep 27 more (0.26%), not worth a second rule.

## Codes and timing

- CTC: transformers' wav2vec2-base-960h in float32 over padded batches, the first GroupNorm's
  statistics taken over each clip's own frames and the encoder given the attention mask, as
  `wav2vec2_mlx` masks. On CPU, 8 clips of a shard batched against each alone keep every argmax,
  their log-probabilities within 2.9e-3 (fp32 rounding: transformers' own fp32 and fp64 forwards
  differ by 3.1e-3); Mimi's batched codes equal the clip-alone ones on all 16 codebooks (moshi's
  Mimi is causal). Mimi as in `mimi_cache_cuda` (float32, TF32 off, 640 s batches). The Viterbi
  runs in the decode workers while Mimi encodes; the next shard is downloaded, read and gated in
  a thread, and its clips decode in the workers behind this shard's alignments.

## Running

- Needs `data/eval` (on the box `.venv/bin/python -m training.prep.eval_sets --sets fleurs
  multimed`, 41 s) and `weights/` ([docs/finetune.md](../finetune.md)). With the TTS and the ASR
  server on the GPU, ~1.3 GB of it is free, so the build runs after them.
