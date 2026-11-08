# Tranche cache

Read when touching `training/tranche/cache.py`: the CUDA build of the real-Vietnamese tranche's
grounded caches, its row gates and their thresholds. The tranche and its other steps are in
[real-vi-tranche](real-vi-tranche.md); the row layout in
[sft-data-caches](sft-data-caches.md#row-layout).
Back to [CONTEXT.md](../../CONTEXT.md) · [Gates](#gates) · [Codes and timing](#codes-and-timing) ·
[Running](#running) · [Scale-up](#scale-up)

`training/tranche/cache.py build <sources>` (CUDA, the training `.venv`) turns each segments shard
whose `text_en`, `en_audio` and `asr_vi` are on the Hub into `<root>/cache/<source>/shard_<shard>.pt`,
pushed to `<source>/` of the cache repo (`ingest.CACHE_REPO`), in `cache_codes`'s format ([sft-data-caches](sft-data-caches.md#row-layout)),
one cache per source (see [Decision](real-vi-tranche.md#decision)). A sample also holds `doc` and
`seq`, so the trainer joins a doc's rows in order ([sft-training](sft-training.md)); the delay
draws use split `train`.

## Gates

- A row is dropped, with the first reason in the shard's `rejected`, when its language is not
  Vietnamese; when its CER is over `--max-cer`; when the ASR hears more nghìn/triệu/tỷ than the
  transcript holds (these three are `filters.vi_gate`); when it was not voiced; when `en_broken`;
  when `en_asr` has Vietnamese letters `text_en` lacks; when it overlaps an evaluation sentence;
  then on `cache_codes`'s own gates (decoded duration, frame cap, CTC alignment score 0.5, token
  timing). The Vietnamese side comes first since 2026-11-07, when voice.py began skipping the
  rows `vi_gate` drops at the default 0.15 (null `en_audio`); shards built before then counted
  rows failing both sides as `en_broken` or "Vietnamese letters". A looser `--max-cer` turns those rows into "not voiced".
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
  row is dropped when the ASR (over `spoken`, so digits count) has more of any one of them than
  the transcript, word by word, so a swapped magnitude drops too (a viVoice transcript's "2,1
  triệu đồng" where the ASR hears tỷ; summed counts kept it): 2.06% of GigaSpeech 2's first
  42,255 rows, 1.45% under the CER cut; 0.2% of PhoAudiobook's and viVoice's, mostly the ASR
  mishearing ("tết tỷ"); word by word adds ~0.13% of PhoAudiobook's. Left as they are (a review,
  2026-11-06): `spoken` reads "20.000đ" or "7.940kg" (digits glued to a unit) without their
  nghìn, 2 kept rows in GigaSpeech 2's 245k; it does not fold "tỉ" into "tỷ", which drops ~14
  rows a source whose transcript writes "tỉ" ("ba tỉ" against the ASR's "3 tỷ"). Changing
  `spoken` would also change the CER `filters asr` stored.
- The full build (2026-11-06, 591 shards, 2.0 h of GPU time) kept 777,090 of 840,520 rows. Each
  gate's share of the source's rows, in the gates' order (a row counts under its first):

  | Source | Rows | Kept | Broken English | VI letters | Language | CER > 0.15 | Magnitudes | Frames | Alignment |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | PhoAudiobook | 290,037 | 98.0% | 0.68% | 0.65% | 0.01% | 0.17% | 0.24% | 0.15% | 0.07% |
  | viVoice | 269,350 | 96.6% | 1.21% | 1.03% | 0.03% | 0.85% | 0.17% | 0.02% | 0.12% |
  | GigaSpeech 2 | 281,133 | 82.7% | 11.07% | 1.91% | 0.02% | 2.61% | 1.15% | 0.03% | 0.48% |

  No row overlapped an evaluation sentence. The trainer plans 134,066 streams from them, 1,760 h
  of Vietnamese source, 26,176 batches an epoch.
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
- Checked end to end on the box (2026-11-06, after the TTS and the ASR server), into a scratch
  `--root` with the push stubbed: PhoAudiobook shards 0–1, viVoice 0, GigaSpeech 2 0. Every
  PhoAudiobook row (2,572) has the trainer's layout (`T = max(en, vi) + 1`, one text EOS at
  `en_frames`, the codec EOS at `vi_frames`, -1 after). Against the CPU build of shard 0's first
  24 rows, shapes, text rows and delays are identical, alignment scores within 0.0021, and
  0.0074% of audio codes differ (Mimi's float32 on CPU against CUDA). `streams` packs the shards
  and plans streams that keep a doc's rows in order: of adjacent pieces, 1,883 are the next `seq`,
  18 skip a dropped row, 61 change doc. transformers reports `wav2vec2.masked_spec_embed` as
  missing from the checkpoint: it only masks in training, so the CTC is unaffected.
- Kept: PhoAudiobook 99.0%, viVoice 96.6%, GigaSpeech 2 83.0% (of its shard 0: `en_broken`
  10.4%, CER 2.7%, Vietnamese letters 2.1%, magnitudes 1.3%, alignment 0.4%, language and frame
  cap 0.06%). GigaSpeech 2's broken English is in [tranche-tts](tranche-tts.md).
- Speed: a warm PhoAudiobook shard (1,299 rows, 9.6 h of Mimi audio, Vietnamese and English) in
  21 s, 1,609× real time, the GPU 74–100% busy; a GigaSpeech 2 shard (19,967 rows, 49.9 h) in
  161 s with the model load, the GPU idle its first ~60 s while the clips decode and 85–95% busy
  after (the build reads and decodes the next shard behind the current one). ~49 GB of GPU at
  its peak. ~2.6 h for the tranche.
- In the full build (2026-11-06) viVoice ran ~1,000–1,170× with the GPU idle 34% of the time:
  two gaps a shard, 2.4–4.2 s after the CTC while `load_next` waited for the next shard's read,
  and 0.6–1.6 s for the save. The read was the Hub download: ~4 MB/s on the box, 10–18 s for a
  viVoice shard's 60 MB (mostly `en_audio`) against ~8 s of GPU work, and parallel downloads got
  no more (one stream; 8 at once reached ~30 MB/s later the same day). `fetch` now takes each
  part's copy under `--root` where the box has one (`ingest.part_file`, which voice and filters
  use too; from GigaSpeech 2 archive 13 on, every part is local), and the save runs in a thread
  while the next shard builds. A review (2026-11-06) found every local `asr_vi` and `en_audio` the
  size of its Hub copy, hashes equal on all `asr_vi` and 12 sampled `en_audio`.

## Scale-up

Before building a larger tranche (the plan's ~10k h: more GigaSpeech 2 and YODAS), optimize the
build first (user, 2026-11-06): at the first tranche's speed, ~1,100–1,600× on the Mimi audio
(both clips, ~2.1× the Vietnamese hours), 10k h of source takes ~13–19 h of GPU. Not done yet:
GigaSpeech 2's 76 archives built at 830× on average (222 s an archive of ~50 h of Mimi audio;
110–292 s, 695–1,510×; 2026-11-06/07), so its ~164 archives left take ~10 h; the user put the optimization
off on 2026-11-08 ("we can optimize cache build later"). In order:

- Keep the inputs on local disk, or download them with `HF_HUB_DISABLE_XET=1`: through Xet the box's
  Hub downloads ran ~4 MB/s (plain HTTP: 143 MB/s with 8 at once, 2026-11-06), and parallel Xet downloads got
  no more, so a build that downloads starves the GPU. Run the TTS and the ASR on the box that
  builds (their outputs stay under `--root`), or move the parts there well before the build.
- Measure where the GPU time goes, Mimi against wav2vec2 (never split yet), with `nvidia-smi`
  every 0.2 s and the gaps lined up with the shard log; aim for under 10% idle.
- Mimi stays float32 with TF32 off (other precisions change the codes, unlike the existing
  caches); `torch.compile` or CUDA graphs on its encoder keep the codes. wav2vec2 may run in
  TF32 or bf16 if word starts stay within a frame of float32 on real rows.
- Per-row costs: GigaSpeech 2 ran at 825× (4 min a 20k-row shard) against PhoAudiobook's 1,478×
  and viVoice's 1,172×; its rows are short (4.3 s), so the main thread's per-row Python (`pending`,
  collecting the alignments, `assemble_codes`, the sample dicts) likely holds the GPU idle longer
  per hour of audio. Measure it before the scale-up, which is mostly GigaSpeech 2, and move it into
  the workers.
- Per-shard fixed costs weigh on small shards (viVoice's 3.8 h of Mimi audio ran ~1,100× against
  PhoAudiobook's ~1,400×): fuller last batches across shard boundaries, or bigger shards.
- After the local-read fix viVoice ran ~1,300–1,400× (~10 s a shard), the GPU idle 17%: ~1.0 s at
  each shard's start (its decodes, queued behind the last shard's alignments, and the `pending`
  loop) and ~0.6 s within it. Queuing the next shard's decodes earlier would close the first.
- CPU: 25 workers on the 26.9-CPU quota averaged 10.5 CPUs but throttled 14% of the 100 ms
  periods in bursts (`/sys/fs/cgroup/cpu.stat`); check the main process is not the one stalled.

