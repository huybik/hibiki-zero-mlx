# Vietnamese tokenizer swap

Read when touching `training/vi_tokenizer/`, the text tokenizer, or two-way (EN→VI) text.
Back to [CONTEXT.md](../../CONTEXT.md) · [Why](#why) · [How it builds](#how-it-builds) ·
[Running](#running) · [Results](#results) · [Starting a checkpoint](#starting-a-checkpoint) ·
[Still to change](#still-to-change)

## Why

The base tokenizer (`tokenizer_spm_48k_multi6_2.model`) spends ~4.1 pieces per Vietnamese word,
~43% of them byte fallbacks, so Vietnamese text needs ~15 tokens/s of speech against the text
stream's 12.5 Hz, before any PAD. Two-way speech (plan stage 7,
[vi_en_parity_plan](../vi_en_parity_plan.md)) needs Vietnamese text that keeps up. The swap gives
Vietnamese letters and syllables their own pieces in slots English never uses, so `text_card`
stays 48,000 and the text embedding, the head and every runtime keep their shapes: only the
tokenizer file and the new rows' weights change.

## How it builds

- The model is a SentencePiece unigram with an identity normalizer: input must be NFC.
- Slots: NORMAL pieces no `text_en` row of the EN-VI dataset uses, safest first: pieces Vietnamese
  never uses either, then accented foreign words, then non-dictionary strings, then English
  dictionary words (`english_words.txt.gz`, macOS's `/usr/share/dict/words`), then single
  characters. The lowest-scoring piece stays, since the minimum score sets the unknown penalty.
- New pieces: the Vietnamese letters the vocab lacks, then the `text_vi` word forms that save the
  most tokens (`▁core` after a space, bare `core` after punctuation, in their own case).
- Scores: shortest first, each new piece scores at least 0.1 above the best other split of its
  own text, kept inside the old NORMAL range, so the unknown penalty and user symbols keep their
  scores. Most land at the floor (`min + 0.01`), since their other splits go through bytes.
- English stays identical: a superset (the base plus the new letters and the top 16k word forms
  appended) is encoded over every `text_en` row once, and each new piece that changes any row is
  excluded. Then each model is built and checked over every row again, excluding further
  culprits, until no English row differs. A new piece that re-splits an excluded surface (bare
  `và` taking ` và` once `▁và` is out) is left out as it is built.
- Each model comes with `<model>_init.json`: for every replaced slot, the base tokenizer's split
  of the new piece. A new row starts as the mean of those ORIGINAL rows of `text_emb`,
  `depformer_text_emb` and `text_linear`, read from a copy of the old tables, since some of them
  are bytes or are themselves replaced slots.

## Running

On the box, CPU only, from the repo root:

```text
python -m training.vi_tokenizer.build run                    # the 6k model, ~7 min plus fetching
python -m training.vi_tokenizer.build run 4000 8000 --sample 20   # a smoke test of other sizes
```

- `run` is the only entry point, and it builds models in `candidates/` under its folder, moving
  each up only after it passes, so a model under its final name has passed the English check.
  Each run starts from no exclusions, and the sizes of one run share them: the 6k model of
  `run 6000 8000` also leaves out a piece that only the 8k model's English check caught.
- The texts: only `text_en`, `text_vi` and `vi_duration_s` of `anquachdev/common-vi-en` (pinned),
  by exact range requests, one small parquet per shard, into
  `/workspace/vi_tokenizer/texts/<revision>/`, kept for later runs. With `cache_type="none"` and
  32 threads it fetches ~3 shards/s without 429s; the default 1 MB readahead over-reads ~40× and
  ran at 0.24 shards/s.
- The other files land in `/workspace/vi_tokenizer/`, or `sample<K>/` under it for
  `--sample K` (K evenly spaced train shards, every validation shard): `excluded.json`,
  `superset_16000.model`, the models with their init maps, and `verify_*.json` (per model:
  Vietnamese tokens per word, byte share, tokens/s of speech, English rows that differ and the
  pieces that changed them).
- It needs `protobuf` beside `sentencepiece` for the model proto (`training/h100.sh setup`).
- The corpus passes run `--workers` processes, by default the container's CPU quota
  (`training.paths.cpu_budget`), not `nproc`, which counts the host's cores (224 on a box with
  a 27-CPU quota).

## Results

The run of 2026-11-04 on the box took ~7 minutes once the texts were fetched. Its models, init
maps, `excluded.json` and verify reports are also in `/Volumes/data/datasets/vi_tokenizer/`.

- The 16k superset changed 892,464 of 3,619,472 English train rows, through 6,156 new pieces
  (`▁s` alone 117k rows, then `▁OF`, `▁hung`, `▁ve`). The first 6k model changed one more row
  (`▁Cô`, in "Cône sur Loire"), the second none: 6,157 pieces are excluded.
- `tokenizer_spm_48k_multi6_2_vi6k.model` holds 98 letters, 3,931 lowercase, 1,225 capitalized
  and 238 all-caps words, and 508 bare pieces (no `▁`). English is identical on all 3,619,472
  train and 9,522 validation rows.
- Vietnamese text, train (validation):

  | | Base | 6k swap |
  |---|---:|---:|
  | Tokens per word | 4.13 (4.10) | 1.28 (1.28) |
  | Byte pieces | 42.5% (42.6%) | 0.11% (0.08%) |
  | Words in one piece | 4.9% (5.9%) | 88.1% (87.7%) |
  | Tokens/s of speech, mean / p99 | 15.0 / 22.0 (13.8 / 21.7) | 4.7 / 6.5 (4.3 / 6.4) |
  | With one PAD per word, mean / p90 | 18.7 / 23.0 (17.2 / 22.4) | 8.3 / 9.9 (7.7 / 9.5) |
  | Rows over 12.5 tokens/s | 81.6% (66.6%) | 0.003% (0) |

- 20 pieces stay out because they would re-split an excluded surface (`và`, `còn`, `Các`,
  `Minh`, …): a few Libriheavy rows hold Vietnamese text. Dropping those rows from the English
  check would bring ~1.16 tokens/word, which is not worth an English change.
- The build is deterministic: the same pinned dataset gives the same files. 4k gives 1.29 and 8k
  1.28 tokens/word on the prototype's sample, so 6k is the size.

## Starting a checkpoint

`python -m training.vi_tokenizer.init_weights --model-weight <a base-tokenizer checkpoint>
--init-map weights/tokenizer_spm_48k_multi6_2_vi6k_init.json --out <new>` writes the start of a
run on the swap, which then trains with `--model-weight <new> --tokenizer
weights/tokenizer_spm_48k_multi6_2_vi6k.model`. Each of the 6,000 replaced rows of `text_emb`,
`depformer_text_emb` and `text_linear` becomes the mean of its `init_from_ids` rows, read from
the old tables (some of those are replaced slots too); every other tensor is copied. The output
records `text_tokenizer` in its metadata, as every training checkpoint does since 2026-11-10,
and the tool refuses an input on any tokenizer but the base.

- On round 3's model (2026-11-10, the H200's CPUs, 47 s; 12 s on the H100 box): only those
  three tensors and 6,000 rows changed; a Vietnamese sentence went 48 → 18 tokens.
- Right after the swap the VI→EN model writes some of the new Vietnamese pieces into its
  English: in 8 of 96 evaluation rows, 32 of 2,351 tokens (a new row starts as the mean of the
  pieces it merges, so its head row scores near theirs). Its FLEURS dev chrF on 40 rows was
  44.5, healthy. The pilot's first validation shows whether training clears it.
- Evaluation is not reproducible run to run on one seed (5 of 96 predictions equal), so compare
  the swap's effect over a set's chrF, not row by row.

## Still to change

The trainers check caches and weights against `--tokenizer`
([two-way](two-way.md#tokenizer-checks)). The two-way runtimes
([plan stage 7](../vi_en_parity_plan.md#7-two-way-speech)) still need:

- The MLX and Swift bundles ship the new file.
- The streaming decoders (`hibiki_mlx/pipeline.py`, `main.py`, Swift `Vocabulary.text()`)
  print one piece at a time and must assemble UTF-8 from byte pieces.
