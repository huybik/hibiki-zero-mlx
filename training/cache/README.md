# Grounded caches

Run from the repository root. Generated caches land under the gitignored `data/`. One build
(`build`, on `mlx` or `cuda`) serves every builder: the EN-VI dataset's here (`en_speech`) and the
real-Vietnamese tranche's (`training.tranche.cache`).

## The EN-VI dataset's caches

`training.cache.en_speech` turns the EN-VI dataset's parquet shards into SFT cache shards, one
per source shard, on the cache build every builder shares (`build`): PyAV decoding, wav2vec2
CTC word timing of the English text, then Mimi, in MLX on the Mac (`--device mlx`, the default,
with the conda Python) or on a CUDA GPU (`--device cuda`). Rows whose audio is damaged (it decodes
off its stored duration), that exceed the trainer's 280-frame cap or that fail alignment are
rejected and listed in their shard; any other error stops the run.
A rerun skips the shards already built. Rules and speed are in
[cache-build](../../docs/agent-memory/cache-build.md), the full build in
[sft-data-caches](../../docs/agent-memory/sft-data-caches.md).

```bash
caffeinate -i -s python -m training.cache.en_speech --out-dir /Volumes/data/datasets/en-vi-grounded-v2 \
  --shards 'train/cv27-*.parquet' 'train/libriheavy-*.parquet' 'validation/*.parquet'
```

Beside each build, `publish` with the same `--out-dir` and `--shards` commits the built
shards to the cache repo, the dataset `huybik/hibiki-zero-vi-full-sft`, under
`<out-dir name>/<split>/`, and brings in the shards another machine built, which the build then
skips. The Mac goes top down; a second machine builds bottom up and they meet in between:

```bash
set -a && . ./.env && set +a                            # HF_TOKEN, with write access to the cache repo
O=/Volumes/data/datasets/en-vi-grounded-v2
nohup caffeinate -i python -m training.cache.publish --out-dir $O --follow \
  --shards 'train/cv27-*.parquet' 'train/libriheavy-*.parquet' 'validation/*.parquet' >> $O/publish.log 2>&1 &
```

The second machine needs Apple Silicon (or a CUDA GPU, with `--device cuda` on the commands
below and without `caffeinate`, which is macOS's), this repo's `main`, `weights/` as in
[docs/finetune.md](../../docs/finetune.md), an `HF_TOKEN` with write access to the cache repo, and
~520 GB of disk: the dataset from `anquachdev/common-vi-en` at the revision the Mac's copy
equals (~440 GB), and the whole cache (~75 GB, the Mac's part pulled in too). Name the out-dir as
the Mac's, since its name is the Hub prefix. First check that it builds the Mac's rows: the
counts should match, with any differing rows down to float near-ties (a few residual codes).

```bash
export HF_TOKEN=...                                      # write access to huybik/hibiki-zero-vi-full-sft
D=~/en_vi_speech O=~/en-vi-grounded-v2 C=~/cache-check
S=('train/cv27-*.parquet' 'train/libriheavy-*.parquet' 'validation/*.parquet')
hf download anquachdev/common-vi-en --repo-type dataset --revision 2ce8777d122123609722814f776f0505ea966ca5 \
  --include "data/*" --local-dir $D
python -m training.cache.en_speech --dataset-dir $D --out-dir $C --shards train/cv27-00000.parquet
hf download huybik/hibiki-zero-vi-full-sft --repo-type dataset --local-dir $C/hub \
  --include "en-vi-grounded-v2/train/shard_cv27-00000.pt"
python -c "import torch; a, b = (torch.load(p, weights_only=True)['samples'] for p in ('$C/train/shard_cv27-00000.pt', '$C/hub/en-vi-grounded-v2/train/shard_cv27-00000.pt')); print(len(a), len(b), sum(x['id'] == y['id'] and x['codes'].equal(y['codes']) for x, y in zip(a, b)), 'rows equal')"
mkdir $O
nohup caffeinate -i -s python -m training.cache.en_speech --dataset-dir $D --out-dir $O \
  --shards "${S[@]}" --bottom-up >> $O/build.log 2>&1 &
nohup caffeinate -i python -m training.cache.publish --dataset-dir $D --out-dir $O \
  --shards "${S[@]}" --follow >> $O/publish.log 2>&1 &
```

## Noisy source cache

`training.cache.noisy` copies a clean cache split shard by shard, re-encoding the source
(Vietnamese in a `vi-en` cache, English in an `en-vi` one) of a seeded 60% of the rows under drawn
room reverb, noise, band and level ([noisy-source](../../docs/agent-memory/noisy-source.md)), on
the pod (`--device cuda`, the default) or the Mac (`--device mlx`). `fetch` downloads the file each
shard was built from (an `anquachdev/common-vi-en` parquet, or for a tranche `en-vi` shard its
`en_audio` part in `anquachdev/en-vi-speech-20k`; or reads a local copy, `--dataset-dir`), keeps
only the selected rows' source audio and can run while the GPU trains; `encode` needs the GPU,
waits for what a running fetch has not written yet, and ends with `every shard written`. A rerun
of either skips the shards written.

```bash
C=data/cache/en-vi-grounded-v2/train O=data/cache/en-vi-grounded-v2-noisy/train
nohup ./.venv/bin/python -m training.cache.noisy fetch --cache-dir $C --out-dir $O >> fetch.log 2>&1 &
./.venv/bin/python -m training.cache.noisy encode --cache-dir $C --out-dir $O   # once the GPU is free
# English to Vietnamese, from a local copy of the dataset the cache was built from; another seed
C=data/cache/en-vi/en-vi-grounded-v2/train O=data/cache/en-vi/en-vi-grounded-v2-noisy/train
./.venv/bin/python -m training.cache.noisy fetch --cache-dir $C --out-dir $O --seed 1 --dataset-dir DATASET
./.venv/bin/python -m training.cache.noisy encode --cache-dir $C --out-dir $O
```

