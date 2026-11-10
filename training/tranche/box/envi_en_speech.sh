#!/usr/bin/env bash
# en_speech's English-to-Vietnamese caches for the two-way run (two-way.md#two-way-run): round 4e's 500 train shards
# (en_speech_sub.txt) and the 3 validation shards, by cache_codes --direction en-vi into data/cache/en-vi/en-vi-grounded-v2/,
# from the dataset's copy in the tranche repo (en_speech/ of anquachdev/en-vi-speech-20k). The sources come down 25 at a
# time into /workspace/en_speech_src while the batch before builds, and each batch is deleted once built, so ~12 GB of
# the 121 GB are ever on disk. Then the caches go to huybik/hibiki-zero-vi-full-sft at en-vi/en-vi-grounded-v2/.
# A rerun skips the shards built. From the repo root, with HF_TOKEN (huybik) and HF_TOKEN_AN set (restart.sh envi-cache).
set -euo pipefail
export HF_HUB_DISABLE_XET=1  # plain HTTP: ~140 MB/s with 8 at once; Xet throttles to ~4 MB/s
S=/workspace/en_speech_src D=/workspace/en_vi_speech/data OUT=data/cache/en-vi/en-vi-grounded-v2 BATCH=25
mkdir -p $S/en_speech/train $S/en_speech/validation $D
ln -sfn $S/en_speech/train $D/train
ln -sfn $S/en_speech/validation $D/validation

fetch() {  # repo paths or patterns
  HF_TOKEN=$HF_TOKEN_AN .venv/bin/python - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("anquachdev/en-vi-speech-20k", repo_type="dataset", local_dir="/workspace/en_speech_src",
                  allow_patterns=sys.argv[1:], max_workers=8)
PY
}
build() {  # shard paths under data/
  .venv/bin/python -m training.prep.cache_codes --device cuda --direction en-vi --dataset-dir ${D%/data} --out-dir $OUT \
    --shards "$@"
}

fetch "en_speech/validation/*.parquet"
build "validation/*.parquet"

names=()
while read -r shard; do
  name=${shard#shard_}; name=${name%.pt}
  [[ -f $OUT/train/shard_$name.pt ]] || names+=("$name")
done < training/tranche/box/en_speech_sub.txt
echo "$(date -u "+%F %T") ${#names[@]} train shards to build"
(( ${#names[@]} )) && fetch $(printf "en_speech/train/%s.parquet " "${names[@]:0:BATCH}")
for (( i = 0; i < ${#names[@]}; i += BATCH )); do
  batch=("${names[@]:i:BATCH}") next=("${names[@]:i+BATCH:BATCH}") ahead=
  if (( ${#next[@]} )); then fetch $(printf "en_speech/train/%s.parquet " "${next[@]}") & ahead=$!; fi
  build $(printf "train/%s.parquet " "${batch[@]}")
  rm -f $(printf "$S/en_speech/train/%s.parquet " "${batch[@]}")
  [[ -z $ahead ]] || wait $ahead
done

.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
# Into en-vi/en-vi-grounded-v2/ of the cache repo, beside the vi-en en-vi-grounded-v2/; resumable.
HfApi().upload_large_folder(repo_id="huybik/hibiki-zero-vi-full-sft", repo_type="dataset", folder_path="data/cache",
                            allow_patterns=["en-vi/en-vi-grounded-v2/*/shard_*.pt"])
PY
echo "$(date -u "+%F %T") en_speech en-vi done"
