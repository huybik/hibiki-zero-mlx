#!/usr/bin/env bash
# en_speech's English-to-Vietnamese caches for the two-way run (two-way.md#two-way-run): the 3 validation shards and
# round 4e's 500 train shards (en_speech_sub.txt), by cache_codes --direction en-vi into data/cache/en-vi/en-vi-grounded-v2/,
# from the dataset's copy in the tranche repo (en_speech/ of anquachdev/en-vi-speech-20k). The sources come down 25 at a
# time into /workspace/en_speech_src while the batch before builds, and each batch is deleted once built, so ~12 GB of
# the 121 GB are ever on disk. Then the caches go to huybik/hibiki-zero-vi-full-sft at en-vi/en-vi-grounded-v2/, and it
# fails unless the Hub lists every one. A shard built or packed (streams --drop-shards) is not built again.
# From the repo root, with HF_TOKEN (huybik) and HF_TOKEN_AN set (restart.sh envi-cache).
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
build() {  # shard paths under data/; the cache builds' free-threaded env (envs-min.sh)
  NO_TORCH_COMPILE=1 /workspace/envs/cache/bin/python -m training.prep.cache_codes --device cuda --direction en-vi --dataset-dir ${D%/data} --out-dir $OUT \
    --shards "$@"
}
built() {  # split name: its cache shard, or the pack that outlived it
  [[ -f $OUT/$1/shard_$2.pt || -d $OUT/$1/.packed/shard_$2 ]]
}

fetch "en_speech/validation/*.parquet"
val=()
for f in $S/en_speech/validation/*.parquet; do
  name=$(basename "$f" .parquet)
  built validation "$name" || val+=("validation/$name.parquet")
done
(( ${#val[@]} == 0 )) || build "${val[@]}"

names=()
while read -r shard; do
  name=${shard#shard_}; name=${name%.pt}
  built train "$name" || names+=("$name")
done < training/tranche/box/en_speech_sub.txt
echo "$(date -u "+%F %T") ${#names[@]} train shards to build"
(( ${#names[@]} == 0 )) || fetch $(printf "en_speech/train/%s.parquet " "${names[@]:0:BATCH}")
for (( i = 0; i < ${#names[@]}; i += BATCH )); do
  batch=("${names[@]:i:BATCH}") next=("${names[@]:i+BATCH:BATCH}") ahead=
  if (( ${#next[@]} )); then fetch $(printf "en_speech/train/%s.parquet " "${next[@]}") & ahead=$!; fi
  build $(printf "train/%s.parquet " "${batch[@]}")
  rm -f $(printf "$S/en_speech/train/%s.parquet " "${batch[@]}")
  [[ -z $ahead ]] || wait $ahead
done

.venv/bin/python - <<'PY'
from pathlib import Path

from huggingface_hub import HfApi

# Into en-vi/en-vi-grounded-v2/ of the cache repo, beside the vi-en en-vi-grounded-v2/; only the shards the Hub lacks.
api, repo, prefix = HfApi(), "huybik/hibiki-zero-vi-full-sft", "en-vi/en-vi-grounded-v2"
local = {p.relative_to("data/cache").as_posix() for p in Path("data/cache", prefix).glob("*/shard_*.pt")}
missing = sorted(local - set(api.list_repo_files(repo, repo_type="dataset")))
if missing:
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path="data/cache", allow_patterns=missing,
                      commit_message=f"{prefix}: {len(missing)} en_speech en-vi cache shards")
left = local - set(api.list_repo_files(repo, repo_type="dataset"))
if left:
    raise SystemExit(f"{len(left)} shards are not on the Hub, e.g. {sorted(left)[0]}")
print(f"{len(local)} shards on the Hub at {repo}/{prefix}")
PY
echo "$(date -u "+%F %T") en_speech en-vi done"
