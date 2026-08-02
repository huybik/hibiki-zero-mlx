#!/usr/bin/env bash
# Fresh vast.ai box -> ready for phase-2 VI SFT. Idempotent; run under nohup.
# Downloads 3B weights, restores phase-1 caches/pairs + warm-start checkpoint,
# fetches FLEURS validation wavs for the greedy val128 gate.
set -euo pipefail
source /venv/main/bin/activate
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
uv pip install -q datasets
mkdir -p weights finetune/runs/vi_full phase1

python - <<PY
from huggingface_hub import hf_hub_download
for f in ["config.json","hibiki-pytorch-77f82164@110.safetensors",
          "mimi-pytorch-e351c8d8@125.safetensors","tokenizer_spm_48k_multi6_2.model"]:
    print("weights:", hf_hub_download("kyutai/hibiki-zero-3b-pytorch-bf16", f, local_dir="weights"), flush=True)
for f in ["mimi_caches.tar.gz","model_step055284.safetensors"]:
    print("phase1:", hf_hub_download("huybik/hibiki-zero-vi-full-sft", f, local_dir="phase1"), flush=True)
PY

tar -xzf phase1/mimi_caches.tar.gz -C finetune/
mv -n phase1/model_step055284.safetensors finetune/runs/vi_full/
rm -f phase1/mimi_caches.tar.gz
echo "--- caches:"; ls finetune/cache; echo "--- pairs:"; ls finetune/pairs
python remote_dataset/download_fleurs_vi_en.py --split validation
echo BOOTSTRAP_BOX_DONE
