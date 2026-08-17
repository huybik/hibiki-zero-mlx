# Mobile student model track

This directory owns the CUDA-to-MLX model path described in
[`docs/mobile_student_plan.md`](../docs/mobile_student_plan.md). The immutable
starting shapes are:

- `hibiki_m_12l_ar.json`: the full-model CUDA distillation intermediate;
- `hibiki_m_12l_parallel_v1.json`: the deployable frozen-backbone shape.

Both retain the official Hibiki-M 1B width, tokenizer, Mimi contract, and eight
source plus eight target codebooks. The AR checkpoint is deliberately larger
than one billion parameters; deleting its large depformer when installing
`parallel_v1` is what takes the deployable model below one billion parameters.

Measure either shape without allocating weights:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py receipt \
  --config student/configs/hibiki_m_12l_ar.json
```

Initialize the AR student from an explicitly downloaded official 1B config and
checkpoint. The command rejects every missing, extra, or shape-mismatched tensor
before renaming the frozen parent-layer selection:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/initialize.py \
  --config student/configs/hibiki_m_12l_ar.json \
  --parent-config PATH/TO/OFFICIAL/config.json \
  --parent-weights PATH/TO/OFFICIAL/hibikim-pytorch.safetensors \
  --output RUN/init.safetensors \
  --receipt RUN/initialization_receipt.json
```

A release model pack is accepted only when all config-selected files, the parity
fixture, and both receipts exist and match `manifest.json`:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py manifest PACK_DIR
/opt/homebrew/Caskroom/miniconda/base/bin/python student/contract.py validate PACK_DIR
```
