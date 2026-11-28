"""Depformer slices for a phone student ([lm-npu](../../docs/agent-memory/lm-npu.md#depformer-slices)):
moshi's depformer, but its last slice predicts the last `parallel_codebooks` codebooks of a frame at
once, so a frame takes dep_q - P + 1 sequential steps (9 for 16 codebooks and P = 8) where the
teacher's takes 16. Each step reads the codebook before it, as moshi's do; the last step's
codebooks each read codebooks 0..dep_q - P of their frame, not each other. Delays are the
teacher's (codebooks 1..15 two frames after codebook 0), so a step's codebooks after the first are
one frame, and the slices' steps follow its residual quantizer as the teacher's do.

    trunk out ─► slice 0 (text) ─► cb 0 ─► slice 1 ─► cb 1 ─► ... ─► slice 8 (cb 7) ─► cb 8..15

A config marks such a model with `"audio_heads": "depformer_parallel"` and `"parallel_codebooks"`,
which `common.load_checkpoint_info` takes out before moshi reads the config. Its depformer keys are
moshi's own, the step schedule codebook j -> slice min(j, dep_q - P) (`schedule`), so moshi builds
the slices' weights, the input linears and the 16 output linears; `install` keeps a slice's input
embedding and output norm, not a codebook's, and runs the steps (`_forward_training`,
`depformer_step`). `inference/tools/convert_mlx.py` stages a checkpoint for MLX, and
`export_lm_coreml.py` exports the staging for the phone, all slices in one Core ML model.

`python -m training.student.slices` gives a student new slices: its trunk, text and input
embeddings copied, its audio part (heads or depformer) dropped, its target delays set to the source
stream's (the teacher's), the slices initialized as moshi initializes a depformer (`--seed`), or
copied from a model of the same slice shape (`--audio-from`, the teacher, whose depformer at 1024 x 3
is the slices' but for its depth: `--audio-layers` picks the layers; the last slice's codebooks
8..15 read their output linears from the teacher's).

    python -m training.student.slices --weight weights/student_heads14_cb16_final.safetensors \\
        --config-path weights/student_heads14_cb16/config.json --out-dir weights/student_slices14
"""

from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path
from typing import Any

import torch
from moshi.models.lm import LMGen, LMModel
from moshi.utils.sampling import sample_token

from training.finetune import lm_training
from training.student import heads

KIND = "depformer_parallel"
PARALLEL_KEY = "parallel_codebooks"
# The tensors of a model's audio part, which new slices replace (a heads model's are under depformer).
AUDIO_PARTS = ("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb.", "depformer_norms.", "linears.")


def schedule(dep_q: int, parallel: int) -> list[int]:
    """The step schedule: codebook j's slice."""
    return [min(j, dep_q - parallel) for j in range(dep_q)]


def is_slices(lm: Any) -> bool:
    return getattr(lm, "parallel_codebooks", None) is not None


def install(lm: LMModel, parallel: int) -> LMModel:
    """`lm` (built by moshi from the config) running its depformer as slices: a slice's input
    embedding and output norm kept, the codebooks' after them dropped."""
    slices = lm.dep_q - parallel + 1
    if lm.depformer_weights_per_step_schedule != schedule(lm.dep_q, parallel) or len(lm.depformer_in) != slices:
        raise ValueError(f"a {KIND} model needs the step schedule {schedule(lm.dep_q, parallel)}")
    lm.depformer_emb = lm.depformer_emb[: slices - 1]
    lm.depformer_norms = lm.depformer_norms[:slices]
    lm.parallel_codebooks = parallel
    lm.forward_depformer_hidden = types.MethodType(_forward_hidden, lm)
    lm.forward_depformer_training = types.MethodType(_forward_training, lm)
    return lm


def _apply_depformer_norms(norms: Any, out: tuple[torch.Tensor, ...], B: int, T: int) -> tuple[torch.Tensor, ...]:
    return tuple(norm(x).view(B, T, -1) for norm, x in zip(norms, out, strict=True))


_compiled_apply_depformer_norms = torch.compile(_apply_depformer_norms, dynamic=True)


def _forward_hidden(self: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Normed per-slice outputs `[slices]` of `[B, T, dep_dim]` before `self.linears`, for
    chunked linear+KL distillation (zero full-vocab logit materialization)."""
    B, _, T = sequence.shape
    slices = len(self.depformer_in)
    proj = lm_training.project_depformer_in(  # [slices, B, T, dep_dim]
        self.depformer_in, transformer_out, lm_training.depformer_fp8(self.depformer))
    inputs = [proj[0] + self.depformer_text_emb(sequence[:, 0])]
    for k in range(1, slices):
        token = sequence[:, k + self.audio_offset - 1]
        inputs.append(proj[k] + self.depformer_emb[k - 1](token))
    out = lm_training.run_depformer(self.depformer, torch.stack(inputs, 0).view(slices, B * T, -1))
    return _compiled_apply_depformer_norms(self.depformer_norms, out, B, T)


def _forward_training(self: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
    """LMModel.forward_depformer_training for slices: logits [B, dep_q, T, card] on the delayed
    timeline, teacher-forced (slice k > 0 reads codebook k - 1 of the step)."""
    B, _, T = sequence.shape
    slices = len(self.depformer_in)
    normed = _forward_hidden(self, sequence, transformer_out)
    ar_logits = [self.linears[cb](normed[cb]) for cb in range(slices - 1)]
    # Fuse parallel codebooks (e.g. 8..15) sharing the last slice into one GEMM.
    par_count = self.dep_q - (slices - 1)
    if par_count > 1 and all(self.linears[cb].bias is None for cb in range(slices - 1, self.dep_q)):
        w_par = torch.cat([self.linears[cb].weight for cb in range(slices - 1, self.dep_q)], dim=0)
        par_logits = torch.nn.functional.linear(normed[-1], w_par).view(B, T, par_count, -1).unbind(2)
        return torch.stack([*ar_logits, *par_logits], 1)
    par_logits = [self.linears[cb](normed[-1]) for cb in range(slices - 1, self.dep_q)]
    return torch.stack([*ar_logits, *par_logits], 1)


_depformer_step = LMGen.depformer_step  # heads.py's, which runs moshi's for any non-heads model


def depformer_step(self: LMGen, text_token: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
    """LMGen.depformer_step: a slices model samples its slices in order, the last slice's codebooks
    at once ([B, dep_q]); any other model as before."""
    lm = self.lm_model
    if not is_slices(lm):
        return _depformer_step(self, text_token, transformer_out)
    if self.cfg_coef != 1.0:
        raise NotImplementedError(f"classifier-free guidance with {KIND}")
    slices = len(lm.depformer_in)
    tokens, previous = [], text_token[:, None]
    with lm.depformer.streaming(text_token.shape[0]):
        for k in range(slices):
            embedding = lm.depformer_text_emb if k == 0 else lm.depformer_emb[k - 1]
            out = lm.depformer_norms[k](lm.depformer(lm.depformer_in[k](transformer_out) + embedding(previous)))
            codebooks = range(k, k + 1) if k < slices - 1 else range(k, lm.dep_q)
            logits = torch.stack([lm.linears[cb](out)[:, 0] for cb in codebooks], 1)  # [B, n, card]
            sampled = sample_token(logits.float(), self.use_sampling, self.temp, self.top_k)  # [B, n]
            tokens.append(sampled)
            previous = sampled[:, -1:]
    return torch.cat(tokens, 1)


LMGen.depformer_step = depformer_step


def slices_config(parent: dict[str, Any], dim: int, num_heads: int, layers: int, parallel: int) -> dict[str, Any]:
    """The parent student's config with slices for its audio part and the source's delays (the
    teacher's) for its target's."""
    dep_q = int(parent["dep_q"])
    if not 1 <= parallel < dep_q:
        raise ValueError(f"--parallel {parallel} for {dep_q} codebooks")
    config = {key: value for key, value in parent.items() if key not in (heads.CONFIG_KEY, PARALLEL_KEY)}
    delays = list(parent["delays"])
    config.update({
        heads.CONFIG_KEY: KIND, PARALLEL_KEY: parallel,
        "delays": [delays[0], *delays[1 + dep_q :], *delays[1 + dep_q :]],
        "depformer_dim": dim, "depformer_num_heads": num_heads, "depformer_num_layers": layers,
        "depformer_dim_feedforward": None, "depformer_multi_linear": True,
        "depformer_weights_per_step": True, "depformer_weights_per_step_schedule": schedule(dep_q, parallel),
        "depformer_norm": "layer_norm", "depformer_pos_emb": "none", "depformer_kv_repeat": 1,
        "depformer_low_rank_embeddings": None,
    })
    return config


def audio_source(weight: Path, keys: list[str], layers: list[int], depth: int) -> dict[str, torch.Tensor]:
    """The tensors of `weight` that the student's audio part `keys` copy: the same names (its first
    slices' input linears, embeddings and norms, and the 16 output linears), but slice layer i reads
    the source's depformer layer `layers[i]`."""
    from safetensors import safe_open

    if len(layers) != depth or layers != sorted(set(layers)) or layers[0] < 0:
        raise ValueError(f"--audio-layers needs {depth} increasing layers, got {layers}")

    def source(key: str) -> str:
        match = re.match(r"^depformer\.layers\.(\d+)\.", key)
        return key if match is None else f"depformer.layers.{layers[int(match[1])]}.{key[match.end():]}"

    with safe_open(str(weight), "pt") as f:
        names = set(f.keys())
        missing = [f"{key} ({source(key)})" for key in keys if source(key) not in names]
        if missing:
            raise KeyError(f"{weight.name} has no tensor for student {', '.join(missing[:5])}")
        return {key: f.get_tensor(source(key)) for key in keys}


def main() -> None:
    from safetensors import safe_open

    from training.finetune import common
    from training.paths import DEFAULT_MIMI_WEIGHT, DEFAULT_TOKENIZER, read_json, repo_display_path, require_file

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", type=Path, required=True, help="the parent student's weight (safetensors)")
    parser.add_argument("--config-path", type=Path, required=True, help="the parent student's config")
    parser.add_argument("--out-dir", type=Path, required=True, help="gets config.json and model.safetensors")
    parser.add_argument("--dim", type=int, default=512, help="the slices' width")
    parser.add_argument("--num-heads", type=int, default=8, help="the slices' attention heads")
    parser.add_argument("--layers", type=int, default=2, help="the slices' layers")
    parser.add_argument("--parallel", type=int, default=8, help="codebooks the last slice predicts at once")
    parser.add_argument("--seed", type=int, default=0, help="the slices' initialization")
    parser.add_argument("--audio-from", type=Path, help="copy the slices from this model's audio part (safetensors)")
    parser.add_argument("--audio-layers", help="with --audio-from, its depformer layers to copy, e.g. 0,2,5")
    args = parser.parse_args()
    if (args.audio_from is None) != (args.audio_layers is None):
        parser.error("--audio-from and --audio-layers go together")

    weight = require_file(args.weight, "parent weight")
    config = slices_config(read_json(require_file(args.config_path, "parent config")),
                           args.dim, args.num_heads, args.layers, args.parallel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path, out_weight = args.out_dir / "config.json", args.out_dir / "model.safetensors"
    config.update(moshi_name=out_weight.name, tokenizer_name=common.weight_tokenizer(weight))
    staged = args.out_dir / ".config.json.tmp"  # config.json only once the weight is in
    staged.write_text(json.dumps(config, indent=2) + "\n")

    with safe_open(str(weight), "pt") as f:
        metadata = dict(f.metadata() or {})
        parent = {key: f.get_tensor(key) for key in f.keys() if not key.startswith(AUDIO_PARTS)}
    dtype = parent["text_linear.weight"].dtype
    info = common.load_checkpoint_info(argparse.Namespace(
        hf_repo="kyutai/hibiki-zero-3b-pytorch-bf16", model_weight=weight, mimi_weight=DEFAULT_MIMI_WEIGHT,
        tokenizer=DEFAULT_TOKENIZER, config_path=staged))  # Mimi and the tokenizer are never read here
    torch.manual_seed(args.seed)
    child = common.build_lm(info, "cpu", dtype, load_weight=False).state_dict()
    audio = None if args.audio_from is None else audio_source(
        require_file(args.audio_from, "audio source"), [key for key in child if key.startswith(AUDIO_PARTS)],
        [int(part) for part in args.audio_layers.split(",")], args.layers)
    state = {}
    for key, tensor in child.items():
        if key.startswith(AUDIO_PARTS):
            if audio is not None:
                if audio[key].shape != tensor.shape:
                    raise ValueError(f"student {key} {tuple(tensor.shape)}, its source {tuple(audio[key].shape)}")
                tensor = audio.pop(key).to(dtype)
            state[key] = tensor.contiguous()
            continue
        if key not in parent or parent[key].shape != tensor.shape:
            raise ValueError(f"student {key} {tuple(tensor.shape)} has no parent tensor of that shape")
        state[key] = parent.pop(key).contiguous()
    if parent:
        raise ValueError(f"parent tensors the student has no place for: {sorted(parent)[:5]}")
    record = {"dim": args.dim, "num_heads": args.num_heads, "layers": args.layers,
              "parallel": args.parallel, "parent_student": metadata.get("student")}
    if args.audio_from is None:
        record["seed"] = args.seed
    else:
        record.update(audio_from=repo_display_path(args.audio_from), audio_layers=args.audio_layers)
    metadata["student_of"] = repo_display_path(weight)
    metadata["student"] = json.dumps(record)
    common.save_state(state, out_weight, metadata)
    staged.replace(config_path)
    audio_params = sum(t.numel() for key, t in state.items() if key.startswith(AUDIO_PARTS))
    origin = f"seed {args.seed}" if args.audio_from is None else f"from {args.audio_from.name} layers {args.audio_layers}"
    print(f"student: {len(state)} tensors, {sum(t.numel() for t in state.values()) / 1e9:.3f}B parameters "
          f"({audio_params / 1e6:.1f}M in {config['dep_q'] - args.parallel + 1} slices of {args.dim} x {args.layers}, "
          f"{origin}) -> {repo_display_path(out_weight)}, {repo_display_path(config_path)}")


if __name__ == "__main__":
    main()
