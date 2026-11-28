"""A staged model directory loaded for the MLX runtime: where models are, the quantization its
config.json records, and the LM with its tokenizer, samplers and Mimi codec."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece

from hibiki_mlx.lm.generate import LmGen
from hibiki_mlx.lm.lm import Lm, LmConfig
from hibiki_mlx.lm.sampling import Sampler

ROOT = Path(__file__).resolve().parents[2]  # repo root (inference/hibiki_mlx/ -> ../..)

W = ROOT / "weights"
# Upstream's 3B, staged from weights/ by `convert_mlx.py --bits 4` (README).
UPSTREAM = W / "hibiki-zero-3b-q4"
ASSETS = ROOT / "inference" / "assets"
SAMPLE = ASSETS / "leon.wav"  # French gate clip


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n{hint}")


def resolve_weights_dir(model: str | Path = "3b") -> Path:
    return UPSTREAM if str(model) == "3b" else Path(model)


def quantize(lm: nn.Module, record: dict) -> None:
    """Quantizes `lm` as a config.json `quantization` record says (convert_mlx.py writes it):
    every Linear and Embedding at its `group_size` and `bits`, unless its module path has an
    entry of its own."""

    def spec(path: str, module: nn.Module) -> bool | dict:
        return hasattr(module, "to_quantized") and (dict(record[path]) if path in record else True)

    nn.quantize(lm, record["group_size"], record["bits"], class_predicate=spec)


@dataclass(frozen=True)
class Model:
    """A loaded model directory: the LM, its tokenizer and how config.json's `lm_gen_config`
    samples it."""

    lm: Lm
    config: LmConfig
    tokenizer: sentencepiece.SentencePieceProcessor
    text_sampler: Sampler
    audio_sampler: Sampler
    mimi_path: Path

    def mimi(self) -> rustymimi.Tokenizer:
        """A fresh Mimi codec. Each thread needs its own (one instance cannot be borrowed by two
        at once), and each stream a fresh one: `reset()` leaves state behind (the first frame of
        the next clip came out different)."""
        nq = max(self.config.other_codebooks, self.config.generated_codebooks)
        return rustymimi.Tokenizer(str(self.mimi_path), num_codebooks=nq)

    def generator(self, max_steps: int) -> LmGen:
        """A stream's generator, at most `max_steps` steps, sampling as `lm_gen_config` says."""
        return LmGen(
            model=self.lm,
            max_steps=max_steps,
            text_sampler=self.text_sampler,
            audio_sampler=self.audio_sampler,
        )


def load(weights_dir: Path) -> Model:
    """A staged model directory: config.json names the MLX weights (`moshi_name`), the tokenizer
    they are on (`tokenizer_name`), the sampling (`lm_gen_config`) and, for quantized weights,
    their quantization (`quantization`); the weights load strictly."""
    cfg_path = weights_dir / "config.json"
    _require_file(
        cfg_path,
        "Use a model directory staged by inference/tools/convert_mlx.py (README: upstream's 3B "
        "is staged into weights/hibiki-zero-3b-q4).",
    )
    cfg = json.loads(cfg_path.read_text())
    for key in ("moshi_name", "tokenizer_name", "lm_gen_config"):
        if key not in cfg:
            raise ValueError(f"{cfg_path} has no {key}: stage the model with convert_mlx.py")
    _require_file(weights_dir / cfg["moshi_name"], "The weights config.json names (moshi_name).")
    _require_file(
        weights_dir / cfg["tokenizer_name"],
        "A model directory holds the tokenizer its config.json names "
        "(convert_mlx.py copies it in).",
    )
    lm_config = LmConfig.from_config_dict(cfg)
    lm = Lm(lm_config)
    lm.set_dtype(mx.bfloat16)
    if "quantization" in cfg:
        quantize(lm, cfg["quantization"])
    lm.load_weights(str(weights_dir / cfg["moshi_name"]), strict=True)
    mx.eval(lm.parameters())
    tok = sentencepiece.SentencePieceProcessor(str(weights_dir / cfg["tokenizer_name"]))
    if tok.vocab_size() != cfg["text_card"]:
        raise RuntimeError("Tokenizer vocabulary does not match config.json")
    mimi_path = weights_dir / cfg.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    _require_file(mimi_path, "Download or copy the Mimi checkpoint into the model directory.")
    sampling = cfg["lm_gen_config"]
    return Model(
        lm=lm,
        config=lm_config,
        tokenizer=tok,
        text_sampler=Sampler(top_k=sampling["top_k_text"], temp=sampling["temp_text"]),
        audio_sampler=Sampler(top_k=sampling["top_k"], temp=sampling["temp"]),
        mimi_path=mimi_path,
    )
