#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_PAIRS_DIR,
    DEFAULT_TOKENIZER,
    read_json,
    read_pair_file,
    repo_display_path,
    require_file,
    resolve_repo_path,
)

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
CACHE_FORMAT = "hibiki_vn_lora_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Mimi audio codes and English text tokens for vi->en LoRA training."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS_DIR / "train.jsonl",
        help="Pair file from finetune/build_pairs.py (.jsonl or .csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT / "train",
        help="Output directory for cache shards.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device for Mimi encoding. Use cpu only if you really want slow caching.",
    )
    parser.add_argument("--shard-size", type=int, default=32, help="Samples per shard.")
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to cache, 0 means all.")
    parser.add_argument(
        "--target-delay-ratio",
        type=float,
        default=0.5,
        help=(
            "Deterministically sample English target delay in [0, ratio * vi_duration_s]. "
            "Set 0 to disable coarse alignment."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic delays.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing shards.")
    return parser.parse_args()


def require_runtime_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import sentencepiece
        import sphn
        import torch
        from moshi.models import loaders
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc.name}") from exc
    return torch, sphn, sentencepiece, loaders


def check_device(torch: Any, device_name: str) -> Any:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but torch.backends.mps is not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def read_audio(path: Path, sphn: Any, torch: Any, device: Any, left_pad_s: float = 0.0) -> Any:
    wav, sr = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{path} loaded at {sr} Hz, expected {SAMPLE_RATE} Hz")
    wav = torch.from_numpy(wav).float()
    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.ndim != 2:
        raise ValueError(f"{path} has unsupported audio shape {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if left_pad_s:
        pad = int(round(left_pad_s * SAMPLE_RATE))
        if pad <= 0:
            raise RuntimeError(f"Non-zero left_pad_s produced no samples: {left_pad_s}")
        wav = torch.nn.functional.pad(wav, (pad, 0))
    return wav[None].to(device)


def encode_audio(
    path: Path, mimi: Any, sphn: Any, torch: Any, device: Any, left_pad_s: float = 0.0
) -> Any:
    wav = read_audio(path, sphn, torch, device, left_pad_s=left_pad_s)
    with torch.no_grad():
        codes = mimi.encode(wav)
    if codes.ndim != 3 or codes.shape[0] != 1:
        raise RuntimeError(f"Mimi returned unexpected codes shape for {path}: {tuple(codes.shape)}")
    return codes[0].cpu().long()


def text_tokens(text: str, tokenizer: Any) -> list[int]:
    eos_id = int(tokenizer.eos_id())
    if eos_id < 0:
        raise RuntimeError("Tokenizer has no EOS id; cannot build text targets.")
    return list(tokenizer.encode(text, out_type=int)) + [eos_id]


def assemble_codes(
    torch: Any,
    row: dict[str, str],
    vi_codes: Any,
    en_codes: Any,
    tokens: list[int],
    cfg: dict[str, Any],
    text_start: int,
) -> Any:
    n_q = int(cfg["n_q"])
    dep_q = int(cfg["dep_q"])
    source_q = n_q - dep_q
    card = int(cfg["card"])
    text_card = int(cfg["text_card"])
    text_pad_id = int(cfg["existing_text_padding_id"])
    zero_id = -1

    if vi_codes.shape[0] < source_q:
        raise RuntimeError(
            f"Vietnamese Mimi codes have {vi_codes.shape[0]} codebooks, need {source_q}"
        )
    if en_codes.shape[0] < dep_q:
        raise RuntimeError(f"English Mimi codes have {en_codes.shape[0]} codebooks, need {dep_q}")
    if any(token < 0 or token >= text_card for token in tokens):
        raise RuntimeError(f"Text token out of range for id={row['id']}")

    vi_codes = vi_codes[:source_q]
    en_codes = en_codes[:dep_q]
    text_len = len(tokens)
    target_len = int(en_codes.shape[1])
    source_len = int(vi_codes.shape[1])
    total_frames = max(text_start + text_len, target_len, source_len + 1)
    if total_frames <= 0:
        raise RuntimeError(f"Empty cache sample for id={row['id']}")

    codes = torch.full((1 + n_q, total_frames), zero_id, dtype=torch.int32)
    codes[0].fill_(text_pad_id)
    codes[0, text_start : text_start + text_len] = torch.tensor(tokens, dtype=torch.int32)
    codes[1 : 1 + dep_q, :target_len] = en_codes.to(torch.int32)

    source_start = 1 + dep_q
    codes[source_start : source_start + source_q, :source_len] = vi_codes.to(torch.int32)
    source_eos_idx = min(source_len, total_frames - 1)
    codes[source_start : source_start + source_q, source_eos_idx] = card
    if source_eos_idx + 1 < total_frames:
        codes[source_start : source_start + source_q, source_eos_idx + 1 :] = zero_id

    return codes


def target_delay_s(row: dict[str, str], ratio: float, seed: int) -> float:
    if ratio < 0:
        raise ValueError("--target-delay-ratio must be non-negative")
    if ratio == 0:
        return 0.0
    max_delay_s = ratio * float(row["vi_duration_s"])
    rng = random.Random(f"{seed}:{row['split']}:{row['id']}")
    return rng.uniform(0.0, max_delay_s)


def shard_path(out_dir: Path, shard_index: int) -> Path:
    return out_dir / f"shard_{shard_index:05d}.pt"


def save_shard(torch: Any, payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def write_index(torch: Any, out_dir: Path) -> None:
    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "split",
                "shard",
                "frames",
                "vi_frames",
                "en_frames",
                "text_tokens",
                "target_delay_s",
                "target_delay_frames",
                "vi_audio",
                "en_audio",
            ],
        )
        writer.writeheader()
        for path in sorted(out_dir.glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu")
            for sample in payload["samples"]:
                writer.writerow(
                    {
                        "id": sample["id"],
                        "split": sample["split"],
                        "shard": path.name,
                        "frames": sample["frames"],
                        "vi_frames": sample["vi_frames"],
                        "en_frames": sample["en_frames"],
                        "text_tokens": sample["text_tokens"],
                        "target_delay_s": f"{sample['target_delay_s']:.6f}",
                        "target_delay_frames": sample["target_delay_frames"],
                        "vi_audio": sample["vi_audio"],
                        "en_audio": sample["en_audio"],
                    }
                )


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    torch, sphn, sentencepiece, loaders = require_runtime_deps()
    device = check_device(torch, args.device)

    cfg = read_json(args.config_path)
    mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    tokenizer_path = require_file(args.tokenizer, "text tokenizer")
    pairs = read_pair_file(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError(f"No pairs to cache from {args.pairs}")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Mimi on {device} from {repo_display_path(mimi_weight)}")
    num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
    mimi = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks, device=device)
    if int(cfg["card"]) != int(mimi.cardinality):
        raise RuntimeError(
            f"Config card={cfg['card']} does not match Mimi cardinality={mimi.cardinality}"
        )
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    shard_count = math.ceil(len(pairs) / args.shard_size)
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        rows = pairs[start : start + args.shard_size]
        out_path = shard_path(out_dir, shard_index)
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing {repo_display_path(out_path)}")
            continue

        samples: list[dict[str, Any]] = []
        for row in rows:
            vi_audio = resolve_repo_path(row["vi_audio"])
            en_audio = resolve_repo_path(row["en_audio"])
            if not vi_audio.is_file():
                raise FileNotFoundError(f"Missing Vietnamese audio for id={row['id']}: {vi_audio}")
            if not en_audio.is_file():
                raise FileNotFoundError(f"Missing English audio for id={row['id']}: {en_audio}")

            delay_s = target_delay_s(row, args.target_delay_ratio, args.seed)
            delay_frames = int(round(delay_s * FRAME_RATE))
            vi_codes = encode_audio(vi_audio, mimi, sphn, torch, device)
            en_codes = encode_audio(en_audio, mimi, sphn, torch, device, left_pad_s=delay_s)
            tokens = text_tokens(row["text_en"], tokenizer)
            codes = assemble_codes(torch, row, vi_codes, en_codes, tokens, cfg, delay_frames)
            samples.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "codes": codes,
                    "frames": int(codes.shape[1]),
                    "vi_frames": int(vi_codes.shape[1]),
                    "en_frames": int(en_codes.shape[1]),
                    "text_tokens": len(tokens),
                    "target_delay_s": delay_s,
                    "target_delay_frames": delay_frames,
                    "vi_audio": repo_display_path(vi_audio),
                    "en_audio": repo_display_path(en_audio),
                    "text_en": row["text_en"],
                    "text_vi": row["text_vi"],
                }
            )

        payload = {
            "format": CACHE_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "config": {
                "n_q": int(cfg["n_q"]),
                "dep_q": int(cfg["dep_q"]),
                "card": int(cfg["card"]),
                "text_card": int(cfg["text_card"]),
                "existing_text_padding_id": int(cfg["existing_text_padding_id"]),
            },
            "samples": samples,
        }
        save_shard(torch, payload, out_path)
        print(f"Wrote {len(samples)} samples -> {repo_display_path(out_path)}")

    write_index(torch, out_dir)
    print(f"Index: {repo_display_path(out_dir / 'index.csv')}")


if __name__ == "__main__":
    main()
