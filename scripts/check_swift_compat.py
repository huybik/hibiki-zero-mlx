#!/usr/bin/env python
"""Validate a staged artifact the way moshi-swift / moshi_mlx.run_inference loads it.

No device (iPhone) needed: it reproduces the exact stock loader path on the Mac —
  LmConfig.from_config_dict(config) -> Lm -> set_dtype(bf16)
  -> nn.quantize(bits=4, group_size=32, predicate=weight.shape[-1] % 32 == 0)
  -> load_weights(strict=True)
(see moshi-mlx/moshi_mlx/run_inference.py). A strict gs32 q4 load succeeding IS the
compatibility proof: any other group size or a shape/name mismatch throws.

  python scripts/check_swift_compat.py [artifact_dir]     # default weights/hibiki-m-mlx-q4

Prints PASS/FAIL per check and a summary; exit 0 iff every check passes.
"""
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece

from moshi_mlx import models
from moshi_mlx.run_inference import quantization_predicate

ROOT = Path(__file__).resolve().parent.parent

# Keys read by LmConfig.from_config_dict via direct data[...] access (no default) —
# absent => KeyError at build time; plus the three loader-required file pointers.
REQUIRED_LOADER_KEYS = ["moshi_name", "mimi_name", "tokenizer_name"]
REQUIRED_CONFIG_KEYS = [
    "dim", "num_heads", "num_layers", "causal", "layer_scale", "context",
    "max_period", "positional_embedding", "text_card", "card", "n_q", "delays",
    "dep_q", "depformer_dim", "depformer_num_heads", "depformer_num_layers",
    "depformer_pos_emb",
]
# Hibiki-Zero deltas the vendored fork honours (have defaults, but wrong values
# silently break audio) — reported for visibility, not gated on presence.
HIBIKI_DELTA_KEYS = ["hidden_scale", "kv_repeat"]


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def check_config(art: Path, results: list) -> dict | None:
    cfg_path = art / "config.json"
    if not cfg_path.exists():
        results.append(("config.json present", False, str(cfg_path)))
        return None
    cfg = json.loads(cfg_path.read_text())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    missing_loader = [k for k in REQUIRED_LOADER_KEYS if k not in cfg]
    ok = not missing and not missing_loader
    detail = "all required architecture + loader keys present"
    if missing or missing_loader:
        detail = f"MISSING config={missing} loader={missing_loader}"
    results.append(("config keys (from_config_dict + loader)", ok, detail))

    moshi = cfg.get("moshi_name", "")
    q4_ok = isinstance(moshi, str) and moshi.endswith(".q4.safetensors")
    results.append((
        "moshi_name -> .q4.safetensors (triggers gs32 q4 loader)",
        q4_ok,
        f"moshi_name={moshi!r}" + ("" if q4_ok else "  (stock loader only q4-loads a *.q4.safetensors)"),
    ))
    deltas = ", ".join(f"{k}={cfg.get(k)}" for k in HIBIKI_DELTA_KEYS)
    results.append(("hibiki deltas honoured (informational)", True, deltas))
    return cfg


def check_q4_weights(art: Path, cfg: dict, results: list) -> None:
    moshi = cfg.get("moshi_name", "")
    wpath = art / moshi
    if not wpath.exists():
        results.append(("q4 weights present", False, str(wpath)))
        return

    # (a) Header scan: every quantized tensor triple (.weight/.scales) must be
    # 4-bit (U32 packing 8 vals/word) at group_size 32 -> scales cols == in/32.
    hdr = _header(wpath)
    bases = [k[:-7] for k in hdr if k.endswith(".scales")]
    gsizes = set()
    bad = []
    for base in bases:
        w, s = hdr[base + ".weight"], hdr[base + ".scales"]
        if w["dtype"] != "U32":
            bad.append(f"{base}: weight dtype {w['dtype']} (not U32/4-bit)")
            continue
        in_features = w["shape"][1] * 8  # 4-bit: 8 values per uint32 word
        gs = in_features // s["shape"][1]
        gsizes.add(gs)
        if gs != 32:
            bad.append(f"{base}: group_size {gs}")
    gs_ok = gsizes == {32} and not bad
    results.append((
        "q4 weights group_size == 32 (moshi-swift hardcodes gs32)",
        gs_ok,
        f"{len(bases)} quantized tensors, group_sizes={sorted(gsizes)}"
        + ("" if gs_ok else f"  bad: {bad[:3]}"),
    ))

    # (b) Definitive: reproduce run_inference.py's build + strict gs32 q4 load.
    try:
        lm_config = models.LmConfig.from_config_dict(cfg)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        nn.quantize(model, bits=4, group_size=32, class_predicate=quantization_predicate(32))
        model.load_weights(str(wpath), strict=True)
        mx.eval(model.parameters())
        results.append((
            "strict gs32 q4 load (== run_inference.py path)",
            True,
            f"{wpath.name}: all tensor names+shapes match a gs32 q4 build",
        ))
    except Exception as e:  # naming/shape/group-size mismatch surfaces here
        results.append((
            "strict gs32 q4 load (== run_inference.py path)", False,
            f"{type(e).__name__}: {e}"[:300],
        ))


def check_sidecars(art: Path, cfg: dict, results: list) -> None:
    tok = art / cfg.get("tokenizer_name", "")
    if tok.exists():
        try:
            sp = sentencepiece.SentencePieceProcessor(str(tok))
            results.append(("tokenizer loads", True, f"{tok.name}: {sp.vocab_size()} pieces"))
        except Exception as e:
            results.append(("tokenizer loads", False, f"{type(e).__name__}: {e}"[:200]))
    else:
        results.append(("tokenizer present", False, str(tok)))

    mimi = art / cfg.get("mimi_name", "")
    if mimi.exists():
        try:
            lm_config = models.LmConfig.from_config_dict(cfg)
            nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
            rustymimi.Tokenizer(str(mimi), num_codebooks=nq)
            results.append(("mimi codec loads", True, f"{mimi.name}: num_codebooks={nq}"))
        except Exception as e:
            results.append(("mimi codec loads", False, f"{type(e).__name__}: {e}"[:200]))
    else:
        results.append(("mimi present", False, str(mimi)))


def main() -> None:
    art = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "weights" / "hibiki-m-mlx-q4"
    print(f"=== moshi-swift compat check: {art} ===\n")
    results: list[tuple[str, bool, str]] = []
    cfg = check_config(art, results)
    if cfg is not None:
        check_q4_weights(art, cfg, results)
        check_sidecars(art, cfg, results)

    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    passed = sum(ok for _, ok, _ in results)
    total = len(results)
    all_ok = passed == total
    print(f"\nsummary: {passed}/{total} checks passed -> {'PASS' if all_ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
