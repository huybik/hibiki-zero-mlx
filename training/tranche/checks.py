#!/usr/bin/env python
"""Check of the tranche's Vietnamese ASR on the box, before the pipeline relies on it
([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#models)).

  asr  /workspace/envs/asr   Qwen3-ASR-1.7B (vLLM backend) on FLEURS dev and the bake-off's clips
                             and references: WER and CER against each set's own transcript, the
                             detected language, speed

It writes <root>/checks/asr.jsonl (one row per input) and prints a summary. From the repo root:
  /workspace/envs/asr/bin/python -m training.tranche.checks asr
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path("/workspace/tranche/bakeoff")
REPO_ROOT = Path(__file__).resolve().parents[2]
ASR = ("Qwen/Qwen3-ASR-1.7B", "7278e1e")


def snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


def inputs(root: Path) -> list[dict]:
    """FLEURS dev rows, then every bake-off clip and reference, each with its transcript."""
    rows = []
    for line in open(REPO_ROOT / "data/eval/fleurs-vi/dev.jsonl"):
        r = json.loads(line)
        rows.append({"set": "fleurs-dev", "id": r["id"], "audio": str(REPO_ROOT / r["vi_audio"]),
                     "text_vi": r["text_vi"], "text_en": r["text_en"]})
    for line in open(root / "clips.jsonl"):
        r = json.loads(line)
        for kind in ("clip", "ref"):
            rows.append({"set": r["source"], "id": f"{r['id']}.{kind}", "audio": r[kind],
                         "text_vi": r[f"{kind}_text"]})
    return rows


def words(text: str) -> list[str]:
    """Lower case, NFC, punctuation dropped: Vietnamese words are its space-separated syllables."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def asr(args: argparse.Namespace) -> None:
    import jiwer
    from qwen_asr import Qwen3ASRModel

    rows = inputs(args.root)
    model = Qwen3ASRModel.LLM(model=snapshot(*ASR), gpu_memory_utilization=0.6,
                              max_inference_batch_size=128, max_new_tokens=512)
    start = time.perf_counter()
    forced = model.transcribe(audio=[r["audio"] for r in rows], language=["Vietnamese"] * len(rows))
    seconds = time.perf_counter() - start
    detected = model.transcribe(audio=[r["audio"] for r in rows], language=None)
    for r, f, d in zip(rows, forced, detected, strict=True):
        ref, hyp = " ".join(words(r["text_vi"])), " ".join(words(f.text))
        r |= {"asr": f.text, "language": d.language, "wer": jiwer.wer(ref, hyp) if ref else None,
              "cer": jiwer.cer(ref, hyp) if ref else None}
    write(args.root / "checks/asr.jsonl", rows)
    print(f"{len(rows)} clips in {seconds:.0f} s (forced Vietnamese)")
    for name in dict.fromkeys(r["set"] for r in rows):
        part = [r for r in rows if r["set"] == name and r["wer"] is not None]
        refs = [" ".join(words(r["text_vi"])) for r in part]
        hyps = [" ".join(words(r["asr"])) for r in part]
        other = sum(r["language"] != "Vietnamese" for r in part)
        print(f"{name:<11} n={len(part):>4}  WER {jiwer.wer(refs, hyps):.3f}  CER {jiwer.cer(refs, hyps):.3f}"
              f"  detected not Vietnamese: {other}")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("step", choices=["asr"])
    args = parser.parse_args()
    asr(args)


if __name__ == "__main__":
    main()
