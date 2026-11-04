#!/usr/bin/env python
"""Checks of the tranche's Vietnamese ASR and VI→EN translation models on the box, before the
pipeline relies on them ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#models)).

  asr  /workspace/envs/asr   Qwen3-ASR-1.7B (vLLM backend) on FLEURS dev and the bake-off's clips
                             and references: WER and CER against each set's own transcript, the
                             detected language, speed
  mt   /workspace/envs/mt    Hy-MT2-30B-A3B (vLLM) on FLEURS dev's Vietnamese text against its
                             English (chrF, BLEU; FLEURS shares FLORES's sentences, which MT models
                             may have trained on) and on the bake-off clips' transcripts, kept for
                             reading; speed

Both write <root>/checks/<step>.jsonl (one row per input) and print a summary. From the repo root:
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
MT = ("tencent/Hy-MT2-30B-A3B", "d3ead4d")
# The card's settings for the 30B-A3B model.
MT_SAMPLING = {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}
MT_PROMPT = ("Translate the following text into English. Note that you should only output the "
             "translated result without any additional explanation:\n\n{text}")


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


def mt(args: argparse.Namespace) -> None:
    import sacrebleu
    from vllm import LLM, SamplingParams

    rows = [r for r in inputs(args.root) if r["set"] == "fleurs-dev" or r["id"].endswith(".clip")]
    llm = LLM(model=snapshot(*MT), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(**MT_SAMPLING, max_tokens=512, seed=1234)
    messages = [[{"role": "user", "content": MT_PROMPT.format(text=r["text_vi"])}] for r in rows]
    start = time.perf_counter()
    outputs = llm.chat(messages, params)
    seconds = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    for r, o in zip(rows, outputs, strict=True):
        r["mt"] = o.outputs[0].text.strip()
    write(args.root / "checks/mt.jsonl", rows)
    fleurs = [r for r in rows if r["set"] == "fleurs-dev"]
    hyps, refs = [r["mt"] for r in fleurs], [[r["text_en"] for r in fleurs]]
    print(f"{len(rows)} texts in {seconds:.0f} s, {tokens / seconds:.0f} output tokens/s")
    print(f"FLEURS dev: chrF {sacrebleu.corpus_chrf(hyps, refs).score:.1f}  "
          f"BLEU {sacrebleu.corpus_bleu(hyps, refs).score:.1f}")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("step", choices=["asr", "mt"])
    args = parser.parse_args()
    {"asr": asr, "mt": mt}[args.step](args)


if __name__ == "__main__":
    main()
