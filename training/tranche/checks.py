#!/usr/bin/env python
"""Checks of the tranche's Vietnamese ASR and VI→EN translation on the box, before the pipeline
relies on them ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#models)).

  asr  /workspace/envs/asr   Qwen3-ASR-1.7B (vLLM backend) on FLEURS dev and the bake-off's clips
                             and references: WER and CER against each set's own transcript, the
                             detected language, speed
  mt   /workspace/envs/mt    Hy-MT2-30B-A3B-FP8 (vLLM): FLEURS dev's Vietnamese sentences against
                             their English (chrF, BLEU; FLEURS shares FLORES's sentences, which MT
                             models may have trained on), then the first --rows rows of each
                             source's segments in the tranche repos, each with up to two earlier
                             rows of its doc as background, timed with the GPU otherwise idle: the
                             cost per million source characters at --price and the box-hours per
                             1,000 h of speech, to weigh against Muse on OpenRouter

Each writes <root>/checks/<step>.jsonl (one row per input) and prints a summary. From the repo root:
  /workspace/envs/asr/bin/python -m training.tranche.checks asr
  /workspace/envs/mt/bin/python -m training.tranche.checks mt --max-num-seqs 512
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
MT = ("tencent/Hy-MT2-30B-A3B-FP8", "b69671c")  # Tencent's own FP8 weights
# The card's settings for the 30B-A3B model, and its plain and background-information prompts.
MT_SAMPLING = {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}
# Each adds that a segment cut from speech may stop mid-sentence: the card's prompt alone made the
# model complete such fragments, inventing what the speaker had not yet said.
FRAGMENT = (" The text is a segment cut from speech and may start or stop mid-sentence: translate "
            "exactly what it says, leaving an unfinished sentence unfinished, without completing, "
            "adding or dropping anything.")
MT_PROMPT = ("Translate the following text into English." + FRAGMENT + " Note that you should only "
             "output the translated result without any additional explanation:\n\n{text}")
MT_CONTEXT_PROMPT = ("[Background Information]\n{background}\n\nPlease translate the following text "
                     "into English, taking the provided background information into consideration."
                     + FRAGMENT + "\n\n[Source Text]\n{text}")
SEGMENTS = {"pho": "huybik/hibiki-zero-vi-real-restricted", "giga": "huybik/hibiki-zero-vi-real-restricted",
            "vivoice": "huybik/hibiki-zero-vi-real"}
BACKGROUND_ROWS = 2


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


def corpus_rows(per_source: int) -> list[dict]:
    """The first rows of each source's segments in order, each with up to BACKGROUND_ROWS earlier
    rows of its doc as background (a doc's rows are consecutive within a shard)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows = []
    for source, repo in SEGMENTS.items():
        taken: list[dict] = []
        for path in sorted(fs.glob(f"datasets/{repo}/{source}/segments/*.parquet")):
            with fs.open(path, "rb", cache_type="none") as fh:  # only these columns, not the audio
                taken += pq.ParquetFile(fh).read(columns=["id", "doc", "text", "duration_s"]).to_pylist()
            if len(taken) >= per_source:
                break
        if len(taken) < per_source:
            raise SystemExit(f"{source}: {len(taken)} segments in {repo}, fewer than {per_source}")
        taken = taken[:per_source]
        for k, r in enumerate(taken):
            earlier = [p["text"] for p in taken[max(0, k - BACKGROUND_ROWS):k] if p["doc"] == r["doc"]]
            rows.append({"set": source, "id": r["id"], "doc": r["doc"], "duration_s": r["duration_s"],
                         "text_vi": r["text"], "background": " ".join(earlier)})
    return rows


def mt(args: argparse.Namespace) -> None:
    import sacrebleu
    from vllm import LLM, SamplingParams

    # FLEURS dev holds 361 readings of 149 sentences: translate each sentence once.
    fleurs, seen = [], set()
    for line in open(REPO_ROOT / "data/eval/fleurs-vi/dev.jsonl"):
        r = json.loads(line)
        if r["text_vi"] not in seen:
            fleurs.append({"set": "fleurs-dev", "id": r["id"], "text_vi": r["text_vi"],
                           "text_en": r["text_en"], "background": ""})
            seen.add(r["text_vi"])
    corpus = corpus_rows(args.rows)
    llm = LLM(model=snapshot(*MT), max_model_len=4096, gpu_memory_utilization=0.9,
              max_num_seqs=args.max_num_seqs, seed=1234)  # a per-request seed samples row by row
    params = SamplingParams(**MT_SAMPLING, max_tokens=1024)

    def translate(rows: list[dict]) -> tuple[float, int, int]:
        messages = [[{"role": "user", "content": (MT_CONTEXT_PROMPT if r["background"] else MT_PROMPT).format(
            text=r["text_vi"], background=r["background"])}] for r in rows]
        start = time.perf_counter()
        outputs = llm.chat(messages, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        for r, o in zip(rows, outputs, strict=True):
            r |= {"mt": o.outputs[0].text.strip(), "finish": o.outputs[0].finish_reason}
        return (seconds, sum(len(o.prompt_token_ids) for o in outputs),
                sum(len(o.outputs[0].token_ids) for o in outputs))

    translate(fleurs)  # also the warm-up
    seconds, tokens_in, tokens_out = translate(corpus)
    write(args.root / "checks/mt.jsonl", fleurs + corpus)
    hyps, refs = [r["mt"] for r in fleurs], [[r["text_en"] for r in fleurs]]
    print(f"FLEURS dev ({len(fleurs)} sentences): chrF {sacrebleu.corpus_chrf(hyps, refs).score:.1f}  "
          f"BLEU {sacrebleu.corpus_bleu(hyps, refs).score:.1f}")
    chars = sum(len(r["text_vi"]) for r in corpus)
    hours = sum(r["duration_s"] for r in corpus) / 3600
    cut = sum(r["finish"] != "stop" for r in fleurs + corpus)
    print(f"{len(corpus)} rows ({hours:.1f} h of speech, {chars / 1e6:.2f}M characters) in {seconds:.0f} s at "
          f"max_num_seqs {args.max_num_seqs}: {tokens_in / seconds:.0f} input and {tokens_out / seconds:.0f} "
          f"output tokens/s; ${seconds / 3600 * args.price / chars * 1e6:.3f} per million source characters "
          f"and {seconds / hours * 1000 / 3600:.1f} box-hours per 1,000 h at ${args.price}/h; {cut} cut short")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("step", choices=["asr", "mt"])
    parser.add_argument("--rows", type=int, default=7000, help="mt: rows from each source")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="mt: vLLM's batch")
    parser.add_argument("--price", type=float, default=1.4, help="mt: the box's $ per hour")
    args = parser.parse_args()
    {"asr": asr, "mt": mt}[args.step](args)


if __name__ == "__main__":
    main()
