#!/usr/bin/env python
"""VI→EN translation of the tranche's segments on the box: Index-Translate-35B-A3B in vLLM, in this
process, on the GPU ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)).

Reads each segments shard's id, doc and text from its tranche repo on the Hub (only those columns, no
audio), translates each row as its own prompt, with up to CONTEXT_ROWS earlier rows of its doc as
context, and writes <root>/<source>/text_en/<shard>.parquet (id, text_en) beside the segments'
layout; `ingest push` uploads it. A transcript all in capitals (GigaSpeech 2's) goes in lowercased:
kept, the model answers in capitals, which the TTS spells out. The first try is greedy; a reply cut
at max_tokens, still in Vietnamese, holding letters of another script or far too long is tried
again sampled, up to TRIES times; a row failing every try keeps text_en null, and a rerun tries
the null rows again (and only those: a written row may already be voiced). It takes the GPU memory
voice and the ASR server hold, so they take turns. From the repo root, in the env /workspace/envs/mt (`envs.sh`):

  python -m training.tranche.translate giga --shards 51:
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training.paths import exclusive
from training.prep.translate_en_speech import broken
from training.tranche.ingest import REPO, ROOT, SOURCES

# Chosen on the box against Index-Translate-9B, Hy-MT2-30B-A3B and Muse (real-vi-tranche, Models).
# Apache-2.0; BF16 weights, quantized to FP8 as they load.
MODEL = ("IndexTeam/Index-Translate-35B-A3B-preview", "7e4029e0df842502cb009cf493d6467c247043b3")
MAX_TOKENS = 512  # a row's English takes ~20 tokens, the longest a few hundred
TRIES = 6  # per row and run: greedy, then sampled; a failing row holds its whole shard back, and retries cost seconds
CONTEXT_ROWS = 2
# The short form scored as the long one on the bake-off (FLEURS dev chrF 64.8 against 64.7, archive
# 50 against Muse 68.7 against 69.1, on Index-Translate-9B) with 21% fewer prompt tokens.
PROMPT = ("Translate this Vietnamese speech segment into English. It may start or stop mid-sentence: translate "
          "exactly what it says, without completing, adding or dropping anything. Use normal capitalization and "
          "punctuation, Vietnamese names without diacritics, no brackets or notes. Output only the translation.")
SCHEMA = pa.schema([("id", pa.string()), ("text_en", pa.string())])
GROUP = 4  # shards translated together: one generate call keeps the GPU's batch full across a shard's end
# A run stops once more rows than this share of those it has tried fail every try (past the first
# MIN_ROWS): a systematic failure would otherwise write every shard null.
MAX_FAILED, MIN_ROWS = 0.01, 5_000
# Frequent Vietnamese words, each with a diacritic: two of them in a reply mean it was left (partly) in
# Vietnamese. Words that are also name syllables stay out: ASCII ones (trong, khi: "Nguyen Phu Trong"
# is right), and như, lại, từ, which a name the model leaves with its diacritics holds ("Như Xuân").
VIETNAMESE_WORDS = {"và", "của", "là", "được", "những", "các", "này", "với", "không", "có", "một", "đã", "để",
                    "thì", "người", "đến", "về", "cũng", "nhiều", "sẽ", "đó", "phải", "còn", "rằng", "bị"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def bad(vi: str, en: str | None) -> bool:
    """A reply cut at max_tokens, empty, holding letters of a script its Vietnamese lacks, left in
    Vietnamese, or far longer than its row (the model now and then translates the context rows too:
    1 in 2,000 rows of the check; English runs ~1.2× the Vietnamese's characters)."""
    if en is None or not en.strip() or broken(vi, en) or len(en) > 2.5 * len(vi) + 20:
        return True
    words = re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", en).lower()).split()
    return sum(w in VIETNAMESE_WORDS for w in words) >= 2


def prompt(texts: list[str], docs: list[str], k: int) -> str:
    """Row k's prompt, with up to CONTEXT_ROWS earlier rows of its doc (only rows of the same doc:
    viVoice's rows are docs of their own)."""
    context = [texts[j] for j in range(max(0, k - CONTEXT_ROWS), k) if docs[j] == docs[k]]
    background = ("\n\nContext, earlier in the recording (do not translate):\n" + "\n".join(context)
                  if context else "")
    return f"{PROMPT}{background}\n\nSegment:\n{texts[k]}"


def engine(gpu_memory: float):
    """Prefix caching off: on this hybrid (linear-attention) model it never hit, and its aligned state
    blocks cut the sequences that fit by ~3×. Images and video off, so requests skip the processor."""
    from huggingface_hub import snapshot_download
    from vllm import LLM

    return LLM(model=snapshot_download(MODEL[0], revision=MODEL[1]), quantization="fp8",
               gpu_memory_utilization=gpu_memory, max_model_len=4096, max_num_seqs=512,
               max_num_batched_tokens=16384, enable_prefix_caching=False,
               limit_mm_per_prompt={"image": 0, "video": 0})


def translate(llm, vi: list[str], prompts: list[str], again: list[bool]) -> tuple[list[str | None], list[str]]:
    """-> each row's English (None where it fails every try) and its last reply. Each try is one
    generate call over the rows still without one: greedy first, then sampled at Qwen3.5's
    non-thinking settings, seeded afresh each try (the engine's own seed is fixed) so that a rerun
    draws anew; a row that failed in an earlier run (`again`) is sampled from its first try. Stops
    the run when more than half the rows fail a try: a systematic failure."""
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)
    en: list[str | None] = [None] * len(vi)
    last = [""] * len(vi)
    todo = list(range(len(vi)))
    for attempt in range(TRIES):
        sampled = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.8, top_k=20,
                                 seed=int.from_bytes(os.urandom(4)))
        params = [sampled if attempt or again[k] else greedy for k in todo]
        outputs = llm.chat([[{"role": "user", "content": prompts[k]}] for k in todo], params, use_tqdm=False,
                           chat_template_kwargs={"enable_thinking": False})
        for k, output in zip(todo, outputs, strict=True):
            reply = output.outputs[0]
            last[k] = reply.text
            text = reply.text.strip() if reply.finish_reason == "stop" else None
            if not bad(vi[k], text):
                en[k] = text
        failing = [k for k in todo if en[k] is None]
        if len(todo) > MIN_ROWS and len(failing) > len(todo) / 2:
            raise SystemExit(f"{len(failing):,} of {len(todo):,} rows failed try {attempt + 1}, a systematic "
                             f"failure: stopping; e.g. {last[failing[0]][:300]!r}")
        todo = failing
        if not todo:
            break
    return en, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                        default=slice(None), help="a slice of the source's segments shards, e.g. 0:1")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="the engine's share of the GPU, with voice and the ASR server stopped")
    args = parser.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    repo = REPO
    sha = HfApi().dataset_info(repo).sha
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{repo}@{sha}/{args.source}/segments/*.parquet"),
                    key=lambda f: [int(n) for n in re.findall(r"\d+", Path(f).name)])[args.shards]
    if not shards:
        raise SystemExit(f"no {args.source} segments in {repo}")
    out = args.root / args.source / "text_en"
    out.mkdir(parents=True, exist_ok=True)
    lock_file = exclusive(out / "translate.lock", f"another translate run is writing {out}")  # noqa: F841
    log(f"{args.source}: {len(shards)} shards, {repo}@{sha[:7]} -> {out}")

    def read(path: str) -> tuple[str, list[str], list[str], list[str], list[str | None]]:
        """A shard's name, ids, docs, texts (all-capital ones lowercased) and existing text_en."""
        with fs.open(path, "rb", cache_type="none") as fh:  # only these columns' chunks, not the audio
            t = pq.ParquetFile(fh).read(columns=["id", "doc", "text"])
        name = Path(path).name
        ids, docs, texts = (t.column(c).to_pylist() for c in ("id", "doc", "text"))
        text_en = [None] * len(ids)
        if (out / name).exists():
            t = pq.read_table(out / name)
            if t.column("id").to_pylist() != ids:
                raise SystemExit(f"{out / name} holds other rows than its segments shard")
            text_en = t.column("text_en").to_pylist()
        return name, ids, docs, [text.lower() if text.isupper() else text for text in texts], text_en

    llm, tried, failed, start = None, 0, 0, time.time()
    with ThreadPoolExecutor(GROUP) as readers:
        groups = [shards[i:i + GROUP] for i in range(0, len(shards), GROUP)]
        pending = readers.map(read, groups[0])
        for g, group in enumerate(groups):
            loaded = list(pending)
            # Rows without a translation: all of a new shard, or on a rerun those left null; never a
            # row already written, which may be voiced.
            todo = [(s, k) for s, (_, _, _, _, text_en) in enumerate(loaded) for k, en in enumerate(text_en)
                    if en is None]
            if todo and llm is None:  # before the next reads start: vLLM forks its engine process
                llm, start = engine(args.gpu_memory), time.time()
            if g + 1 < len(groups):  # the next group's reads overlap this group's translation
                pending = readers.map(read, groups[g + 1])
            if not todo:
                log(f"{', '.join(shard[0] for shard in loaded)}: complete")
                continue
            again = [(out / loaded[s][0]).exists() for s, _ in todo]
            en, last = translate(llm, [loaded[s][3][k] for s, k in todo],
                                 [prompt(loaded[s][3], loaded[s][2], k) for s, k in todo], again)
            for (s, k), text in zip(todo, en, strict=True):
                loaded[s][4][k] = text
            for name, ids, _, _, text_en in loaded:
                part = out / f"{name}.part"
                pq.write_table(pa.Table.from_pydict({"id": ids, "text_en": text_en}, schema=SCHEMA), part)
                os.replace(part, out / name)
            for (s, k), text, reply in zip(todo, en, last, strict=True):
                if text is None:
                    log(f"failed every try: {loaded[s][1][k]} {loaded[s][3][k][:120]!r} -> {reply[:200]!r}")
            tried += len(todo)
            failed += sum(text is None for text in en)
            minutes = (time.time() - start) / 60
            log(f"{', '.join(shard[0] for shard in loaded)}: {len(todo)} rows translated, "
                f"{sum(text is None for text in en)} failed; {min((g + 1) * GROUP, len(shards))}/{len(shards)} "
                f"shards, {tried / 60 / minutes:.0f} rows/s, {minutes:.1f} min")
            if tried > MIN_ROWS and failed > MAX_FAILED * tried:
                raise SystemExit(f"{failed:,} of {tried:,} rows failed every try, a systematic failure: stopping")
    log(f"{args.source}: done; {failed} rows without a translation")


if __name__ == "__main__":
    main()
