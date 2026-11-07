#!/usr/bin/env python
"""VI→EN translation of the tranche's segments on the box: Index-Translate-35B-A3B in vLLM, in this
process, on the GPU ([tranche-translate](../../docs/agent-memory/tranche-translate.md)).

Reads each segments shard's id, doc and text from its tranche repo on the Hub (only those columns, no
audio), translates each row as its own prompt, with up to CONTEXT_ROWS earlier rows of its doc as
context, and writes <root>/<source>/text_en/<shard>.parquet (id, text_en) beside the segments'
layout; `ingest push` uploads it. A transcript all in capitals (GigaSpeech 2's) goes in lowercased:
kept, the model answers in capitals, which the TTS spells out. The first try is greedy; a reply cut
at max_tokens, still in Vietnamese, holding letters of another script or brackets, or far too long
is tried again sampled, up to TRIES times, the later half without context rows, never again with
the tokens it wrote another script or the Vietnamese it echoed with. A row failing every try
keeps text_en null, and a rerun tries the null rows again (and only those: a written row may
already be voiced). It takes the GPU memory voice and the ASR server
hold, so they take turns. From the repo root, in the env /workspace/envs/mt (`envs.sh`):

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
from training.prep.translate_en_speech import NON_LATIN, broken
from training.tranche.ingest import REPO, ROOT, SOURCES

# Chosen on the box against Index-Translate-9B, Hy-MT2-30B-A3B and Muse (tranche-translate, Bake-off).
# Apache-2.0; BF16 weights, quantized to FP8 as they load.
MODEL = ("IndexTeam/Index-Translate-35B-A3B-preview", "7e4029e0df842502cb009cf493d6467c247043b3")
MAX_TOKENS = 512  # a row's English takes ~20 tokens, the longest a few hundred
BANNED = 1000  # token ids a row's later tries may not emit, at most: vLLM's V2 runner takes 1,024 biases
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
VIETNAMESE_LETTER = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fold(text: str) -> str:
    """Without diacritics (đ as d)."""
    text = unicodedata.normalize("NFD", text.replace("Đ", "D").replace("đ", "d"))
    return "".join(c for c in text if not unicodedata.combining(c))


# "do" (đó) is also English.
FOLDED_WORDS = {fold(w) for w in VIETNAMESE_WORDS} - {"do"}
EDGES = ".,;:!?\"'()…“”‘’"


def left_in_vietnamese(en: str, folded: bool = False) -> bool:
    """Two or more of VIETNAMESE_WORDS; `folded`, written without their diacritics, lowercase and
    whole (not name syllables such as Thi, nor 've or co- in I've, co-founder)."""
    if folded:
        return sum(w.strip(EDGES) in FOLDED_WORDS for w in fold(en).split()) >= 2
    words = re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", en).lower()).split()
    return sum(w in VIETNAMESE_WORDS for w in words) >= 2


def vietnamese_letters(text: str) -> bool:
    """A Vietnamese letter, precomposed or as a base letter and combining marks; or a lone mark."""
    return bool(VIETNAMESE_LETTER.search(unicodedata.normalize("NFC", text))) or any(
        unicodedata.combining(c) for c in text)


def bad(vi: str, en: str | None, context: bool, banned: bool) -> bool:
    """A reply cut at max_tokens, empty, holding letters of a script its Vietnamese lacks or brackets,
    left in Vietnamese, or far longer than its row. From a prompt with context rows the model now
    and then translates them too (1 in 2,000 rows of the check; English runs ~1.2× the Vietnamese's
    characters), so the limit is tight there; without them a long reply is mostly an expansion
    ("nội ngoại sản nhi": "internal medicine, surgery, obstetrics and gynecology, pediatrics"), and
    the limit only stops a reply that runs on. A row whose tokens are `banned` may hold no
    Vietnamese letter at all and no Vietnamese words without their diacritics: the model dodges a
    ban with the word's first piece and another ending ("ChỊ Huong", "Bá Hàng"), or in ASCII."""
    if en is None or not en.strip() or broken(vi, en):
        return True
    if len(en) > (2.5 * len(vi) + 20 if context else 4 * len(vi) + 40):
        return True
    if "[" in en and "[" not in vi:  # "[Name]", "[a specific time]": the TTS would read them
        return True
    if banned and (vietnamese_letters(en) or left_in_vietnamese(en, folded=True)):
        return True
    return left_in_vietnamese(en)


def prompt(texts: list[str], docs: list[str], k: int, rows: int = CONTEXT_ROWS) -> str:
    """Row k's prompt, with up to `rows` earlier rows of its doc (only rows of the same doc:
    viVoice's rows are docs of their own)."""
    context = [texts[j] for j in range(max(0, k - rows), k) if docs[j] == docs[k]]
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


def translate(llm, vi: list[str], prompts: list[str], bare: list[str],
              again: list[bool]) -> tuple[list[str | None], list[str]]:
    """-> each row's English (None where it fails every try) and its last reply. Each try is one
    generate call over the rows still without one: greedy first, then sampled at Qwen3.5's
    non-thinking settings, seeded afresh each try (the engine's own seed is fixed) so that a rerun
    draws anew. The second half of the tries, and every try of a row that failed in an earlier run
    (`again`, sampled from its first), use the prompt without context rows (`bare`). Stops the run
    when more than half the rows of a try over more than MIN_ROWS fail it: a systematic failure."""
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)
    tokenizer = llm.get_tokenizer()
    en: list[str | None] = [None] * len(vi)
    last = [""] * len(vi)
    # Token ids a row's later tries may not emit, in the order banned (the newest BANNED are kept):
    # - those its earlier replies wrote other scripts with. The model writes some Sino-Vietnamese
    #   names in Chinese every try ("King Tran Thai宗" for Trần Thái Tông); exact ids, as a letter
    #   also hides in multi-letter tokens (太宗) and byte pieces (which decode to U+FFFD);
    # - those a reply left in Vietnamese wrote Vietnamese letters with: some rows come back as
    #   their own Vietnamese, capitalized, every try.
    banned: dict[int, dict[int, None]] = {}
    todo = list(range(len(vi)))
    for attempt in range(TRIES):
        sampling = {"max_tokens": MAX_TOKENS, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "seed": int.from_bytes(os.urandom(4))}
        sampled = SamplingParams(**sampling)
        plain = {k: again[k] or attempt >= TRIES // 2 for k in todo}
        params = [greedy if not (attempt or again[k]) else
                  SamplingParams(**sampling, logit_bias={t: -100 for t in list(banned[k])[-BANNED:]})
                  if k in banned else sampled for k in todo]
        outputs = llm.chat([[{"role": "user", "content": bare[k] if plain[k] else prompts[k]}] for k in todo],
                           params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for k, output in zip(todo, outputs, strict=True):
            reply = output.outputs[0]
            last[k] = reply.text
            text = reply.text.strip() if reply.finish_reason == "stop" else None
            if not bad(vi[k], text, context=not plain[k] and prompts[k] != bare[k], banned=k in banned):
                en[k] = text
                continue
            own = set(NON_LATIN.findall(vi[k]))
            foreign = bool(set(NON_LATIN.findall(reply.text)) - own)
            # Under bans, any Vietnamese letter: its tokens are never needed in this row's English.
            echoed = left_in_vietnamese(reply.text) or (k in banned and vietnamese_letters(reply.text))
            if foreign or echoed:
                for t in reply.token_ids:
                    piece = tokenizer.decode([t])
                    if (foreign and ("\ufffd" in piece or set(NON_LATIN.findall(piece)) - own)
                            or echoed and vietnamese_letters(piece)):
                        banned.setdefault(k, {}).pop(t, None)  # re-banned: the newest again
                        banned[k][t] = None
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
            if todo and llm is None:  # before the next group's reads: none is in flight as it starts
                llm, start = engine(args.gpu_memory), time.time()
            if g + 1 < len(groups):  # the next group's reads overlap this group's translation
                pending = readers.map(read, groups[g + 1])
            if not todo:
                log(f"{', '.join(shard[0] for shard in loaded)}: complete")
                continue
            again = [(out / loaded[s][0]).exists() for s, _ in todo]
            en, last = translate(llm, [loaded[s][3][k] for s, k in todo],
                                 [prompt(loaded[s][3], loaded[s][2], k) for s, k in todo],
                                 [prompt(loaded[s][3], loaded[s][2], k, rows=0) for s, k in todo], again)
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
