#!/usr/bin/env python
"""VI→EN translation of the tranche's segments on the box: Index-Translate-35B-A3B in vLLM, in this
process, on the GPU ([tranche-translate](../../docs/agent-memory/tranche-translate.md)).

Reads each segments shard's id, doc and text from its tranche repo on the Hub (only those columns, no
audio), translates each row as its own prompt (or in multi-segment windows of consecutive doc rows
with `--window`, falling back to single-row translation on any failed window), with up to
CONTEXT_ROWS earlier rows of its doc as context, and writes <root>/<source>/text_en/<shard>.parquet
(id, text_en) beside the segments' layout; `ingest push` uploads it. A transcript all in capitals
(GigaSpeech 2's) goes in lowercased: kept, the model answers in capitals, which the TTS spells out.
The first try is greedy; a reply cut at max_tokens, still in Vietnamese, holding letters of another
script or brackets, or far too long is tried again sampled, up to TRIES times, the later half
without context rows, never again with the tokens it wrote another script or the Vietnamese it
echoed with. A row failing every try keeps text_en null, and a rerun tries the null rows again (and
only those: a written row may already be voiced). It takes the GPU memory voice and the ASR server
hold, so they take turns. From the repo root, in the env /workspace/envs/mt (`box/envs.sh`):

  python -m training.tranche.translate giga --shards 51:
"""
from __future__ import annotations

import argparse
import math
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
# Apache-2.0; BF16 weights, quantized to FP8 as they load. On a card under 64 GiB (an RTX 5090's 32
# GB hold no FP8 35B) IndexTeam's own NVFP4 build of it, which Blackwell runs natively.
MODEL = ("IndexTeam/Index-Translate-35B-A3B-preview", "7e4029e0df842502cb009cf493d6467c247043b3")
MODEL_FP4 = ("IndexTeam/Index-Translate-35B-A3B-preview-FP4", "111ff5e950f75dcccb1e631f65e936781a9c34d7")
MAX_TOKENS = 512  # hard ceiling per row; per-request caps tighten to bad()'s length gate
MAX_WINDOW_CHARS = 1_600  # caps packed window Vietnamese input so output stays well under max_model_len
BANNED = 1000  # token ids a row's later tries may not emit, at most: vLLM's V2 runner takes 1,024 biases
TRIES = 6  # per row and run: greedy, then sampled; a failing row holds its whole shard back, and retries cost seconds
CONTEXT_ROWS = 2
# The short form scored as the long one on the bake-off (FLEURS dev chrF 64.8 against 64.7, archive
# 50 against Muse 68.7 against 69.1, on Index-Translate-9B) with 21% fewer prompt tokens.
PROMPT = ("Translate this Vietnamese speech segment into English. It may start or stop mid-sentence: translate "
          "exactly what it says, without completing, adding or dropping anything. Use normal capitalization and "
          "punctuation, Vietnamese names without diacritics, no brackets or notes. Output only the translation.")
WINDOW_PROMPT = ("Translate these numbered Vietnamese speech segments into English. Each line may start or stop "
                 "mid-sentence: translate exactly what each line says, without completing, adding, dropping, "
                 "merging, splitting, reordering or moving words between lines. Output exactly one line "
                 "`n|English` for each input line, in the same order, and nothing else. Use normal "
                 "capitalization and punctuation, Vietnamese names without diacritics, no brackets or notes.")
SCHEMA = pa.schema([("id", pa.string()), ("text_en", pa.string())])
GROUP = 8  # shards translated together: keeps the GPU's batch full across shard boundaries
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


# "do" (đó) and "den" (đến) are also English.
FOLDED_WORDS = {fold(w) for w in VIETNAMESE_WORDS} - {"do", "den"}
EDGES = ".,;:!?\"'()…“”‘’"


def left_in_vietnamese(en: str, folded: bool = False) -> bool:
    """Two or more of VIETNAMESE_WORDS; `folded`, two different ones written without their
    diacritics, lowercase and whole (not name syllables such as Thi, nor 've or co- in I've,
    co-founder)."""
    if folded:
        return len({w.strip(EDGES) for w in fold(en).split()} & FOLDED_WORDS) >= 2
    words = re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", en).lower()).split()
    return sum(w in VIETNAMESE_WORDS for w in words) >= 2


def vietnamese_letters(text: str) -> bool:
    """A Vietnamese letter, precomposed or as a base letter and combining marks; or any combining
    mark."""
    return bool(VIETNAMESE_LETTER.search(unicodedata.normalize("NFC", text))) or any(
        unicodedata.combining(c) for c in text)


def bad(vi: str, en: str | None, context: bool, strict: bool) -> bool:
    """A reply cut at max_tokens, empty, holding letters of a script its Vietnamese lacks or brackets,
    left in Vietnamese, or far longer than its row. From a prompt with context rows the model now
    and then translates them too (1 in 2,000 rows of the check; English runs ~1.2× the Vietnamese's
    characters), so the limit is tight there; without them a long reply is mostly an expansion
    ("nội ngoại sản nhi": "internal medicine, surgery, obstetrics and gynecology, pediatrics"), and
    the limit only stops a reply that runs on. A `strict` reply (its row under bans, or failed in
    an earlier run) may hold no Vietnamese letter at all and no Vietnamese words without their
    diacritics: the model dodges a ban with the word's first piece and another ending ("ChỊ Huong",
    "Bá Hàng"), or in ASCII."""
    if en is None or not en.strip() or broken(vi, en):
        return True
    if len(en) > (2.5 * len(vi) + 20 if context else 4 * len(vi) + 40):
        return True
    if "[" in en and "[" not in vi:  # "[Name]", "[a specific time]": the TTS would read them
        return True
    if strict and (vietnamese_letters(en) or left_in_vietnamese(en, folded=True)):
        return True
    return left_in_vietnamese(en)


def row_max_tokens(vi: str, context: bool, cap: int = MAX_TOKENS) -> int:
    """Per-row token ceiling aligned with bad()'s character limit (conservative 2.2 chars/token +
    16-token margin, bucketed to multiples of 32): never clips a reply bad() would accept, while
    aborting looping/runaway replies 2–5× sooner than a flat 512."""
    char_limit = 2.5 * len(vi) + 20 if context else 4 * len(vi) + 40
    tokens = max(64, math.ceil(char_limit / 2.2) + 16)
    bucketed = ((tokens + 31) // 32) * 32
    return min(cap, bucketed)


def window_max_tokens(vis: list[str], cap: int = 1024) -> int:
    """Token ceiling for a packed multi-segment window of rows `vis`."""
    char_limit = sum(2.5 * len(v) + 20 for v in vis)
    tokens = max(96, math.ceil(char_limit / 2.2) + 16 * len(vis))
    bucketed = ((tokens + 31) // 32) * 32
    return min(cap, bucketed)


def _trim_context(context: list[str], max_chars: int) -> list[str]:
    """Keep at most the trailing `max_chars` characters across context rows, trimmed on word edges."""
    if max_chars <= 0 or not context:
        return context
    joined = "\n".join(context)
    if len(joined) <= max_chars:
        return context
    tail = joined[-max_chars:]
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    return [tail] if tail.strip() else []


def prompt(texts: list[str], docs: list[str], k: int, rows: int = CONTEXT_ROWS,
           max_context_chars: int = 0) -> str:
    """Row k's prompt, with up to `rows` earlier rows of its doc (only rows of the same doc:
    viVoice's rows are docs of their own)."""
    context = [texts[j] for j in range(max(0, k - rows), k) if docs[j] == docs[k]]
    context = _trim_context(context, max_context_chars)
    background = ("\n\nContext, earlier in the recording (do not translate):\n" + "\n".join(context)
                  if context else "")
    return f"{PROMPT}{background}\n\nSegment:\n{texts[k]}"


def window_prompt(texts: list[str], docs: list[str], indices: list[int],
                  context_rows: int = 1, max_context_chars: int = 0) -> str:
    """Multi-segment prompt for consecutive rows `indices` of the same doc, with up to
    `context_rows` earlier rows before `indices[0]` as carry-over context."""
    first = indices[0]
    context = [texts[j] for j in range(max(0, first - context_rows), first) if docs[j] == docs[first]]
    context = _trim_context(context, max_context_chars)
    background = ("\n\nContext, earlier in the recording (do not translate):\n" + "\n".join(context)
                  if context else "")
    numbered = "\n".join(f"{i}|{texts[idx]}" for i, idx in enumerate(indices, 1))
    return f"{WINDOW_PROMPT}{background}\n\nSegments:\n{numbered}"


def parse_window(text: str, n: int) -> list[str] | None:
    """Parse a numbered `n|English` multi-segment reply; returns None unless all 1..n match."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) != n:
        return None
    out: list[str] = []
    for i, line in enumerate(lines, 1):
        number, sep, en = line.partition("|")
        if not sep or number.strip() != str(i) or not en.strip():
            return None
        out.append(en.strip())
    return out


def pack_windows(todo: list[tuple[int, int]], docs_by_shard: list[list[str]],
                 texts_by_shard: list[list[str]], again: list[bool], window: int,
                 max_chars: int = MAX_WINDOW_CHARS) -> tuple[list[list[int]], list[int]]:
    """Group consecutive fresh `todo` rows of the same shard and doc into packed windows of size
    2..`window` (capped at `max_chars` Vietnamese characters); returns (packed_positions, single_positions)
    where each entry is a position in `todo`."""
    if window <= 1:
        return [], list(range(len(todo)))
    packed: list[list[int]] = []
    singles: list[int] = []
    cur: list[int] = []
    cur_chars = 0

    def flush() -> None:
        nonlocal cur, cur_chars
        if len(cur) >= 2:
            packed.append(cur)
        else:
            singles.extend(cur)
        cur, cur_chars = [], 0

    for pos, (s, k) in enumerate(todo):
        vi_text = texts_by_shard[s][k]
        if again[pos] or "\n" in vi_text:
            flush()
            singles.append(pos)
            continue
        text_len = len(vi_text)
        if cur:
            prev_s, prev_k = todo[cur[-1]]
            same_run = (s == prev_s and k == prev_k + 1
                        and docs_by_shard[s][k] == docs_by_shard[prev_s][prev_k])
            if not same_run or len(cur) >= window or cur_chars + text_len > max_chars:
                flush()
        cur.append(pos)
        cur_chars += text_len
    flush()
    return packed, singles


def _supported_engine_args() -> set[str]:
    """Fields accepted by vLLM's EngineArgs in the installed version."""
    try:
        from dataclasses import fields
        from vllm.engine.arg_utils import EngineArgs

        return {f.name for f in fields(EngineArgs)}
    except Exception:
        return set()


def engine(
    gpu_memory: float | None = None,
    *,
    model: str | None = None,
    quantization: str | None = None,
    max_model_len: int = 2048,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int = 16384,
    kv_cache_dtype: str | None = None,
    mamba_ssm_cache_dtype: str | None = None,
    moe_backend: str | None = None,
    mtp_tokens: int = 0,
    async_scheduling: bool = True,
):
    """Build the in-process vLLM engine. Prefix caching stays off: on this hybrid (linear-attention)
    model, our ~73-token instruction prefix is shorter than one aligned hybrid block (1,088 tokens),
    so prefix caching never hits while cutting concurrency ~3×. On a 32 GB card (NVFP4), pairing
    `mamba_ssm_cache_dtype="bfloat16"` (halving the 30-layer GDN state from float32's 61.4 MiB/seq
    to 31.4 MiB/seq) with `kv_cache_dtype="fp8"` keeps the hybrid attention/Mamba page ratio matched
    at 1,088 tokens and unlocks `max_num_seqs=512` (+19–22% YODAS rows/s)."""
    os.environ.setdefault("VLLM_HAS_FLASHINFER_CUBIN", "1")
    from huggingface_hub import snapshot_download
    from vllm import LLM
    from vllm.platforms import current_platform

    # The FP8 build's 35 GB of weights leave a card under 64 GiB too little for its sequences (a 48 GB
    # one ~8 GB at 0.9, as the 5090 had when FP4 refused 512): there the NVFP4 build (Blackwell; its
    # quantization is the checkpoint's own). NVML's figure: torch's would start CUDA in this process.
    fp4 = current_platform.get_device_total_memory() < 64 * 2**30
    if model is None:
        repo, revision = MODEL_FP4 if fp4 else MODEL
        model_path = snapshot_download(repo, revision=revision)
        quant = quantization if quantization is not None else (None if fp4 else "fp8")
    else:
        model_path = model if Path(model).exists() else snapshot_download(model)
        quant = quantization
    # FP4 (a 32 GB RTX 5090): default fp8 KV + bfloat16 GDN SSM state unlocks 512 sequences at 0.95
    # (default float32 SSM state from config.json capped Mamba cache blocks at ~317–339).
    gpu_memory = gpu_memory or (0.95 if fp4 else 0.9)
    supported = _supported_engine_args()
    kv_cache = kv_cache_dtype or ("fp8" if fp4 else "auto")
    ssm_cache = mamba_ssm_cache_dtype or ("bfloat16" if fp4 else "auto")
    half_ssm = ssm_cache in ("bfloat16", "float16") and "mamba_ssm_cache_dtype" in supported
    seqs = max_num_seqs or (512 if (not fp4 or half_ssm) else 256)
    kwargs: dict[str, object] = {
        "model": model_path,
        "quantization": quant,
        "gpu_memory_utilization": gpu_memory,
        "max_model_len": max_model_len,
        "max_num_seqs": seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": False,
    }
    if "language_model_only" in supported:
        kwargs["language_model_only"] = True
    else:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if kv_cache != "auto" and "kv_cache_dtype" in supported:
        kwargs["kv_cache_dtype"] = kv_cache
    if ssm_cache != "auto" and "mamba_ssm_cache_dtype" in supported:
        kwargs["mamba_ssm_cache_dtype"] = ssm_cache
    if moe_backend and "moe_backend" in supported:
        kwargs["moe_backend"] = moe_backend
    if async_scheduling and "async_scheduling" in supported:
        kwargs["async_scheduling"] = True
    if mtp_tokens > 0 and "speculative_config" in supported:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": mtp_tokens}
    return LLM(**kwargs)


def _generate(llm, tokenizer, contents: list[str], params: list) -> list:
    """Run one batched generation pass. Pre-renders the non-thinking chat template and runs the
    parallel Rust batch tokenizer into `TokensPrompt` when vLLM exposes it (cutting single-threaded
    Python overhead in `llm.chat`), falling back cleanly to `llm.chat` otherwise."""
    if hasattr(llm, "generate") and hasattr(tokenizer, "apply_chat_template") and callable(tokenizer):
        try:
            from vllm.inputs import TokensPrompt

            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for content in contents
            ]
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
            return llm.generate(prompts, params, use_tqdm=False)
        except Exception:
            pass
    return llm.chat([[{"role": "user", "content": content}] for content in contents],
                    params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})


def _ban_tokens(tokenizer, vi_text: str, reply, strict: bool,
                banned: dict[int, dict[int, None]], k: int) -> None:
    """Record token ids that wrote a foreign script or echoed Vietnamese so later tries forbid them."""
    own = set(NON_LATIN.findall(vi_text))
    foreign = bool(set(NON_LATIN.findall(reply.text)) - own)
    # A strict reply's Vietnamese letters: their tokens are never needed in its row's English.
    echoed = left_in_vietnamese(reply.text) or (strict and vietnamese_letters(reply.text))
    if foreign or echoed:
        for t in reply.token_ids:
            piece = tokenizer.decode([t])
            if (foreign and ("\ufffd" in piece or set(NON_LATIN.findall(piece)) - own)
                    or echoed and vietnamese_letters(piece)):
                banned.setdefault(k, {}).pop(t, None)  # re-banned: the newest again
                banned[k][t] = None


def translate(llm, vi: list[str], prompts: list[str], bare: list[str],
              again: list[bool], *,
              windows: list[list[int]] | None = None,
              window_prompts: list[str] | None = None,
              max_model_len: int = 2048) -> tuple[list[str | None], list[str]]:
    """-> each row's English (None where it fails every try) and its last reply. When `windows` is
    provided, translates packed multi-segment windows first (greedy) and accepts a window only when
    every line parses cleanly and passes `bad()`; any failing window falls back to single-row
    translation. Each single-row try is one generate call sorted longest-first with per-row
    `max_tokens` aligned to `bad()`'s length gate: greedy first (deduplicated across identical
    prompts), then sampled at Qwen3.5's non-thinking settings, seeded afresh each try. The second
    half of the tries, and every try of a row that failed in an earlier run (`again`, sampled from
    its first), use the prompt without context rows (`bare`). Stops the run when more than half the
    rows of a try over more than MIN_ROWS fail it: a systematic failure."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    en: list[str | None] = [None] * len(vi)
    last = [""] * len(vi)
    greedy_cache: dict[int, SamplingParams] = {}

    def greedy_for(limit: int) -> SamplingParams:
        if limit not in greedy_cache:
            greedy_cache[limit] = SamplingParams(max_tokens=limit, temperature=0)
        return greedy_cache[limit]

    # Phase 3 optional fast path: packed multi-segment windows (all-or-nothing acceptance per window,
    # falling back to K=1 single-row translation on any parse or quality gate miss).
    if windows and window_prompts:
        w_order = sorted(range(len(windows)), key=lambda i: (-len(window_prompts[i]), i))
        w_cap = max(MAX_TOKENS, max_model_len // 2)
        w_contents = [window_prompts[i] for i in w_order]
        w_params = [greedy_for(window_max_tokens([vi[k] for k in windows[i]], cap=w_cap))
                    for i in w_order]
        w_outputs = _generate(llm, tokenizer, w_contents, w_params)
        accepted_w, accepted_rows = 0, 0
        for i, output in zip(w_order, w_outputs, strict=True):
            reply = output.outputs[0]
            rows = windows[i]
            parsed = parse_window(reply.text, len(rows)) if reply.finish_reason == "stop" else None
            if parsed is not None and all(
                not bad(vi[k], text, context=True, strict=False)
                for k, text in zip(rows, parsed, strict=True)
            ):
                accepted_w += 1
                accepted_rows += len(rows)
                for k, text in zip(rows, parsed, strict=True):
                    en[k] = text
                    last[k] = text
        packed_total = sum(len(w) for w in windows)
        log(f"windows: {accepted_w}/{len(windows)} accepted ({accepted_rows}/{packed_total} rows, "
            f"{100 * accepted_rows / max(packed_total, 1):.1f}%); "
            f"{packed_total - accepted_rows} fallback to single-row")

    # Token ids a row's later tries may not emit, in the order banned (the newest BANNED are kept):
    # - those its earlier replies wrote other scripts with. The model writes some Sino-Vietnamese
    #   names in Chinese every try ("King Tran Thai宗" for Trần Thái Tông); exact ids, as a letter
    #   also hides in multi-letter tokens (太宗) and byte pieces (which decode to U+FFFD);
    # - those a reply left in Vietnamese, or a strict reply (bad()) holding Vietnamese letters,
    #   wrote Vietnamese letters with: some rows come back as their own Vietnamese, capitalized,
    #   every try, and under bans the model writes the word's first token with another ending.
    banned: dict[int, dict[int, None]] = {}
    todo = [k for k in range(len(vi)) if en[k] is None]
    for attempt in range(TRIES):
        if not todo:
            break
        sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "seed": int.from_bytes(os.urandom(4))}
        sampled_cache: dict[int, SamplingParams] = {}

        def sampled_for(limit: int) -> SamplingParams:
            if limit not in sampled_cache:
                sampled_cache[limit] = SamplingParams(max_tokens=limit, **sampling)
            return sampled_cache[limit]

        plain = {k: again[k] or attempt >= TRIES // 2 for k in todo}
        content_of = {k: bare[k] if plain[k] else prompts[k] for k in todo}
        has_ctx = {k: not plain[k] and prompts[k] != bare[k] for k in todo}
        cap_of = {k: row_max_tokens(vi[k], context=has_ctx[k]) for k in todo}

        # Sort longest-first (LPT scheduling) so long prompts start early and short rows backfill
        # the batch tail instead of straggling. On greedy attempt 0, deduplicate identical prompts.
        ordered = sorted(todo, key=lambda k: (-len(content_of[k]), k))
        if attempt == 0:
            rep_order: list[int] = []
            groups_by_key: dict[tuple[str, int, bool], list[int]] = {}
            for k in ordered:
                if again[k]:
                    rep_order.append(k)
                    continue
                key = (content_of[k], cap_of[k], has_ctx[k])
                if key not in groups_by_key:
                    groups_by_key[key] = [k]
                    rep_order.append(k)
                else:
                    groups_by_key[key].append(k)
        else:
            rep_order = ordered
            groups_by_key = {}

        params = [
            greedy_for(cap_of[k]) if not (attempt or again[k]) else
            SamplingParams(max_tokens=cap_of[k], **sampling,
                           logit_bias={t: -100 for t in list(banned[k])[-BANNED:]})
            if k in banned else sampled_for(cap_of[k])
            for k in rep_order
        ]
        outputs = _generate(llm, tokenizer, [content_of[k] for k in rep_order], params)
        for rep_k, output in zip(rep_order, outputs, strict=True):
            reply = output.outputs[0]
            members = (groups_by_key.get((content_of[rep_k], cap_of[rep_k], has_ctx[rep_k]), [rep_k])
                       if attempt == 0 and not again[rep_k] else [rep_k])
            for k in members:
                last[k] = reply.text
                text = reply.text.strip() if reply.finish_reason == "stop" else None
                strict = k in banned or again[k]
                if not bad(vi[k], text, context=has_ctx[k], strict=strict):
                    en[k] = text
                    continue
                _ban_tokens(tokenizer, vi[k], reply, strict, banned, k)
        failing = [k for k in todo if en[k] is None]
        if len(todo) > MIN_ROWS and len(failing) > len(todo) / 2:
            raise SystemExit(f"{len(failing):,} of {len(todo):,} rows failed try {attempt + 1}, a systematic "
                             f"failure: stopping; e.g. {last[failing[0]][:300]!r}")
        todo = failing
    return en, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                        default=slice(None), help="a slice of the source's segments shards, e.g. 0:1")
    parser.add_argument("--gpu-memory", type=float,
                        help="the engine's share of the GPU, with voice and the ASR server stopped "
                             "(default 0.9, FP4 0.95)")
    parser.add_argument("--group", type=int, default=GROUP,
                        help=f"shards translated together per batch (default {GROUP})")
    parser.add_argument("--window", type=int, default=1,
                        help="consecutive doc rows packed per prompt (default 1; 4..8 for multi-segment "
                             "packing with automatic single-row fallback)")
    parser.add_argument("--context-rows", type=int, default=CONTEXT_ROWS,
                        help=f"earlier doc rows as context for single-row prompts (default {CONTEXT_ROWS})")
    parser.add_argument("--window-context-rows", type=int, default=1,
                        help="earlier doc rows as carry-over context for packed windows (default 1)")
    parser.add_argument("--max-context-chars", type=int, default=0,
                        help="cap context length to trailing N chars (0 = unlimited)")
    parser.add_argument("--model", type=str,
                        help="override HF repo or local checkpoint path (e.g. custom FP8-GDN or 9B NVFP4)")
    parser.add_argument("--quantization", type=str,
                        help="override vLLM quantization (default: None for FP4, 'fp8' for BF16 base)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="vLLM max_model_len (default 2048)")
    parser.add_argument("--max-num-seqs", type=int,
                        help="vLLM max_num_seqs (default 512)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        help="vLLM max_num_batched_tokens (default 16384)")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8", "fp8_e4m3", "fp8_e5m2"),
                        help="vLLM full-attention KV cache dtype (default fp8 on <64 GiB FP4, auto on FP8)")
    parser.add_argument("--mamba-ssm-cache-dtype", choices=("auto", "float32", "bfloat16", "float16"),
                        help="vLLM Gated DeltaNet SSM state cache dtype (default bfloat16 on <64 GiB FP4, auto on FP8)")
    parser.add_argument("--moe-backend", type=str,
                        help="vLLM MoE kernel backend override (e.g. flashinfer_cutlass, cutlass, marlin)")
    parser.add_argument("--mtp-tokens", type=int, default=0,
                        help="native MTP speculative tokens per step (default 0 = off; 1..3 to enable)")
    parser.add_argument("--no-async-scheduling", action="store_true",
                        help="disable vLLM V1 async CPU-GPU scheduling overlap")
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
    group_size = max(1, args.group)
    with ThreadPoolExecutor(min(group_size, 8)) as readers:
        groups = [shards[i:i + group_size] for i in range(0, len(shards), group_size)]
        pending = readers.map(read, groups[0])
        for g, group in enumerate(groups):
            loaded = list(pending)
            # Rows without a translation: all of a new shard, or on a rerun those left null; never a
            # row already written, which may be voiced.
            todo = [(s, k) for s, (_, _, _, _, text_en) in enumerate(loaded) for k, en in enumerate(text_en)
                    if en is None]
            if todo and llm is None:  # before the next group's reads: none is in flight as it starts
                llm = engine(
                    args.gpu_memory,
                    model=args.model,
                    quantization=args.quantization,
                    max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    kv_cache_dtype=args.kv_cache_dtype,
                    mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
                    moe_backend=args.moe_backend,
                    mtp_tokens=args.mtp_tokens,
                    async_scheduling=not args.no_async_scheduling,
                )
                start = time.time()
            if g + 1 < len(groups):  # the next group's reads overlap this group's translation
                pending = readers.map(read, groups[g + 1])
            if not todo:
                log(f"{', '.join(shard[0] for shard in loaded)}: complete")
                continue
            again = [(out / loaded[s][0]).exists() for s, _ in todo]
            packed, _ = pack_windows(
                todo, [shard[2] for shard in loaded], [shard[3] for shard in loaded], again, args.window
            )
            w_prompts = [
                window_prompt(
                    loaded[todo[w[0]][0]][3],
                    loaded[todo[w[0]][0]][2],
                    [todo[pos][1] for pos in w],
                    context_rows=args.window_context_rows,
                    max_context_chars=args.max_context_chars,
                )
                for w in packed
            ]
            en, last = translate(
                llm,
                [loaded[s][3][k] for s, k in todo],
                [prompt(loaded[s][3], loaded[s][2], k, rows=args.context_rows,
                        max_context_chars=args.max_context_chars) for s, k in todo],
                [prompt(loaded[s][3], loaded[s][2], k, rows=0) for s, k in todo],
                again,
                windows=packed,
                window_prompts=w_prompts,
                max_model_len=args.max_model_len,
            )
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
                f"{sum(text is None for text in en)} failed; {min((g + 1) * group_size, len(shards))}/{len(shards)} "
                f"shards, {tried / 60 / minutes:.0f} rows/s, {minutes:.1f} min")
            if tried > MIN_ROWS and failed > MAX_FAILED * tried:
                raise SystemExit(f"{failed:,} of {tried:,} rows failed every try, a systematic failure: stopping")
    log(f"{args.source}: done; {failed} rows without a translation")


if __name__ == "__main__":
    main()
