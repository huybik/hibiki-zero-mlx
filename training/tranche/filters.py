"""Row filters of the tranche and the text measures behind them
([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#filters)): the cache build drops rows on
them, voice.py skips the rows `vi_gate` drops, YODAS's ingest hears its rows with `Hearer` and gates them,
eval_sets builds giga-heldout by the same rules, and translate.py rejects replies `broken` flags.

- `spoken`: a transcript as compared, lower-case syllables with numbers written out;
- the evaluation overlap: a row sharing a run of OVERLAP_N syllables (or a whole shorter sentence) with
  a FLEURS vi or MultiMed-ST vi evaluation sentence (data/eval/, built by training.prep.eval_sets;
  `eval_grams`, `overlaps`);
- `vi_gate`: the Vietnamese side's gate on a row's asr_vi (its language, its CER against the transcript);
- `Hearer`: the ASR server's client, its language detected rather than forced.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.tranche.ingest import REPO, SOURCES

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SETS = ("fleurs-vi/dev", "fleurs-vi/test", "multimed-vi/select", "multimed-vi/final")
# Syllables. Over the first tranche, no row shares a 13-syllable run with an evaluation sentence;
# runs of 10 or fewer match only stock broadcast phrases ("cảm ơn quý vị và các bạn đã quan tâm", 80 rows).
OVERLAP_N = 13
# An evaluation sentence of MIN_SENTENCE..OVERLAP_N - 1 syllables is matched whole; shorter ones
# ("được", "alo", "vùng nông thôn") would match everyday speech, not leaked content.
MIN_SENTENCE = 9
# Spellings the corpora mix for the same number word, folded to num2words' choice.
NUMBER_VARIANTS = {"ngàn": "nghìn", "linh": "lẻ", "nhăm": "lăm"}
# A longer run of digits is an id or an ASR loop, read digit by digit: num2words overflows on one of
# hundreds of digits (a YODAS row's transcript, 2026-11-25).
MAX_DIGITS = 15
ASR_URL = "http://localhost:8100/v1/audio/transcriptions"
ASR_MAX_TOKENS = 512  # as voice.py: a looping transcript comes back long instead of running on
ASR_TIMEOUT_S = 60
# A letter outside the Latin blocks Vietnamese uses (Basic to Extended-B, Extended Additional).
NON_LATIN = re.compile(r"[^\W\d_\x00-ɏḀ-ỿ]")


def spoken(text: str) -> list[str]:
    """Lower-case NFC syllables without punctuation, numbers written out as Vietnamese words:
    PhoAudiobook and GigaSpeech 2 spell numbers, while Qwen3-ASR and FLEURS write digits (on
    PhoAudiobook, digit rows were 48% of those over 0.1 CER before this)."""
    from num2words import num2words

    def say(digits: str) -> str:
        if len(digits) > MAX_DIGITS:
            return " ".join(num2words(int(d), lang="vi") for d in digits)
        return num2words(int(digits), lang="vi")

    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)  # 1.500.000: thousands separators
    text = re.sub(r"\d+", lambda m: f" {say(m.group())} ", text)
    return [NUMBER_VARIANTS.get(w, w) for w in re.sub(r"[^\w\s]", " ", text).split()]


Grams = dict[int, set[tuple[str, ...]]]  # evaluation runs by length


def runs(words: list[str]) -> set[tuple[str, ...]]:
    """A sentence's runs as the overlap check matches them: every OVERLAP_N-syllable run, or the
    whole sentence of MIN_SENTENCE..OVERLAP_N - 1 syllables; none for a shorter one."""
    if len(words) < MIN_SENTENCE:
        return set()
    n = min(len(words), OVERLAP_N)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def grams_of(texts: Iterable[str]) -> Grams:
    """The runs of these sentences, by length."""
    grams: Grams = {}
    for text in texts:
        if found := runs(spoken(text)):
            grams.setdefault(len(next(iter(found))), set()).update(found)
    return grams


def eval_grams(eval_dir: Path = REPO_ROOT / "data/eval") -> Grams:
    """The runs of the evaluation sentences."""
    import json

    return grams_of(json.loads(line)["text_vi"] for name in EVAL_SETS for line in open(eval_dir / f"{name}.jsonl"))


def overlaps(text: str, grams: Grams) -> bool:
    """Whether the transcript holds one of the runs."""
    w = spoken(text)
    return any(tuple(w[i:i + n]) in found for n, found in grams.items() for i in range(len(w) - n + 1))


def matches(text: str, grams: Grams) -> set[tuple[str, ...]]:
    """The runs the transcript holds."""
    w = spoken(text)
    return {run for n, found in grams.items() for i in range(len(w) - n + 1) if (run := tuple(w[i:i + n])) in found}


def training_texts(revision: str) -> Iterator[tuple[str, list[tuple[str, str, str]]]]:
    """(source, its (id, doc, text) rows in their order) for each segments shard of REPO at
    `revision` but the held-out ones (`Source.heldout`), sixteen shards read at a time; only those
    columns' chunks are read, not the audio. A doc's rows are consecutive (ingest)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def read(path: str) -> list[tuple[str, str, str]]:
        with fs.open(path, "rb", cache_type="none") as fh:
            t = pq.ParquetFile(fh).read(columns=["id", "doc", "text"])
        return list(zip(*t.to_pydict().values()))

    with ThreadPoolExecutor(16) as pool:
        for source, src in SOURCES.items():
            shards = [p for p in sorted(fs.glob(f"datasets/{REPO}@{revision}/{source}/segments/*.parquet"))
                      if p.rsplit("/", 1)[1] not in src.heldout]
            for i in range(0, len(shards), 16):
                for rows in pool.map(read, shards[i:i + 16]):
                    yield source, rows


# Above 0.15 the corpus transcript and the clip differ by phrases (PhoAudiobook's repeated clauses);
# from 0.10 to 0.15 mostly the ASR's misses on names and loanwords, numbers and percentages spelled
# otherwise, a word at a cut edge (tranche-cache#gates).
MAX_CER = 0.15
# GigaSpeech 2's transcripts drop these from numbers ("hơn một chuyến" where the speaker says "hơn
# một nghìn chuyến"), a wrong fact the CER barely sees (tranche-cache#gates).
MAGNITUDES = ("nghìn", "triệu", "tỷ")


def magnitudes(text: str) -> list[int]:
    words = spoken(text)
    return [words.count(m) for m in MAGNITUDES]


def vi_gate(row: dict, max_cer: float = MAX_CER) -> str | None:
    """The first Vietnamese-side filter that drops a row (its `text` and its asr_vi part's columns):
    the cache build drops such rows, so voice.py does not voice them."""
    if row["language"] != "Vietnamese":
        return f"language {row['language'] or 'none'}"
    if round(row["cer"], 6) > max_cer:  # float32: 0.1 is stored as 0.10000000149
        return f"CER {round(row['cer'], 6)} over {max_cer}"
    # Word by word: a swapped magnitude (the ASR's tỷ for the transcript's triệu) is a wrong fact.
    if any(heard > written for heard, written in zip(magnitudes(row["asr_vi"]), magnitudes(row["text"]))):
        return "the ASR hears a nghìn/triệu/tỷ the transcript lacks"
    return None


def broken(en: str, vi: str) -> bool:
    """Whether a translation holds letters of a script its English lacks: Muse, Gemini and, now and
    then, Sonnet write some mid-word (`kính畏`, `thế là спас`, `征 dụng`), and the TTS would speak them.
    Where the English has them too (`京都`), they stay."""
    return bool(NON_LATIN.search(vi)) and not NON_LATIN.search(en)


class Hearer:
    """One HTTP client to the ASR server; each call sends one clip as 16 kHz WAV."""

    def __init__(self, url: str, concurrency: int):
        import httpx

        self.url = url
        self.client = httpx.Client(timeout=ASR_TIMEOUT_S, limits=httpx.Limits(max_connections=concurrency))

    def hear(self, audio: bytes) -> tuple[str, str]:
        """WAV bytes -> (language, transcript). Unforced, the model writes `language <Name><asr_text><text>`,
        and `language None` for audio with no speech; a reply without the header is all
        transcript, its language unknown (""), as qwen_asr's own parser reads it. Clips over 30 s,
        which the server would split and join headers into, do not occur (rows are under 22.4 s)."""
        import httpx

        for attempt in range(3):  # the server drops an idle keep-alive connection now and then
            try:
                reply = self.client.post(self.url, files={"file": ("a.wav", audio, "audio/wav")},
                                         data={"model": "asr", "max_completion_tokens": ASR_MAX_TOKENS})
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
        reply.raise_for_status()
        raw = reply.json()["text"]
        head, tag, text = raw.partition("<asr_text>")
        if not tag or not head.startswith("language "):
            return "", raw.strip()
        language = head.removeprefix("language ").strip()
        return ("", "") if language == "None" else (language, text.strip())
