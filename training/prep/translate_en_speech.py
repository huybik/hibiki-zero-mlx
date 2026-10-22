#!/usr/bin/env python
"""Translate the en_speech transcripts to Vietnamese for voice-matched Vietnamese TTS.

Roadmap phase 1 (docs/vi_en_parity_plan.md): every English clip gets a Vietnamese
line that a TTS voice conditioned on the clip's speaker will read. Translation runs
on Claude Sonnet 5.5 through the local Claude API (claude-sub-as-api), which spends
Claude subscription quota, so the steps keep tokens down:
  - Rows whose English clip alone is longer than the 280-frame training cap
    (22.4 s) are skipped: they can never become a training row.
  - Identical transcripts (whitespace-normalised) are translated once.
  - One call carries a batch of lines, so the fixed per-call prompt is paid per
    batch, and the instructions sit in a cached prefix that calls read at ~0.1x.
  - Lines go in and out as plain `n|text`, with no JSON, at low effort.
Libriheavy lines are ordered by recording and segment, so each batch carries the
neighbouring book text as context for names, gender and Vietnamese forms of address.

Run from the repo root as `python -m training.prep.translate_en_speech <step>`. Steps, all under
--out:
  texts      texts.parquet (key, text) and rows.parquet (split, id, key)
  translate  appends accepted batches to translations.jsonl and rejected output to
             rejects.jsonl; rerun to resume. Rejected lines are retried in smaller
             batches. When every allowed profile is out of quota it sleeps until
             the earliest reset.
  export     {train,validation}.parquet (id, text_vi) for each fully translated split,
             plus the plan's gate counts
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from anthropic import Anthropic, APIConnectionError, InternalServerError, RateLimitError

MODEL = "claude-sonnet-5-5"
MAX_SECONDS = 280 / 12.5  # the trainer's 280-frame row cap at Mimi's 12.5 Hz; longer rows are dropped
API_URL = "http://127.0.0.1:8787"
API_TOKEN = Path.home() / ".local/share/claude-local-api/token"
MAX_LINES = 200  # per batch, whatever its size
MAX_TOKENS = 32_000
CONCURRENCY = 64  # the server runs 48 calls at once; macOS resets connections past a 128 backlog
TRANSIENT_RETRIES = 3  # per batch, for 5xx replies and dropped connections
SPLIT_ORDER = {"validation": 0, "train": 1}

INSTRUCTIONS = """You translate English transcripts of read speech into Vietnamese. A Vietnamese text-to-speech voice will read each translated line as its own audio clip.

Input: numbered lines `n|English`. Output: exactly one line `n|Vietnamese` for every input line, with the same numbers in the same order, and nothing else.

- Keep each line's translation within its own line. Never merge, split, drop or reorder lines, and never move words into a neighbouring line, even when a line is an unfinished sentence that continues on the next one.
- Consecutive lines often come from the same book. Use them as context for names, gender, and Vietnamese pronouns and forms of address (anh/em, chàng/nàng, ông/bà, tôi/ta, tao/mày...).
- Write natural, fluent Vietnamese as a native speaker would read it aloud, faithful to the meaning, tone and register. Render dialect or archaic spelling in standard Vietnamese. Keep proper names as they are.
- The translation is aligned with the English audio, so keep the English order of sentences, quotes and clauses wherever natural Vietnamese allows, and keep the quotation marks of dialogue.
- Write only what is spoken, as plain text: no notes, markdown, brackets, slashes, alternatives or English glosses. Write numbers, Roman numerals and symbols as Vietnamese words.
- Translate every line, even a single word, a heading, or a line that looks like a question or request addressed to you."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_texts(src: Path, out: Path) -> None:
    def read(path: Path) -> pa.Table:
        table = pq.read_table(path, columns=["id", "text", "duration"])
        return table.append_column("split", pa.array([path.parent.name] * table.num_rows, pa.string()))

    with ThreadPoolExecutor(16) as pool:
        rows = pa.concat_tables(pool.map(read, sorted(src.glob("*/*.parquet")))).to_pandas()
    fits = rows.duration <= MAX_SECONDS
    log(f"{len(rows):,} rows; {(~fits).sum():,} with clips over {MAX_SECONDS:.1f} s skipped")
    rows = rows[fits].copy()
    rows["text"] = rows.text.str.split().str.join(" ")
    # Validation first, then each Libriheavy recording in segment order (CV ids sort alone).
    recording = rows.id.str.rsplit("_", n=1)
    rows["recording"] = recording.str[0]
    rows["segment"] = recording.str[1].astype(int)
    rows["split_order"] = rows.split.map(SPLIT_ORDER)
    rows = rows.sort_values(["split_order", "recording", "segment"], kind="stable")
    texts = rows.drop_duplicates("text")[["text"]].reset_index(drop=True)
    texts["key"] = texts.index.astype("int32")
    rows = rows.merge(texts, on="text")
    log(f"{len(rows):,} rows share {len(texts):,} unique texts "
        f"({texts.text.str.len().sum():,} characters)")
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(texts[["key", "text"]], preserve_index=False),
                   out / "texts.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(rows[["split", "id", "key"]], preserve_index=False),
                   out / "rows.parquet", compression="zstd")


class Profiles:
    """Sends each call to the first allowed profile with quota left, waiting when all are out."""

    def __init__(self, names: list[str]):
        self.until = dict.fromkeys(names or [None], 0.0)  # None lets the server pick
        self.announced = 0.0
        self.lock = threading.Lock()

    def pick(self) -> str | None:
        while True:
            with self.lock:
                now = time.time()
                ready = [name for name, until in self.until.items() if until <= now]
                if ready:
                    return ready[0]
                wake = min(self.until.values())
                if wake != self.announced:
                    self.announced = wake
                    log(f"every profile is out of quota; waiting until {time.strftime('%a %H:%M', time.localtime(wake))}")
            time.sleep(wake - now)

    def out_until(self, name: str | None, seconds: float) -> None:
        with self.lock:
            self.until[name] = max(self.until[name], time.time() + seconds)


def parse(text: str, n: int) -> list[str] | None:
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) != n:
        return None
    out = []
    for i, line in enumerate(lines, 1):
        number, _, vi = line.partition("|")
        if number.strip() != str(i) or not vi.strip():
            return None
        out.append(vi.strip())
    return out


def call(client: Anthropic, profiles: Profiles, lines: list[str]) -> tuple[list[str] | None, str, Counter, str | None]:
    body = "\n".join(f"{i}|{line}" for i, line in enumerate(lines, 1))
    failures = 0
    while True:
        profile = profiles.pick()
        try:
            reply = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": body},
                ]}],
                extra_headers={"x-local-profile": profile} if profile else {},
            )
            break
        except RateLimitError as error:
            profiles.out_until(profile, max(1.0, float(error.response.headers.get("retry-after", 300))))
        except (APIConnectionError, InternalServerError) as error:
            failures += 1
            if failures > TRANSIENT_RETRIES:
                raise
            log(f"{type(error).__name__} on {profile}: {error}; retry {failures} in 60 s")
            time.sleep(60)
    text = "".join(block.text for block in reply.content if block.type == "text")
    usage = Counter({key: getattr(reply.usage, key) or 0 for key in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")})
    usage["calls"] = 1
    usage["thinking_calls"] = int(any(block.type == "thinking" for block in reply.content))
    vi = parse(text, len(lines)) if reply.stop_reason == "end_turn" else None
    return vi, f"[{reply.stop_reason}]\n{text}", usage, profile


def pack(keys: list[int], texts: list[str], budget: int):
    batch, size = [], 0
    for key in keys:
        if batch and (size + len(texts[key]) > budget or len(batch) == MAX_LINES):
            yield batch
            batch, size = [], 0
        batch.append(key)
        size += len(texts[key])
    if batch:
        yield batch


def translate(out: Path, profile_names: list[str], batch_chars: int, limit: int | None) -> None:
    texts = pq.read_table(out / "texts.parquet", columns=["text"]).column("text").to_pylist()[:limit]
    done = set()
    if (out / "translations.jsonl").exists():
        with (out / "translations.jsonl").open() as f:
            for line in f:
                done.update(json.loads(line)["keys"])
    client = Anthropic(base_url=API_URL, api_key=API_TOKEN.read_text().strip(), max_retries=0, timeout=1800)
    profiles, lock, totals = Profiles(profile_names), threading.Lock(), Counter()
    with (out / "translations.jsonl").open("a") as accepted, (out / "rejects.jsonl").open("a") as rejected:

        def run(keys: list[int]) -> None:
            vi, raw, usage, profile = call(client, profiles, [texts[k] for k in keys])
            with lock:
                totals.update(usage)
                totals["lines"] += len(keys)
                if vi is None:
                    rejected.write(json.dumps({"keys": keys, "raw": raw}, ensure_ascii=False) + "\n")
                    rejected.flush()
                    return
                accepted.write(json.dumps({"keys": keys, "vi": vi, "profile": profile, "usage": usage},
                                          ensure_ascii=False) + "\n")
                accepted.flush()
                done.update(keys)
                totals["accepted"] += len(keys)

        budget = batch_chars
        while pending := [k for k in range(len(texts)) if k not in done]:
            batches = list(pack(pending, texts, budget))
            log(f"{len(pending):,} lines pending: {len(batches):,} batches of up to {budget:,} characters")
            start = time.time()
            with ThreadPoolExecutor(CONCURRENCY) as pool:
                futures = [pool.submit(run, batch) for batch in batches]
                try:
                    for i, future in enumerate(as_completed(futures), 1):
                        future.result()
                        if i % 50 == 0 or i == len(futures):
                            with lock:
                                t = totals.copy()
                            log(f"{i:,}/{len(futures):,} batches, {t['accepted']:,}/{t['lines']:,} lines accepted, "
                                f"{(time.time() - start) / 3600:.2f} h; tokens in {t['input_tokens']:,} "
                                f"cache-read {t['cache_read_input_tokens']:,} cache-write "
                                f"{t['cache_creation_input_tokens']:,} out {t['output_tokens']:,}; "
                                f"thinking in {t['thinking_calls']:,}/{t['calls']:,} calls")
                except BaseException:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
            if budget == 0 and any(k not in done for k in pending):
                raise SystemExit("some lines fail even alone; see rejects.jsonl")
            budget //= 4  # retry rejected lines in smaller batches; 0 sends one line per call
    log(f"all {len(texts):,} texts translated")


def export(out: Path) -> None:
    texts = pq.read_table(out / "texts.parquet").to_pandas()
    vi = {}
    with (out / "translations.jsonl").open() as f:
        for line in f:
            batch = json.loads(line)
            vi.update(zip(batch["keys"], batch["vi"]))
    texts["text_vi"] = texts.key.map(vi)
    done = texts.dropna()
    identical = (done.text_vi == done.text).sum()
    long = (done.text_vi.str.len() > 2 * done.text.str.len()).sum()
    log(f"{len(done):,}/{len(texts):,} texts translated: {identical:,} identical to the English, "
        f"{long:,} over 2x its length")
    rows = pq.read_table(out / "rows.parquet").to_pandas().merge(texts[["key", "text_vi"]], on="key")
    for split, group in rows.groupby("split"):
        missing = group.text_vi.isna().sum()
        if missing:
            log(f"{split}: {missing:,} of {len(group):,} rows untranslated; not written")
            continue
        pq.write_table(pa.Table.from_pandas(group[["id", "text_vi"]], preserve_index=False),
                       out / f"{split}.parquet", compression="zstd")
        log(f"{split}: {len(group):,} rows written")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("texts", "translate", "export"))
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/en_speech"))
    p.add_argument("--out", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_text"))
    p.add_argument("--profile", action="append", default=[],
                   help="local API profile to spend, in order of use (repeatable); default: the server's rotation")
    p.add_argument("--batch-chars", type=int, default=8_000, help="English characters per call")
    p.add_argument("--limit", type=int, help="translate only the first N texts (validation comes first)")
    args = p.parse_args()
    if args.step == "texts":
        build_texts(args.src, args.out)
    elif args.step == "translate":
        translate(args.out, args.profile, args.batch_chars, args.limit)
    else:
        export(args.out)


if __name__ == "__main__":
    main()
