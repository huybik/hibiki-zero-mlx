#!/usr/bin/env python
"""Translate the en_speech transcripts to Vietnamese for voice-matched Vietnamese TTS.

Roadmap phase 1 (docs/vi_en_parity_plan.md): every English clip gets a Vietnamese
line that a TTS voice conditioned on the clip's speaker will read. Translation runs
on Claude Sonnet 5.5 through the local Claude API (claude-sub-as-api), which spends
Claude subscription quota, and, for the Libriheavy texts claimed for them, on Gemini 3.8
Flash through the paid Gemini Batch API or on Muse Spark 1.3 Contributor through the paid
OpenRouter API. The steps keep tokens down:
  - The corpus holds only clips that fit the trainer's 280-frame cap (22.4 s), so
    every row is kept except transcripts without letters; `texts` stops if a
    longer clip is found.
  - Identical transcripts (whitespace-normalised) are translated once.
  - One call carries a batch of lines, so the fixed per-call prompt is paid per
    batch, and the instructions sit in a cached prefix that calls read at ~0.1x.
  - Lines go in and out as plain `n|text`, with no JSON, at low effort.
Libriheavy lines are ordered by recording and segment, so each batch carries the
neighbouring book text as context for names, gender and Vietnamese forms of address.

Run from the repo root as `python -m training.prep.translate_en_speech <step>`. Steps, all under
--out:
  texts      texts.parquet (text: the unique transcripts, in translation order) and
             rows.parquet (split, id, text, duration) for every en_speech row of the dataset
  translate  keeps --concurrency calls in flight and appends accepted batches, keyed by
             their English text, to translations.jsonl and rejected output to
             rejects.jsonl; rerun to resume. Rejected lines are retried in smaller
             batches, then alone. When every --profile is out of quota it sleeps until
             the earliest reset. Claimed texts are skipped.
  claim      claims.parquet: --hours more of untranslated Libriheavy audio for each --key, or
             all of it, from the end of the translation order; run it while translate is
             stopped
  gemini     sends each --key's claimed lines as a Batch API job, appends accepted batches
             to gemini/translations.jsonl and rejected output to gemini/rejects.jsonl, and
             resubmits rejected lines in smaller batches, then alone; lines that fail alone
             go back to translate. Rerun to resume
  muse       translates the one --key's claimed lines as translate does, into
             muse/translations.jsonl and muse/rejects.jsonl; lines that fail alone go back
             to translate. Rerun to resume
  export     {train,validation}.parquet (id, text_vi) for each fully translated split,
             plus the plan's gate counts
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from anthropic import Anthropic, APIConnectionError, InternalServerError, RateLimitError
from google import genai
from google.genai import types

MODEL = "claude-sonnet-5-5"
MAX_SECONDS = 280 / 12.5  # the trainer's 280-frame row cap at Mimi's 12.5 Hz; longer rows are dropped
LETTER = re.compile(r"[^\W\d_]")  # a transcript without one ("." alone) has nothing to translate
API_URL = "http://127.0.0.1:8787"
API_TOKEN = Path.home() / ".local/share/claude-local-api/token"
MAX_LINES = 200  # per batch, whatever its size
MAX_TOKENS = 32_000
SERVER_SLOTS = 48  # calls the local API runs at once; more in flight only queue
QUOTA_CHECK = 60  # seconds between wall-clock checks while every profile is out of quota
TRANSIENT_RETRIES = 5  # per batch, for 5xx, dropped connections and one-off 429s: ~30 min of backoff
RETRY_WAIT = 60  # seconds before the first retry; doubles on each further retry
SPLIT_ORDER = {"validation": 0, "train": 1}
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_POLL = 60  # seconds between checks on running batch jobs
GEMINI_TIMEOUT = 600_000  # ms an HTTP read may stall before the call fails and is retried
MUSE_MODEL = "meta/muse-spark-1.3-contributor"  # Meta trains on its input and output: Libriheavy only
MUSE_URL = "https://openrouter.ai/api/v1/chat/completions"
MUSE_TIMEOUT = 600  # seconds an HTTP read may stall before the call fails and is retried

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
        table = pq.read_table(path, columns=["id", "text_en", "en_duration_s"]).rename_columns(["id", "text", "duration"])
        return table.append_column("split", pa.array([path.parent.name] * table.num_rows, pa.string()))

    # The en_speech shards of the EN-VI dataset; PhoMT comes with its own Vietnamese.
    shards = sorted(p for p in src.glob("data/*/*.parquet") if not p.name.startswith("phomt-"))
    with ThreadPoolExecutor(16) as pool:
        rows = pa.concat_tables(pool.map(read, shards)).to_pandas()
    rows["text"] = rows.text.str.split().str.join(" ")
    if (too_long := (rows.duration > MAX_SECONDS).sum()):
        raise SystemExit(f"{too_long:,} clips last over {MAX_SECONDS:.1f} s and can never become a training row; "
                         "drop them from the dataset first")
    spoken = rows.text.map(lambda x: bool(LETTER.search(x)))
    log(f"{len(rows):,} rows; skipped {(~spoken).sum():,} without letters")
    rows = rows[spoken].copy()
    recording = rows.id.str.rsplit("_", n=1)
    rows["recording"] = recording.str[0]
    rows["segment"] = recording.str[1].astype(int)
    for (split, cv), group in rows.groupby([rows.split, rows.id.str.startswith("cv27/")]):
        log(f"{split} {'Common Voice' if cv else 'Libriheavy'}: {len(group):,} rows, "
            f"{group.duration.sum() / 3600:,.1f} h")
    # Validation first, then each Libriheavy recording in segment order (CV ids sort alone).
    rows["split_order"] = rows.split.map(SPLIT_ORDER)
    rows = rows.sort_values(["split_order", "recording", "segment"], kind="stable")
    texts = rows.drop_duplicates("text")[["text"]]
    log(f"{len(rows):,} rows share {len(texts):,} unique texts "
        f"({texts.text.str.len().sum():,} characters)")
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(texts, preserve_index=False), out / "texts.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(rows[["split", "id", "text", "duration"]], preserve_index=False),
                   out / "rows.parquet", compression="zstd")


class Stopped(Exception):
    """The run is shutting down."""


class Profiles:
    """Sends each call to the first named profile with quota left, waiting when all are out."""

    def __init__(self, names: list[str]):
        self.until = dict.fromkeys(names, 0.0)
        self.announced = ""
        self.stop = threading.Event()  # set on shutdown, so no thread keeps waiting for quota
        self.lock = threading.Lock()

    def sleep(self, seconds: float) -> None:
        if self.stop.wait(seconds):
            raise Stopped

    def pick(self) -> str:
        while True:
            if self.stop.is_set():
                raise Stopped
            with self.lock:
                now = time.time()
                ready = [name for name, until in self.until.items() if until <= now]
                if ready:
                    return ready[0]
                wake = min(self.until.values())
                # Concurrent 429s move `wake` by milliseconds; log once per minute shown.
                if (shown := time.strftime("%a %H:%M", time.localtime(wake))) != self.announced:
                    self.announced = shown
                    log(f"every profile is out of quota; waiting until {shown}")
            # A timed wait stops counting while the Mac sleeps, so check the clock again often.
            self.sleep(min(wake - now, QUOTA_CHECK))

    def out_until(self, name: str, seconds: float) -> None:
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


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}|{line}" for i, line in enumerate(lines, 1))


def call(client: Anthropic, profiles: Profiles, lines: list[str]) -> tuple[list[str] | None, str, Counter, dict]:
    failures = 0
    while True:
        profile = profiles.pick()
        try:
            reply = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": numbered(lines)},
                ]}],
                # A retried batch must be generated again, not served from the response cache.
                extra_headers={"x-local-profile": profile, "x-local-cache": "false"},
            )
            break
        except RateLimitError as error:
            retry_after = error.response.headers.get("retry-after")
            if retry_after is not None:  # the profile is out of quota until then
                profiles.out_until(profile, max(1.0, float(retry_after)))
                continue
            transient = error  # only this request was refused
        except (APIConnectionError, InternalServerError) as error:
            transient = error
        failures += 1
        if failures > TRANSIENT_RETRIES:
            raise transient
        wait = RETRY_WAIT * 2 ** (failures - 1)
        log(f"{type(transient).__name__} on {profile}: {transient}; retry {failures} in {wait} s")
        profiles.sleep(wait)
    text = "".join(block.text for block in reply.content if block.type == "text")
    usage = Counter({key: getattr(reply.usage, key) or 0 for key in (
        "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")})
    usage["calls"] = 1
    usage["thinking_calls"] = int(any(block.type == "thinking" for block in reply.content))
    vi = parse(text, len(lines)) if reply.stop_reason == "end_turn" else None
    return vi, f"[{reply.stop_reason}]\n{text}", usage, {"profile": profile}


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


def read_batches(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for number, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"{path}:{number} is not a complete JSON line, likely cut off by a "
                                 "killed run; delete that line to continue") from None


def translated(path: Path) -> set[str]:
    return {en for batch in read_batches(path) for en in batch["en"]}


def check_tail(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                raise SystemExit(f"{path} ends in a cut-off line from a killed run; delete that line to continue")


def exclusive(path: Path, busy: str):
    lock_file = path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise SystemExit(busy) from None
    return lock_file


def read_claims(out: Path) -> dict[str, str]:
    path = out / "claims.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    return dict(zip(table.column("text").to_pylist(), table.column("key").to_pylist()))


def rounds(texts: list[str], pending: list[int], call, stop: threading.Event, done: set[str], accepted, rejected,
           batch_chars: int, concurrency: int) -> list[str]:
    """Translate the pending lines in batches, their rejected lines in quarter-size batches, then alone.

    `call(lines)` returns the Vietnamese lines or None, the raw reply, the usage and the accepted record's
    other fields. Returns the lines that fail even alone.
    """
    lock, totals, budget = threading.Lock(), Counter(), batch_chars

    def run(keys: list[int]) -> None:
        lines = [texts[k] for k in keys]
        vi, raw, usage, fields = call(lines)
        with lock:
            totals.update(usage)
            totals["lines"] += len(lines)
            if vi is None:
                rejected.write(json.dumps({"en": lines, "raw": raw}, ensure_ascii=False) + "\n")
                rejected.flush()
                return
            accepted.write(json.dumps({"en": lines, "vi": vi, **fields, "usage": usage}, ensure_ascii=False) + "\n")
            accepted.flush()
            done.update(lines)
            totals["accepted"] += len(lines)

    while pending:
        batches = list(pack(pending, texts, budget))
        log(f"{len(pending):,} lines pending: {len(batches):,} batches of up to {budget:,} characters")
        start = time.time()
        with ThreadPoolExecutor(concurrency) as pool:
            try:
                futures = [pool.submit(run, batch) for batch in batches]
                for i, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if i % 50 == 0 or i == len(futures):
                        with lock:
                            t = totals.copy()
                        usage = ", ".join(f"{name} {t[name]:,}" for name in sorted(t) if name not in ("lines", "accepted"))
                        log(f"{i:,}/{len(futures):,} batches, {t['accepted']:,}/{t['lines']:,} lines accepted, "
                            f"{(time.time() - start) / 3600:.2f} h; {usage}")
            except BaseException:
                stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
        pending = [k for k in pending if texts[k] not in done]
        if pending and budget == 0:
            return [texts[k] for k in pending]
        budget = budget // 4 if budget > 500 else 0  # retry rejected lines in smaller batches, then alone
    return []


def translate(out: Path, profile_names: list[str], batch_chars: int, concurrency: int) -> None:
    texts = pq.read_table(out / "texts.parquet", columns=["text"]).column("text").to_pylist()
    with exclusive(out / "translate.lock", f"another translate run is using {out}"):
        log_path = out / "translations.jsonl"
        check_tail(log_path)
        done = translated(log_path)
        claimed = set(read_claims(out))  # read once: claims are only made while translate is stopped
        if claimed:
            log(f"skipping {len(claimed):,} claimed texts")
        client = Anthropic(base_url=API_URL, api_key=API_TOKEN.read_text().strip(), max_retries=0, timeout=1800)
        profiles = Profiles(profile_names)
        pending = [k for k, text in enumerate(texts) if text not in done and text not in claimed]
        with log_path.open("a") as accepted, (out / "rejects.jsonl").open("a") as rejected:
            failed = rounds(texts, pending, lambda lines: call(client, profiles, lines), profiles.stop, done,
                            accepted, rejected, batch_chars, concurrency)
        if failed:
            raise SystemExit(f"{len(failed):,} lines fail even alone; see rejects.jsonl")
    log(f"all {len(texts):,} texts translated or claimed")


def write_claims(out: Path, claims: dict[str, str]) -> None:
    path = out / "claims.parquet"
    pq.write_table(pa.table({"text": list(claims), "key": list(claims.values())}), path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def common_voice(out: Path) -> set[str]:
    """Texts with a Common Voice row, which never go to Google or Meta."""
    rows = pq.read_table(out / "rows.parquet", columns=["id", "text"]).to_pandas()
    return set(rows.text[rows.id.str.startswith("cv27/")])


def check_claims(out: Path, claims: dict[str, str]) -> None:
    """Stop a paid run if a corpus rebuild has given a claimed text a Common Voice row."""
    if any(text in claims for text in common_voice(out)):
        raise SystemExit("claimed texts now have a Common Voice row; run the claim step, which releases them")


def claim(out: Path, keys: list[str], hours: float | None) -> None:
    """Give each key `hours` more of untranslated Libriheavy audio, or all of it, from the end of the translation order."""
    texts = pq.read_table(out / "texts.parquet", columns=["text"]).column("text").to_pylist()
    seconds = pq.read_table(out / "rows.parquet", columns=["text", "duration"]).to_pandas() \
        .groupby("text").duration.sum().to_dict()
    cv = common_voice(out)
    # translate reads the claims once at start, so it must not run while they change.
    with exclusive(out / "translate.lock", f"translate or export is using {out}; stop translate before claiming, "
                                           "and start it again afterwards so it skips the claimed texts"), \
            exclusive(out / "claims.lock", f"a gemini or muse run is using {out}"):
        claims = read_claims(out)
        # A round's smaller budget would count newly claimed lines as rejected ones.
        if busy := [key for key in keys if (out / "gemini" / f"{key}.job.json").exists()]:
            raise SystemExit(f"{', '.join(busy)} have a gemini round in progress; let gemini finish first")
        if moved := [text for text in claims if text in cv]:
            for text in moved:
                del claims[text]
            log(f"released {len(moved):,} claimed texts that now have a Common Voice row")
        done = translated(out / "translations.jsonl")
        # Lines a paid engine released after they failed alone go to translate.
        refused = {text for engine in ("gemini", "muse") for batch in read_batches(out / engine / "rejects.jsonl")
                   for text in batch["en"]}
        free = (text for text in reversed(texts)
                if text not in done and text not in claims and text not in cv and text not in refused)
        for key in keys:
            total, count = 0.0, 0
            for text in free:
                claims[text] = key
                total, count = total + seconds[text], count + 1
                if hours is not None and total >= hours * 3600:
                    break
            if not count:
                raise SystemExit(f"no untranslated Libriheavy text is left to claim for {key}")
            log(f"claimed {count:,} texts, {total / 3600:,.1f} h of audio, for {key}")
        write_claims(out, claims)


def gemini_request(lines: list[str]) -> dict:
    return {"system_instruction": {"parts": [{"text": INSTRUCTIONS}]},
            "contents": [{"role": "user", "parts": [{"text": numbered(lines)}]}],
            "generation_config": {"max_output_tokens": MAX_TOKENS, "thinking_config": {"thinking_level": "LOW"}}}


def read_result(result: dict | None, n: int) -> tuple[list[str] | None, str, Counter]:
    if result is None:
        return None, "[missing from the results]", Counter()
    if not (response := result.get("response")):
        return None, f"[no response] {json.dumps(result, ensure_ascii=False)}", Counter()
    meta = response.get("usageMetadata", {})
    usage = Counter(input_tokens=meta.get("promptTokenCount", 0), output_tokens=meta.get("candidatesTokenCount", 0),
                    thinking_tokens=meta.get("thoughtsTokenCount", 0))
    candidate = (response.get("candidates") or [{}])[0]
    finish = candidate.get("finishReason")
    text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                   if not part.get("thought"))
    return (parse(text, n) if finish == "STOP" else None), f"[{finish}]\n{text}", usage


class GeminiKey:
    """One key's rounds: a batch job of its pending claimed lines, then its rejected lines in smaller batches."""

    def __init__(self, out: Path, key: str, batch_chars: int):
        retry = types.HttpRetryOptions(attempts=TRANSIENT_RETRIES + 1, initial_delay=RETRY_WAIT,
                                       max_delay=RETRY_WAIT * 2 ** (TRANSIENT_RETRIES - 1))
        self.client = genai.Client(api_key=os.environ[key], http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT, retry_options=retry))
        self.out, self.key, self.batch_chars = out, key, batch_chars
        self.path = out / "gemini" / f"{key}.job.json"
        # {"budget"} between rounds; {"display_name", "batches"} from just before a job is created;
        # plus {"name"} once it is.
        self.job = json.loads(self.path.read_text()) if self.path.exists() else {"budget": batch_chars}
        self.state = ""

    def save(self, job: dict) -> None:
        self.job = job
        self.path.with_suffix(".tmp").write_text(json.dumps(job, ensure_ascii=False))
        self.path.with_suffix(".tmp").replace(self.path)

    def step(self, texts: list[str], claims: dict[str, str], done: set[str], accepted, rejected) -> bool:
        """Submit, check or collect this key's job; False once every claimed line is translated."""
        if "display_name" in self.job and "name" not in self.job:
            self.adopt()
        if "name" not in self.job:
            mine = [k for k, text in enumerate(texts) if claims.get(text) == self.key]
            pending = [k for k in mine if texts[k] not in done]
            if not pending:
                self.path.unlink(missing_ok=True)
                log(f"{self.key}: every claimed text is translated")
                return False
            if self.job["budget"] < self.batch_chars and len(pending) > len(mine) / 10:
                raise SystemExit(f"{self.key}: {len(pending):,} of {len(mine):,} claimed lines were rejected; see "
                                 f"{self.out / 'gemini' / 'rejects.jsonl'} before running again")
            self.submit([[texts[k] for k in batch] for batch in pack(pending, texts, self.job["budget"])])
            return True
        job = self.client.batches.get(name=self.job["name"])
        if job.state.name != self.state:
            self.state = job.state.name
            log(f"{self.key}: {job.name} {self.state}")
        if job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            self.collect(job, claims, done, accepted, rejected)
        elif job.state.name == "JOB_STATE_EXPIRED":  # unfinished after 48 h, with nothing returned
            self.save({"budget": self.job["budget"]})
        elif job.state.name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            self.save({"budget": self.job["budget"]})  # a rerun submits the lines again
            raise SystemExit(f"{self.key}: {job.name} ended {job.state.name}")
        return True

    def adopt(self) -> None:
        """Find the job a stopped run may have created without recording it, so it is not paid for twice."""
        for job in self.client.batches.list():
            if job.display_name == self.job["display_name"]:
                log(f"{self.key}: found {job.name}, created by a stopped run")
                self.save(self.job | {"name": job.name})
                return
        self.save({"budget": self.job["budget"]})

    def submit(self, batches: list[list[str]]) -> None:
        display_name = f"en-vi {self.key} {uuid.uuid4().hex[:8]}"
        self.save({"budget": self.job["budget"], "display_name": display_name, "batches": batches})
        requests = self.out / "gemini" / f"{self.key}.requests.jsonl"
        with requests.open("w") as f:
            for i, lines in enumerate(batches):
                f.write(json.dumps({"key": str(i), "request": gemini_request(lines)}, ensure_ascii=False) + "\n")
        upload = self.client.files.upload(file=requests, config={"display_name": requests.name, "mime_type": "jsonl"})
        # Creating a job is not idempotent: a retried create can start, and bill, a second job.
        job = self.client.batches.create(model=GEMINI_MODEL, src=upload.name, config={
            "display_name": display_name, "http_options": {"retry_options": {"attempts": 1}}})
        requests.unlink()
        self.save(self.job | {"name": job.name})
        log(f"{self.key}: submitted {job.name}: {len(batches):,} batches of up to {self.job['budget']:,} "
            f"characters, {sum(map(len, batches)):,} lines")

    def collect(self, job: types.BatchJob, claims: dict[str, str], done: set[str], accepted, rejected) -> None:
        results = {}
        # Split on newlines only: str.splitlines() also breaks at U+2028 and others, which JSON leaves unescaped.
        for line in self.client.files.download(file=job.dest.file_name).split(b"\n"):
            if line.strip():
                result = json.loads(line)
                results[int(result["key"])] = result
        totals, failed = Counter(), []
        for i, lines in enumerate(self.job["batches"]):
            vi, raw, usage = read_result(results.get(i), len(lines))
            totals.update(usage)
            totals["lines"] += len(lines)
            if vi is None:
                rejected.write(json.dumps({"en": lines, "raw": raw}, ensure_ascii=False) + "\n")
                failed += lines
                continue
            accepted.write(json.dumps({"en": lines, "vi": vi, "key": self.key, "usage": usage},
                                      ensure_ascii=False) + "\n")
            done.update(lines)
            totals["accepted"] += len(lines)
        accepted.flush()
        rejected.flush()
        log(f"{self.key}: {totals['accepted']:,}/{totals['lines']:,} lines accepted; tokens in "
            f"{totals['input_tokens']:,} out {totals['output_tokens']:,} thinking {totals['thinking_tokens']:,}")
        budget = self.job["budget"]
        if failed and budget == 0:
            for text in failed:
                del claims[text]
            write_claims(self.out, claims)
            log(f"{self.key}: {len(failed):,} lines fail even alone; released them to translate, which takes them "
                "when started again")
        self.save({"budget": budget // 4 if budget > 500 else 0})  # rejected lines go again in smaller batches


def gemini(out: Path, keys: list[str], batch_chars: int) -> None:
    texts = pq.read_table(out / "texts.parquet", columns=["text"]).column("text").to_pylist()
    with exclusive(out / "claims.lock", f"a claim, gemini or muse run is using {out}"):
        claims = read_claims(out)
        check_claims(out, claims)
        (out / "gemini").mkdir(exist_ok=True)
        if unclaimed := [key for key in keys if key not in set(claims.values())]:
            raise SystemExit(f"{', '.join(unclaimed)} hold no claimed texts; run the claim step first")
        log_path = out / "gemini" / "translations.jsonl"
        check_tail(log_path)
        done = translated(log_path)
        runs = [GeminiKey(out, key, batch_chars) for key in keys]
        with log_path.open("a") as accepted, (out / "gemini" / "rejects.jsonl").open("a") as rejected:
            while runs := [run for run in runs if run.step(texts, claims, done, accepted, rejected)]:
                time.sleep(GEMINI_POLL)


def muse_call(client: httpx.Client, stop: threading.Event, key: str,
              lines: list[str]) -> tuple[list[str] | None, str, Counter, dict]:
    failures = 0
    while True:
        try:
            response = client.post(MUSE_URL, json={
                "model": MUSE_MODEL, "max_tokens": MAX_TOKENS, "reasoning": {"effort": "minimal"},  # cannot be off
                "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": numbered(lines)}]})
            reply = response.json()
        except (httpx.TransportError, json.JSONDecodeError) as error:
            transient = f"{type(error).__name__}: {error}"
        else:
            error = reply.get("error") or {}  # a 200 may carry only an error, from the provider
            if response.status_code == 200 and not error:
                break
            if response.status_code == 403 and (error.get("metadata") or {}).get("reasons"):
                return None, f"[moderation] {json.dumps(error)}", Counter(calls=1), {}
            # Other 403s are spent key or budget limits; they and other client errors stop the run.
            if response.status_code not in (200, 408, 429) and response.status_code < 500:
                raise SystemExit(f"OpenRouter refused a call with HTTP {response.status_code}: {response.text[:500]}")
            transient = f"HTTP {response.status_code}: {json.dumps(error)[:300]}"
        failures += 1
        if failures > TRANSIENT_RETRIES:
            raise SystemExit(f"a Muse call failed {failures} times; last: {transient}")
        wait = RETRY_WAIT * 2 ** (failures - 1)
        log(f"{key}: {transient}; retry {failures} in {wait} s")
        if stop.wait(wait):
            raise Stopped
    choice, meta = reply["choices"][0], reply["usage"]
    usage = Counter(calls=1, input_tokens=meta["prompt_tokens"], output_tokens=meta["completion_tokens"],
                    reasoning_tokens=(meta.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    text = choice["message"].get("content") or ""
    error = f" {json.dumps(choice['error'])}" if "error" in choice else ""
    vi = parse(text, len(lines)) if choice.get("finish_reason") == "stop" else None
    return vi, f"[{choice.get('finish_reason')}{error}]\n{text}", usage, {"key": key}


def muse(out: Path, key: str, batch_chars: int, concurrency: int) -> None:
    texts = pq.read_table(out / "texts.parquet", columns=["text"]).column("text").to_pylist()
    with exclusive(out / "claims.lock", f"a claim, gemini or muse run is using {out}"):
        claims = read_claims(out)
        check_claims(out, claims)
        mine = [k for k, text in enumerate(texts) if claims.get(text) == key]
        if not mine:
            raise SystemExit(f"{key} holds no claimed texts; run the claim step first")
        (out / "muse").mkdir(exist_ok=True)
        log_path = out / "muse" / "translations.jsonl"
        check_tail(log_path)
        done = translated(log_path)
        pending = [k for k in mine if texts[k] not in done]
        client = httpx.Client(timeout=MUSE_TIMEOUT, limits=httpx.Limits(max_connections=concurrency),
                              headers={"Authorization": f"Bearer {os.environ[key]}"})
        stop = threading.Event()
        with log_path.open("a") as accepted, (out / "muse" / "rejects.jsonl").open("a") as rejected:
            failed = rounds(texts, pending, lambda lines: muse_call(client, stop, key, lines), stop, done,
                            accepted, rejected, batch_chars, concurrency)
        if len(failed) > len(mine) / 10:
            raise SystemExit(f"{len(failed):,} of {len(mine):,} claimed lines fail even alone, a systematic "
                             f"failure; see {out / 'muse' / 'rejects.jsonl'} before running again")
        if failed:
            for text in failed:
                del claims[text]
            write_claims(out, claims)
            log(f"{len(failed):,} lines fail even alone; released them to translate, which takes them when "
                "started again")
    log(f"{key}: every claimed text is translated or released")


def export(out: Path) -> None:
    texts = pq.read_table(out / "texts.parquet").to_pandas()
    vi = {}
    # A running translate, gemini or muse could be mid-write.
    with exclusive(out / "translate.lock", f"a translate run is using {out}"), \
            exclusive(out / "claims.lock", f"a gemini or muse run is using {out}"):
        for path in (out / "translations.jsonl", out / "gemini" / "translations.jsonl",
                     out / "muse" / "translations.jsonl"):
            for batch in read_batches(path):
                vi.update(zip(batch["en"], batch["vi"]))
    texts["text_vi"] = texts.text.map(vi)
    done = texts.dropna()
    identical = (done.text_vi == done.text).sum()
    long = (done.text_vi.str.len() > 2 * done.text.str.len()).sum()
    log(f"{len(done):,}/{len(texts):,} texts translated: {identical:,} identical to the English, "
        f"{long:,} over 2x its length")
    rows = pq.read_table(out / "rows.parquet").to_pandas()
    rows["text_vi"] = rows.text.map(vi)
    for split, group in rows.groupby("split"):
        missing = group.text_vi.isna().sum()
        if missing:
            (out / f"{split}.parquet").unlink(missing_ok=True)
            log(f"{split}: {missing:,} of {len(group):,} rows untranslated; not written")
            continue
        pq.write_table(pa.Table.from_pandas(group[["id", "text_vi"]], preserve_index=False),
                       out / f"{split}.parquet", compression="zstd")
        log(f"{split}: {len(group):,} rows written")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("texts", "claim", "translate", "gemini", "muse", "export"))
    p.add_argument("--src", type=Path, default=Path("/Volumes/data/datasets/en_vi_speech"), help="the EN-VI dataset")
    p.add_argument("--out", type=Path, default=Path("/Volumes/data/datasets/en_speech_vi_text"))
    p.add_argument("--profile", action="append", default=[],
                   help="local API profile to spend, in order of use (repeatable; translate needs one)")
    p.add_argument("--key", action="append", default=[],
                   help="environment variable holding a Gemini or OpenRouter API key (claim, gemini, muse; "
                        "repeatable, muse takes one)")
    p.add_argument("--hours", type=float,
                   help="audio hours to claim for each --key (claim); all untranslated Libriheavy when omitted")
    p.add_argument("--batch-chars", type=int, default=8_000, help="English characters per call")
    p.add_argument("--concurrency", type=int, default=8,
                   help=f"calls in flight (translate: at most {SERVER_SLOTS}, and quota drains in proportion; muse)")
    args = p.parse_args()
    if args.step == "texts":
        build_texts(args.src, args.out)
    elif args.step == "translate":
        if not args.profile:
            p.error("translate needs --profile: name the subscriptions it may spend")
        if not 1 <= args.concurrency <= SERVER_SLOTS:
            p.error(f"--concurrency must be 1 to {SERVER_SLOTS}, the calls the local API runs at once")
    elif args.step in ("claim", "gemini", "muse"):
        if not args.key:
            p.error(f"{args.step} needs --key: name the environment variables holding the API keys")
        if missing := [key for key in args.key if not os.environ.get(key)]:
            p.error(f"{', '.join(missing)} not set; load .env first")
        if args.step == "muse" and len(args.key) != 1:
            p.error("muse takes one --key")
    if args.step in ("translate", "gemini", "muse"):
        # A background run ignores SIGINT; let `kill` stop it as cleanly as Ctrl-C does.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.default_int_handler)
    if args.step == "claim":
        claim(args.out, args.key, args.hours)
    elif args.step == "translate":
        translate(args.out, args.profile, args.batch_chars, args.concurrency)
    elif args.step == "gemini":
        gemini(args.out, args.key, args.batch_chars)
    elif args.step == "muse":
        muse(args.out, args.key[0], args.batch_chars, args.concurrency)
    elif args.step == "export":
        export(args.out)


if __name__ == "__main__":
    main()
