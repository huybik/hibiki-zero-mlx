#!/usr/bin/env python
"""VieNeu-TTS v3 Turbo voicing both ways, each row in its own speaker's voice, on its engine (vieneu/engine.py)
([tranche-tts](../../docs/agent-memory/tranche-tts.md); speed: [tranche-tts-scaleup](../../docs/agent-memory/tranche-tts-scaleup.md#optimizing-vieneu)):
- English for the real-Vietnamese tranche (`run <source>`): each segments row's text_en from its Vietnamese clip;
- Vietnamese for the EN-VI dataset (`run en_speech`): each en_speech train row's text_vi from its English clip, where
  the Vietnamese the dataset holds (the Mac's MLX voicing, training/en_vi_dataset/tts.py) is broken.

Per row, the loader's processes prepare its reference and its line a shard ahead, the engine voices it at 48 kHz,
and the try is heard by the ASR server (Qwen3-ASR, its language forced; asr.py) and broken when its error is over
BROKEN (the English's WER under Whisper's normalizer, the Vietnamese's CER over `filters.spoken`, as ingest scores
its ASR: `asr.cer`) or its pace is off PACE by 2x. A row is voiced once: a second try costs more TTS time than its
rows give back ([tranche-tts](../../docs/agent-memory/tranche-tts.md#the-check)).

Env /workspace/envs/vieneu (box/envs.sh). From the repo root, with HF_TOKEN_AN (pushes) in the environment and the
ASR server up (`restart.sh asr`):
  /workspace/envs/vieneu/bin/python -m training.tranche.voice run yodas
      every segments shard (each has its text_en and asr_vi) on the Hub without en_audio yet, but those held out
      for evaluation (`repo.SOURCES[...].heldout`); writes <root>/<source>/en_audio/<shard> (id, en_audio as Ogg
      Opus from 24 kHz, decoded at 48 kHz, en_duration_s, en_asr, en_wer, en_tries, en_broken). Rows the cache build
      drops on the Vietnamese side (`filters.vi_gate` on asr_vi), and those whose text_en is missing or has nothing
      speakable, are not voiced: en_tries 0, the other columns null.
  /workspace/envs/vieneu/bin/python -m training.tranche.voice run en_speech
      every en_speech train shard of the tranche repo (the EN-VI dataset's copy) without vi_check yet: each row's
      Vietnamese as the dataset holds it heard first, the broken ones (and any without audio) voiced again; writes
      <root>/en_speech/vi_check/<shard> (id, vi_audio as Ogg Opus from 48 kHz where a new try is kept and null where
      the dataset's stays, then of the row's kept audio vi_duration_s, vi_asr, vi_cer, vi_tries (0: the dataset's,
      heard only) and vi_broken). A row without text_vi is neither heard nor voiced: all null but vi_tries 0. Each
      shard's download is deleted once read.
  Each source's folder is pushed every 30 min and at its end. A rerun skips the shards already written or pushed,
  and first pushes any written but not yet pushed.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.tranche.vieneu.engine import ENROL_PIECE, OUT_RATE, Request, Voicer, log, phonemize, reference

EN_SPEECH = "en_speech"  # the EN-VI dataset's rows in the tranche repo, voiced in Vietnamese; every other source in English
WHISPER_NORMALIZER = ("openai/whisper-large-v3", "06f233f")  # its English normalizer only
# A try is broken over these: the English's WER under Whisper's normalizer (as the TTS bake-off scored), the
# Vietnamese's CER over filters.spoken at the gate's filters.MAX_CER.
BROKEN = {"en": 0.2, "vi": 0.15}
# Seconds of audio per character of the line: the bake-off's median English, within the Mac's Vietnamese medians on
# en_speech (0.060-0.068, 2026-11-28); a try off it by 2x either way is broken.
PACE = 0.0636
# The kept Opus's input rate: the tranche's English at Mimi's 24 kHz, the EN-VI dataset's Vietnamese at its 48 kHz.
KEPT_RATE = {"en": 24000, "vi": 48000}
# The server stalls past a try now and then and recovers within minutes (and drops an idle keep-alive connection
# now and then): 3 tries ended a run three times in two days (2026-11-06, 11-07 17:22 and 23:17 UTC+7), each losing
# the shards in flight.
ASR_TRIES = 10
ASR_PAUSE_S = 10  # a refused connection (the server restarting) fails at once
CHECKERS = 32  # checking processes
# rows a checking process hears at once, each on its own thread: 256 requests in flight, the server's --max-num-seqs.
# The server's time is its decode steps, mostly a read of its weights: one request a process (~30 in flight) held
# the whole run to ~28 rows/s beside the voicer (2026-11-28).
CHECK_ROWS = 8
LOADER_SHARE = 4  # processes preparing the next shard's references and lines: a quarter of the CPU quota
OPUS_AT_ONCE = 4
PUSH_S = 1800
MAX_OPEN = 3  # shards in the voicer at once: those finishing their slowest rows, and the next


# --- run ---------------------------------------------------------------------------------------


@dataclass
class Shard:
    source: str
    name: str  # its file's name: a segments shard's, or an en_speech train shard's
    lang: str  # the language it is voiced in: "en", or "vi" for en_speech
    dst: Path  # <root>/<source>/en_audio/<name>, or <root>/en_speech/vi_check/<name>
    ids: list[str]
    texts: list[str]  # each row's line, in `lang`
    clips: list[np.ndarray] | None  # the voiced rows' references (`reference`), until enrolled
    lines: list[tuple[list[str], list]]  # the voiced rows' chunks and pauses (`phonemize`)
    # id -> the kept try: {"err", "broken", "asr", "audio", "duration", "tries"}; an en_speech row's Vietnamese
    # as the dataset holds it is try 0, its "audio" None
    results: dict
    skipped: set[str]  # rows not voiced (`load_segments`, `load_en_speech`)


class Checker:
    """Hears a try with Qwen3-ASR (`qwen-asr-serve`, its language forced) and scores it against its line: the
    English's WER under Whisper's normalizer (the line unbracketed first), the Vietnamese's CER over `filters.spoken`
    (`asr.cer`), and its pace (BROKEN, PACE). A voiced try is also encoded as the Ogg Opus kept (KEPT_RATE)."""

    def __init__(self, url: str, encoding):
        from transformers import WhisperTokenizer

        from training.tranche.asr import Client

        self.asr = Client(url, CHECK_ROWS, tries=ASR_TRIES, pause_s=ASR_PAUSE_S)  # CHECK_ROWS threads share it
        # PyAV's encoder resamples into libopus's 20 ms frames through an ffmpeg filter graph that
        # holds a thread per host CPU (224 on the box) while it lives: a semaphore shared by the
        # checking processes keeps a few encodes at a time, under the container's thread limit.
        self.encoding = encoding
        from huggingface_hub import snapshot_download

        repo, revision = WHISPER_NORMALIZER
        path = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.txt"])
        self.norm = WhisperTokenizer.from_pretrained(path).normalize

    def error(self, lang: str, text: str, asr: str) -> float:
        import jiwer

        from training.tranche.asr import cer

        if lang == "en":
            ref = self.norm(re.sub(r"[()\[\]{}<>]", " ", text))
            return jiwer.wer(ref, self.norm(asr)) if ref.strip() else 0.0
        return cer(text, asr)

    def check(self, lang: str, text: str, wav: np.ndarray, keep: bool = True) -> dict:
        """A 48 kHz try of `text` in `lang` -> its score, and with `keep` the Ogg Opus kept of it."""
        from scipy.signal import resample_poly

        from training.audio.decode import encode_opus
        from training.tranche.asr import wav_16k

        asr = self.asr.hear(wav_16k(wav, OUT_RATE), lang)[1]
        err = self.error(lang, text, asr)
        duration = len(wav) / OUT_RATE
        pace = duration / max(len(text), 1)
        broken = err > BROKEN[lang] or not 0.5 * PACE <= pace <= 2 * PACE
        audio = None
        if keep:
            rate = KEPT_RATE[lang]
            pcm = wav if rate == OUT_RATE else resample_poly(wav, rate // np.gcd(rate, OUT_RATE), OUT_RATE // np.gcd(rate, OUT_RATE))
            with self.encoding:
                audio = encode_opus(pcm.astype(np.float32), rate)
        return {"err": round(err, 4), "broken": broken, "asr": asr, "duration": round(duration, 3), "audio": audio}


_checker: Checker | None = None
_hearing = None  # the checking process's threads, one a row of a part


def _start_checker(url: str, encoding) -> None:
    from concurrent.futures import ThreadPoolExecutor

    global _checker, _hearing
    _checker = Checker(url, encoding)
    _hearing = ThreadPoolExecutor(CHECK_ROWS)


def _check(rows: list[tuple[str, str, np.ndarray]]) -> list[dict]:
    """Voiced tries (lang, text, 48 kHz audio) -> their scores and Opus."""
    return list(_hearing.map(lambda row: _checker.check(*row), rows))


def _hear_held(rows: list[tuple[str, bytes]]) -> list[dict]:
    """The EN-VI dataset's Vietnamese (text_vi, its Ogg Opus) -> their scores; nothing encoded."""
    from training.audio.decode import decode

    def hear(row):
        pcm, rate = decode(row[1])
        if rate != OUT_RATE:  # libopus decodes at 48 kHz whatever rate it encoded
            raise ValueError(f"Vietnamese audio decoded at {rate} Hz")
        return _checker.check("vi", row[0], pcm, keep=False)

    return list(_hearing.map(hear, rows))


def prepare(key: str, pool, rows: list[tuple[str, bytes, str]]) -> tuple[list, list, set[str]]:
    """The references and lines of rows to voice (id, their clip, their line), in the loader's processes; those
    whose line has nothing speakable (the SDK would say nothing) come back apart, as ids."""
    clips = pool.map(reference, [c for _, c, _ in rows], chunksize=32)  # both maps run at once
    lines = pool.map(phonemize, [t for _, _, t in rows], chunksize=32)
    clips, lines = list(clips), list(lines)
    keep = [k for k, (chunks, _) in enumerate(lines) if chunks]
    mute = {rows[k][0] for k in set(range(len(rows))) - set(keep)}
    if mute:
        log(f"{key}: {len(mute)} rows whose line has nothing speakable, e.g. {min(mute)}")
    return [clips[k] for k in keep], [lines[k] for k in keep], mute


def load_segments(source: str, name: str, root: Path, pool) -> Shard:
    """A tranche segments shard, voiced in English: all its rows but those the cache build drops on the Vietnamese
    side (`filters.vi_gate` on asr_vi), those missing text_en and those with nothing speakable in it."""
    import pyarrow.parquet as pq

    from training.tranche.filters import vi_gate
    from training.tranche.repo import part_file

    get = lambda part: pq.read_table(part_file(root, source, part, name))  # noqa: E731
    seg = get("segments").select(["id", "text", "audio"])
    en = dict(zip(*get("text_en").select(["id", "text_en"]).to_pydict().values()))
    ids, texts = seg.column("id").to_pylist(), seg.column("text").to_pylist()
    no_en = {i for i in ids if not en.get(i)}
    if no_en:
        log(f"{source}/{name}: {len(no_en)} rows without text_en, e.g. {next(x for x in ids if x in no_en)}")
    vi = get("asr_vi").select(["id", "asr_vi", "language", "cer"]).to_pylist()
    if [r["id"] for r in vi] != ids:
        raise SystemExit(f"{source}/{name}: its asr_vi does not hold the segments' rows in their order")
    # The cache build rejects the vi_gate rows as "not voiced" and the no_en rows gracefully.
    skipped = no_en | {i for i, text, r in zip(ids, texts, vi) if vi_gate(r | {"text": text})}
    voiced = [(i, a["bytes"], en[i]) for i, a in zip(ids, seg.column("audio").to_pylist()) if i not in skipped]
    clips, lines, mute = prepare(f"{source}/{name}", pool, voiced)
    return Shard(source, name, "en", root / source / "en_audio" / name, ids, [en.get(i, "") for i in ids], clips,
                 lines, {}, skipped | mute)


def load_en_speech(name: str, root: Path, pool, checks) -> Shard:
    """An en_speech train shard, voiced in Vietnamese: every row's Vietnamese as the dataset holds it heard first (in
    the checking processes), then the rows it is broken on, or missing for, voiced again. Downloaded beside <root>
    (as ingest's corpus shards), not into the HF cache, and deleted once read: the 1,832 shards are ~447 GB."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from training.tranche.repo import REPO

    path = Path(hf_hub_download(REPO, f"{EN_SPEECH}/train/{name}", repo_type="dataset",
                                local_dir=root / EN_SPEECH / ".download"))
    try:
        t = pq.read_table(path, columns=["id", "text_vi", "vi_audio", "en_audio"])
    finally:
        path.unlink()
    ids, texts = t.column("id").to_pylist(), [s or "" for s in t.column("text_vi").to_pylist()]
    held = [a and a["bytes"] for a in t.column("vi_audio").to_pylist()]
    heard = [k for k in range(len(ids)) if texts[k] and held[k]]
    parts = [heard[p: p + CHECK_ROWS] for p in range(0, len(heard), CHECK_ROWS)]
    futures = [checks.submit(_hear_held, [(texts[k], held[k]) for k in part]) for part in parts]
    del held
    results = {}
    for part, future in zip(parts, futures):
        for k, r in zip(part, future.result()):
            results[ids[k]] = r | {"tries": 0}
    voice = [k for k in range(len(ids)) if texts[k] and (ids[k] not in results or results[ids[k]]["broken"])]
    english = t.column("en_audio").take(pa.array(voice, pa.int64())).to_pylist()  # typed: an empty list too
    clips, lines, mute = prepare(f"{EN_SPEECH}/{name}", pool,
                                 [(ids[k], a["bytes"], texts[k]) for k, a in zip(voice, english)])
    log(f"{EN_SPEECH}/{name}: {len(heard)} of {len(ids)} rows heard, {len(voice)} to voice "
        f"({sum(r['broken'] for r in results.values())} broken, {sum(map(bool, texts)) - len(heard)} without audio)")
    return Shard(EN_SPEECH, name, "vi", root / EN_SPEECH / "vi_check" / name, ids, texts, clips, lines, results,
                 (set(ids) - {ids[k] for k in voice}) | mute)


ERROR = {"en": "wer", "vi": "cer"}  # the error column's name


def write_shard(shard: Shard) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    lang = shard.lang
    r = [shard.results.get(i) for i in shard.ids]  # None: neither voiced nor heard
    col = lambda f, kind: pa.array([None if x is None else f(x) for x in r], kind)  # noqa: E731
    table = pa.table({
        "id": shard.ids,
        f"{lang}_audio": pa.array([None if x is None or x["audio"] is None else
                                   {"bytes": x["audio"], "path": f"{i.replace('/', '_')}.opus"}
                                   for i, x in zip(shard.ids, r)],
                                  pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        f"{lang}_duration_s": col(lambda x: x["duration"], pa.float64()),
        f"{lang}_asr": col(lambda x: x["asr"], pa.string()),
        f"{lang}_{ERROR[lang]}": col(lambda x: x["err"], pa.float32()),
        f"{lang}_tries": pa.array([0 if x is None else x["tries"] for x in r], pa.int8()),
        f"{lang}_broken": col(lambda x: x["broken"], pa.bool_()),
    })
    shard.dst.parent.mkdir(parents=True, exist_ok=True)
    part = shard.dst.with_name(shard.dst.name + ".part")
    pq.write_table(table, part)
    os.replace(part, shard.dst)


def out_part(source: str) -> str:
    """The part a source's shards are voiced into."""
    return "vi_check" if source == EN_SPEECH else "en_audio"


def shards_to_voice(args: argparse.Namespace, files: list[str]) -> list[tuple[str, str]]:
    """(source, name) of every shard on the Hub to voice that neither the Hub nor <root> has the voicing of: for the
    tranche's sources, their segments shards but those held out for evaluation, once translated and heard (a later
    run takes the others); for en_speech, its train shards."""
    import pyarrow.parquet as pq

    from training.tranche import repo

    todo = []
    for source in args.sources:
        part = out_part(source)
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
        done |= {p.name for p in (args.root / source / part).glob("*.parquet")}
        if source == EN_SPEECH:
            todo += [(source, n) for n in sorted(f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/train/"))
                     if n not in done]
            continue
        # A shard waits for its translation, pushed or still only here (translate runs beside voice),
        # and while a row of it has none (translate's next pass fills it).
        translated = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/text_en/")}
        for p in (args.root / source / "text_en").glob("*.parquet"):
            if pq.read_table(p, columns=["text_en"]).column("text_en").null_count:
                translated.discard(p.name)
            else:
                translated.add(p.name)
        # and for its Vietnamese ASR (asr_vi: YODAS's from ingest), whose gates skip rows.
        heard = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/asr_vi/")}
        heard |= {p.name for p in (args.root / source / "asr_vi").glob("*.parquet")}
        waiting = 0
        for f in sorted(f for f in files if f.startswith(f"{source}/segments/")):
            name = f.rsplit("/", 1)[1]
            if name in repo.SOURCES[source].heldout or name not in translated or name in done:
                continue  # held out: an evaluation set's shard, never trained on, so never voiced
            if name in heard:
                todo.append((source, name))
            else:
                waiting += 1
        if waiting:
            log(f"{source}: {waiting} translated shards wait for their asr_vi")
    return todo[: args.limit]


def run(args: argparse.Namespace) -> None:
    """Every shard of each source in turn, the next one loading while the voicer voices; the voicer is kept full
    of new rows. A shard is written once every voiced row has its try checked, and each source's written shards
    are pushed every PUSH_S and at its end."""
    import queue
    import threading
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from huggingface_hub import HfApi

    from training.paths import cpu_budget
    from training.tranche import repo

    repo.an_token()
    files = HfApi().list_repo_files(repo.REPO, repo_type="dataset")
    for source in args.sources:
        part = out_part(source)
        hub = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
        if {p.name for p in (args.root / source / part).glob("*.parquet")} - hub:  # written by an earlier run
            repo.push_part(args.root, source, part)                                # that stopped before pushing them
    todo = shards_to_voice(args, files)
    log(f"{len(todo)} shards to voice: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
    if not todo:
        return

    # Loading and checking run in processes: as threads, their Python (PyAV's frame loop, the trim,
    # the phonemizer, the normalizer, scipy, Opus) would hold the GIL from the frame loop and starve
    # the GPU. Their thread pools stay at one thread each.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    spawn = get_context("spawn")
    pool = ProcessPoolExecutor(max(1, int(cpu_budget() // LOADER_SHARE)), mp_context=spawn)
    # en_speech's loading hears its rows through these too, a shard ahead of the voicer.
    checks = ProcessPoolExecutor(CHECKERS, mp_context=spawn, initializer=_start_checker,
                                 initargs=(args.asr, spawn.BoundedSemaphore(OPUS_AT_ONCE)))
    # A shard's references are ~17 GB (8 s at 44.1 kHz in float32 a row), freed as they enrol: the next shard loads
    # once the one before it starts enrolling, so at most two are held. Its ~80 s load hides behind a shard's voicing.
    loaded: queue.Queue = queue.Queue()
    ahead = threading.Semaphore(1)

    def loader():
        try:
            for source, name in todo:
                ahead.acquire()
                began_load = time.monotonic()
                shard = (load_en_speech(name, args.root, pool, checks) if source == EN_SPEECH
                         else load_segments(source, name, args.root, pool))
                log(f"{source}/{name}: loaded {len(shard.clips)} references and lines in "
                    f"{time.monotonic() - began_load:.0f} s")
                loaded.put(shard)
        except BaseException as exc:  # raised again in the main thread
            loaded.put(exc)
        loaded.put(None)

    threading.Thread(target=loader, name="shard-loader", daemon=True).start()
    voicer = Voicer()
    checking: dict = {}  # future -> its part's requests
    inflight = set()
    fresh: Iterator[Request] = iter(())
    shards: dict[str, Shard] = {}  # by source/name, until written
    owner: dict[str, Shard] = {}  # row id -> its shard
    remaining: dict[str, int] = {}  # shard -> rows without a checked try
    loading_done = False
    left = {s: sum(t[0] == s for t in todo) for s in args.sources}  # shards not yet written
    pushes = {s: repo.Pushes(args.root, out_part(s)) for s in args.sources}  # uploads run beside the voicing
    pushed = {s: time.monotonic() for s in args.sources}
    began, voiced_s, attempts = time.monotonic(), 0.0, 0

    def push(source: str) -> None:
        pushed[source] = time.monotonic()
        pushes[source].push(source)

    def requests(key: str, shard: Shard, rows: list) -> Iterator[Request]:
        """The shard's rows, enrolled ENROL_PIECE at a time as the voicer takes them, shortest references first (a
        piece's references share few lengths): the frames and the ASR keep running between pieces."""
        if len(rows) != len(shard.clips) or len(rows) != len(shard.lines):
            raise SystemExit(f"{key}: {len(rows)} rows to voice, {len(shard.clips)} references, {len(shard.lines)} lines")
        order = sorted(range(len(rows)), key=lambda k: len(shard.clips[k]))
        spent = 0.0
        for p in range(0, len(order), ENROL_PIECE):
            piece = order[p: p + ENROL_PIECE]
            began_enrol = time.monotonic()
            refs = voicer.enrol([shard.clips[k] for k in piece])
            spent += time.monotonic() - began_enrol
            for k in piece:
                shard.clips[k] = None
            for k, (spk, codes) in zip(piece, refs):
                (i, line), (chunks, gaps) = rows[k], shard.lines[k]
                yield Request(i, line, chunks, gaps, spk, codes)
        log(f"{key}: enrolled {len(rows)} references in {spent:.0f} s")

    def next_shard() -> bool:
        """Opens the next loaded shard; waits for one only while the voicer is idle (en_speech's loads hear a whole
        shard, longer than voicing the last one's broken rows)."""
        nonlocal fresh, loading_done
        try:
            shard = loaded.get(block=not voicer.active)
        except queue.Empty:
            return False
        if isinstance(shard, BaseException):
            raise shard
        if shard is None:
            loading_done = True
            return False
        ahead.release()  # the next shard loads while this one's references enrol, piece by piece
        key = f"{shard.source}/{shard.name}"
        rows = [(i, line) for i, line in zip(shard.ids, shard.texts) if i not in shard.skipped]
        shards[key], remaining[key] = shard, len(rows)
        for i, _ in rows:
            owner[i] = shard
        fresh = requests(key, shard, rows)
        if not rows:
            finish(shard)
        return True

    def finish(shard: Shard) -> None:
        """Writes a shard whose voiced rows all have a checked try, and pushes as due."""
        key = f"{shard.source}/{shard.name}"
        write_shard(shard)
        voiced = [shard.results[i] for i in shard.ids if i not in shard.skipped]
        log(f"{key}: {len(shard.ids)} rows, {len(voiced)} voiced, {len(shard.skipped)} not; "
            f"{sum(x['broken'] for x in voiced) / max(len(voiced), 1):.1%} of them still broken; overall "
            f"{attempts / (time.monotonic() - began):.1f} rows/s, RTF {(time.monotonic() - began) / max(voiced_s, 1):.5f} "
            f"over {voiced_s / 3600:.1f} h voiced ({attempts} attempts)")
        for i in shard.ids:
            owner.pop(i, None)
        del shards[key], remaining[key]
        left[shard.source] -= 1
        if left[shard.source] == 0 or time.monotonic() - pushed[shard.source] > PUSH_S:
            push(shard.source)

    def settle(req: Request, result: dict) -> None:
        nonlocal attempts, voiced_s
        attempts += 1
        voiced_s += result["duration"]
        shard = owner[req.key]
        # An en_speech row's Vietnamese as the dataset holds it (try 0) stays unless this try beats it.
        held = shard.results.get(req.key)
        result["tries"] = 1
        if held is None or result["err"] < held["err"] or (held["broken"] and not result["broken"]):
            shard.results[req.key] = result
        else:
            held["tries"] = 1
        key = f"{shard.source}/{shard.name}"
        remaining[key] -= 1
        if remaining[key] == 0:
            finish(shard)

    try:
        while True:
            while voicer.room:
                req = next(fresh, None)
                if req is None:
                    if loading_done or len(shards) >= MAX_OPEN or not next_shard():
                        break  # a few shards open: those finishing and the next
                    continue
                voicer.submit(req)
            out = voicer.step()
            for p in range(0, len(out), CHECK_ROWS):
                part = out[p: p + CHECK_ROWS]
                fut = checks.submit(_check, [(owner[req.key].lang, req.text, wav) for req, wav in part])
                checking[fut] = [req for req, _ in part]
                inflight.add(fut)
            for fut in [f for f in inflight if f.done()]:
                inflight.discard(fut)
                for req, result in zip(checking.pop(fut), fut.result()):
                    settle(req, result)
            for p in pushes.values():
                p.check()  # a failed upload stops the run; a rerun pushes what is written
            if loading_done and not voicer.active and not inflight:
                break
            if not voicer.active and not out:
                time.sleep(0.05)  # waiting on checks: the open shards' finish lets the next one open
    finally:  # after a crash, queued checks are dropped, not run, so the GPU frees promptly
        checks.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False, cancel_futures=True)
    for p in pushes.values():
        p.wait()
    log(f"done: {voiced_s / 3600:.1f} h voiced in {(time.monotonic() - began) / 3600:.2f} h")


def main() -> None:
    from training.tranche import asr, repo

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("run")
    p.add_argument("sources", nargs="+", choices=sorted([*repo.SOURCES, EN_SPEECH]))
    p.add_argument("--root", type=Path, default=repo.ROOT)
    p.add_argument("--asr", default=asr.URL)
    p.add_argument("--limit", type=int, help="voice only this many shards (a staged start)")
    # As the bench ran: the prefill, decode and enrolment batches of many shapes fragment PyTorch's cached blocks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
