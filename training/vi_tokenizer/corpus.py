"""Corpus passes for the Vietnamese tokenizer swap, over the EN-VI speech dataset's text
([vi-tokenizer](../../docs/agent-memory/vi-tokenizer.md)).

  texts:  the dataset's text_en, text_vi and vi_duration_s, read from the Hub by range requests
          (no audio), one parquet per shard under <work>/texts/{train,validation}/
  count:  English piece use, Vietnamese word-form counts and Vietnamese piece use -> counts.pkl
  verify: for each candidate model, the rows whose text_en ids differ from the base model's (and
          the new pieces in them, merged into --exclude), plus Vietnamese tokens per word, byte
          share, unk, round trip and tokens/s of speech of every text_vi row -> a JSON report

Runs on the box (CUDA-free, many cores): --workers defaults to every core. --sample K reads K
evenly spaced train shards plus every validation shard (a smoke test).

  python -m training.vi_tokenizer.corpus texts  [--work W]
  python -m training.vi_tokenizer.corpus count  [--work W] [--sample K]
  python -m training.vi_tokenizer.corpus verify [--work W] [--sample K] --out report.json MODEL...

`python -m training.vi_tokenizer.build run` chains these with the model builds.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from training.paths import DEFAULT_TOKENIZER

DATASET = ("anquachdev/common-vi-en", "2ce8777")  # repo, revision (checked 2026-11-05)
COLUMNS = ["text_en", "text_vi", "vi_duration_s"]
WORK = Path("/workspace/vi_tokenizer")
SPLITS = ("train", "validation")

_cfg: dict = {}
_sps: list = []


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --- texts -------------------------------------------------------------------------------------


def texts(work: Path, threads: int = 32) -> None:
    """threads: concurrent shard reads, each waiting on the Hub's latency."""
    from huggingface_hub import HfApi, HfFileSystem

    repo, rev = DATASET
    sha = HfApi().dataset_info(repo, revision=rev).sha
    names = [f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=sha)
             if f.endswith(".parquet") and f.split("/")[1] in SPLITS]
    fs = HfFileSystem()

    def fetch(name: str) -> None:
        dst = work / "texts" / name.split("/", 1)[1]
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Exact range reads of just these columns' chunks (~50 KB a row group, beside ~12 MB of
        # audio); read_table would pre-buffer the audio too.
        with fs.open(f"datasets/{repo}@{sha}/{name}", "rb", cache_type="none") as fh:
            table = pq.ParquetFile(fh).read(columns=COLUMNS)
        part = dst.with_name(dst.name + ".part")
        pq.write_table(table, part)
        os.replace(part, dst)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        for n, _ in enumerate(pool.map(fetch, names), 1):
            if n % 200 == 0:
                log(f"{n}/{len(names)} shards {time.time() - t0:.0f}s")
    log(f"{len(names)} shards' text in {work / 'texts'}")


# --- count and verify --------------------------------------------------------------------------


def shards(work: Path, sample: int | None) -> list[Path]:
    train = sorted((work / "texts/train").glob("*.parquet"))
    val = sorted((work / "texts/validation").glob("*.parquet"))
    if not train or not val:
        raise SystemExit(f"no train or validation texts under {work}/texts: run `texts` first")
    if sample:
        step = len(train) / sample
        train = [train[int(i * step)] for i in range(sample)]
    return train + val


def split_word(word: str) -> tuple[str, str]:
    """-> (leading punctuation, the alphanumeric core without trailing punctuation)."""
    i, j = 0, len(word)
    while i < j and not word[i].isalnum():
        i += 1
    while j > i and not word[j - 1].isalnum():
        j -= 1
    return word[:i], word[i:j]


def init(cfg: dict) -> None:
    global _cfg, _sps
    _cfg = cfg
    _sps = [spm.SentencePieceProcessor(model_file=str(m)) for m in [cfg["base_model"]] + cfg.get("models", [])]


def read(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    t = pq.read_table(path, columns=COLUMNS)
    return t.column("text_en").to_pylist(), t.column("text_vi").to_pylist(), t.column("vi_duration_s").to_numpy()


def count_shard(path: Path) -> dict:
    sp = _sps[0]
    en, vi, _ = read(path)
    en_ids = sp.encode(en, num_threads=1)
    vi_ids = sp.encode(vi, num_threads=1)
    en_chars = collections.Counter(c for s in en for c in s if ord(c) > 127)
    forms = collections.Counter()  # (bare, core): bare=1 when punctuation precedes the word core
    for s in vi:
        for w in s.split():
            lead, core = split_word(w)
            if core and core.isalpha():
                forms[(1 if lead else 0, core)] += 1
    return {
        "split": path.parent.name,
        "rows": len(en),
        "en_counts": np.bincount(np.fromiter((i for r in en_ids for i in r), np.int64), minlength=sp.vocab_size()),
        "vi_counts": np.bincount(np.fromiter((i for r in vi_ids for i in r), np.int64), minlength=sp.vocab_size()),
        "en_chars": en_chars,
        "forms": forms,
    }


def verify_shard(path: Path) -> dict:
    base = _sps[0]
    en, vi, dur = read(path)
    base_en = base.encode(en, num_threads=1)
    out = {"split": path.parent.name, "vi_dur": dur,
           "vi_words": np.array([len(s.split()) for s in vi], np.int32), "models": []}
    for k, sp in enumerate(_sps):
        r = {}
        if k:
            new_ids = {i for i in range(sp.vocab_size())
                       if i >= base.vocab_size() or sp.id_to_piece(i) != base.id_to_piece(i)}
            new_en = sp.encode(en, num_threads=1)
            diffs = [j for j in range(len(en)) if new_en[j] != base_en[j]]
            culprits, examples = collections.Counter(), {}
            for j in diffs:
                hit = {sp.id_to_piece(i) for i in new_en[j] if i in new_ids}
                culprits.update(hit)
                for piece in hit:
                    examples.setdefault(piece, en[j][:300])
            r = {"en_rows": len(en), "en_diff_rows": len(diffs), "en_culprits": culprits,
                 "en_diff_without_new_piece": sum(not any(i in new_ids for i in new_en[j]) for j in diffs),
                 "en_culprit_examples": examples}
        ids = sp.encode(vi, num_threads=1)
        r["vi_tokens"] = np.array([len(x) for x in ids], np.int32)
        r["vi_bytes"] = np.array([sum(sp.is_byte(i) for i in x) for x in ids], np.int32)
        r["vi_unk"] = sum(i == sp.unk_id() for x in ids for i in x)
        r["roundtrip_fail"] = sum(sp.decode(x) != s for x, s in zip(ids, vi))
        words = [w for s in vi for w in s.split() if w.isalpha()]
        r["word_pieces"] = collections.Counter(len(x) for x in sp.encode(words, num_threads=1))
        out["models"].append(r)
    return out


def run_pool(fn, files: list[Path], cfg: dict, workers: int):
    with Pool(workers, initializer=init, initargs=(cfg,)) as pool:
        yield from pool.imap_unordered(fn, files, chunksize=1)


def count(work: Path, base_model: Path, workers: int, sample: int | None) -> Path:
    files = shards(work, sample)
    t0 = time.time()
    agg = {s: {"en_counts": 0, "vi_counts": 0, "en_chars": collections.Counter(), "forms": collections.Counter(),
               "rows": 0, "shards": 0} for s in SPLITS}
    for n, r in enumerate(run_pool(count_shard, files, {"base_model": base_model}, workers), 1):
        a = agg[r["split"]]
        a["en_counts"] = a["en_counts"] + r["en_counts"]
        a["vi_counts"] = a["vi_counts"] + r["vi_counts"]
        a["en_chars"].update(r["en_chars"])
        a["forms"].update(r["forms"])
        a["rows"] += r["rows"]
        a["shards"] += 1
        if n % 200 == 0:
            log(f"count: {n}/{len(files)} shards {time.time() - t0:.0f}s")
    out = work / "counts.pkl"
    with out.open("wb") as f:
        pickle.dump(agg, f)
    log(f"counted {len(files)} shards, rows {({s: a['rows'] for s, a in agg.items()})} in {time.time() - t0:.0f}s")
    return out


def summarize(per_model: list[list[dict]], dur: np.ndarray, words: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for k, parts in enumerate(per_model):
        tok = np.concatenate([p["vi_tokens"] for p in parts])
        byt = np.concatenate([p["vi_bytes"] for p in parts])
        tps, fps = tok / dur, (tok + words) / dur
        hist = collections.Counter()
        for p in parts:
            hist.update(p["word_pieces"])
        row = {
            "model": names[k], "rows": int(len(tok)),
            "tokens_per_word": float(tok.sum() / words.sum()), "byte_share": float(byt.sum() / tok.sum()),
            "unk": int(sum(p["vi_unk"] for p in parts)), "roundtrip_fail": int(sum(p["roundtrip_fail"] for p in parts)),
            "one_piece_word_share": hist[1] / sum(hist.values()),
            "tps_mean": float(tps.mean()), "tps_p90": float(np.percentile(tps, 90)),
            "tps_p99": float(np.percentile(tps, 99)), "tps_max": float(tps.max()),
            "tokens_plus_word_gaps_per_s_mean": float(fps.mean()),
            "tokens_plus_word_gaps_p90": float(np.percentile(fps, 90)),
            "rows_over_12_5_tps": float((tps > 12.5).mean()),
        }
        if k:
            cul, examples = collections.Counter(), {}
            for p in parts:
                cul.update(p["en_culprits"])
                for piece, text in p["en_culprit_examples"].items():
                    examples.setdefault(piece, text)
            row.update({"en_rows": int(sum(p["en_rows"] for p in parts)),
                        "en_diff_rows": int(sum(p["en_diff_rows"] for p in parts)),
                        "en_diff_without_new_piece": int(sum(p["en_diff_without_new_piece"] for p in parts)),
                        "en_culprits": {piece: {"rows": n, "example": examples[piece]} for piece, n in cul.most_common()}})
        rows.append(row)
    return rows


def verify(work: Path, base_model: Path, models: list[Path], out: Path, workers: int,
           sample: int | None) -> dict:
    """Report on every model; merge the new pieces that changed an English row into
    <work>/excluded.json. -> the report."""
    files = shards(work, sample)
    names = [m.name for m in [base_model] + models]
    t0 = time.time()
    res = {s: {"dur": [], "words": [], "models": [[] for _ in names]} for s in SPLITS}
    cfg = {"base_model": base_model, "models": models}
    for n, r in enumerate(run_pool(verify_shard, files, cfg, workers), 1):
        a = res[r["split"]]
        a["dur"].append(r["vi_dur"])
        a["words"].append(r["vi_words"])
        for slot, m in zip(a["models"], r["models"]):
            slot.append(m)
        if n % 200 == 0:
            log(f"verify: {n}/{len(files)} shards {time.time() - t0:.0f}s")
    report = {"shards": len(files), "sample": sample}
    for split, a in res.items():
        report[split] = summarize(a["models"], np.concatenate(a["dur"]), np.concatenate(a["words"]), names)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    culprits = {p for split in SPLITS for row in report[split][1:] for p in row["en_culprits"]}
    excluded = work / "excluded.json"
    old = set(json.loads(excluded.read_text())) if excluded.exists() else set()
    excluded.write_text(json.dumps(sorted(old | culprits), ensure_ascii=False))
    for split in SPLITS:
        for row in report[split][1:]:
            log(f"{split} {row['model']}: {row['en_diff_rows']} of {row['en_rows']} English rows differ; "
                f"Vietnamese {row['tokens_per_word']:.3f} tokens/word")
    log(f"{len(culprits)} new pieces changed English rows; verified in {time.time() - t0:.0f}s")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["texts", "count", "verify"])
    p.add_argument("models", nargs="*", type=Path)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--base-model", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--out", type=Path, help="verify: the report")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="count, verify: processes")
    p.add_argument("--threads", type=int, default=32, help="texts: concurrent shard reads")
    p.add_argument("--sample", type=int, help="evenly spaced train shards to read (default: all)")
    args = p.parse_args()
    if args.step == "verify" and (not args.models or not args.out):
        p.error("verify needs --out and at least one model")
    if args.step == "texts":
        texts(args.work, args.threads)
    elif args.step == "count":
        count(args.work, args.base_model, args.workers, args.sample)
    else:
        verify(args.work, args.base_model, args.models, args.out, args.workers, args.sample)


if __name__ == "__main__":
    main()
