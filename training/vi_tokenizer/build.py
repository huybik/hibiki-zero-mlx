"""The Vietnamese tokenizer swap: N NORMAL pieces English never uses become Vietnamese pieces, so
`text_card` stays 48,000 ([vi-tokenizer](../../docs/agent-memory/vi-tokenizer.md)).

`run [N...]` builds on the box, N slots per model (a multiple of 1000, default 6000):
  1. the texts and counts (`corpus`);
  2. a superset: the base model with the new letters and the top 16k word forms appended (vocab
     over 48k), whose English check finds every new piece that would change an English row;
  3. the 48k-vocab models, each with N never-used slots replaced and its embedding init map
     (<model>_init.json), checked over every row; each new piece that changed an English row is
     excluded and the models rebuilt until none does (5 rounds at most). They are built in
     <work>/candidates/ and move up into <work> only once they pass.

Each run starts from no exclusions. Its files land in --work (default /workspace/vi_tokenizer),
or <work>/sample<K>/ with --sample K: excluded.json (the new pieces that changed an English row),
the superset, the models, their init maps and the verify reports. The texts stay in
<work>/texts/<revision>/ for later runs. From the repo root:

  python -m training.vi_tokenizer.build run
  python -m training.vi_tokenizer.build run 4000 6000 8000 --sample 20
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from training.paths import DEFAULT_TOKENIZER, cpu_budget
from training.vi_tokenizer import corpus

WORDS = Path(__file__).with_name("english_words.txt.gz")  # macOS /usr/share/dict/words (Webster's 2nd)
WORK = Path("/workspace/vi_tokenizer")
SUPERSET = 16000
ROUNDS = 5

T = pb.ModelProto.SentencePiece
VI_LETTERS = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ"
MARGIN = 0.1  # a new piece beats the best other split of its own surface by this much
MAX_PIECE = 16


def load_proto(path: Path) -> pb.ModelProto:
    m = pb.ModelProto()
    m.ParseFromString(path.read_bytes())
    return m


def processor(m: pb.ModelProto, dummy_prefix: bool = True) -> spm.SentencePieceProcessor:
    m2 = pb.ModelProto()
    m2.CopyFrom(m)
    m2.normalizer_spec.add_dummy_prefix = dummy_prefix
    return spm.SentencePieceProcessor(model_proto=m2.SerializeToString())


def score_range(base: pb.ModelProto) -> tuple[float, float]:
    s = [p.score for p in base.pieces if p.type == T.NORMAL]
    return min(s), max(s)


def ranked_slots(base: pb.ModelProto, counts: dict) -> list[int]:
    """NORMAL pieces English never uses, safest first."""
    en = counts["train"]["en_counts"] + counts["validation"]["en_counts"]
    vi = counts["train"]["vi_counts"] + counts["validation"]["vi_counts"]
    lo, _ = score_range(base)
    words = set(gzip.decompress(WORDS.read_bytes()).decode().lower().split())
    keyed = []
    for i, p in enumerate(base.pieces):
        if p.type != T.NORMAL or en[i] or p.score == lo:  # the min stays: it sets the unk penalty
            continue
        text = p.piece.replace("▁", "")
        if len(text) <= 1:
            tier = 3  # single characters: removing one sends unseen text to bytes
        elif any(ord(c) > 127 for c in text):
            tier = 0  # accented foreign words and subwords
        elif text.lower() not in words:
            tier = 1
        else:
            tier = 2  # English dictionary words
        keyed.append((bool(vi[i]), tier, p.score, i))
    keyed.sort()
    return [k[3] for k in keyed]


def chars_to_add(base_sp: spm.SentencePieceProcessor, counts: dict, excluded: set) -> list[str]:
    """Vietnamese letters the vocab lacks and no English row contains."""
    en_chars = counts["train"]["en_chars"] + counts["validation"]["en_chars"]
    return [c for c in VI_LETTERS + VI_LETTERS.upper()
            if base_sp.piece_to_id(c) == base_sp.unk_id() and not en_chars[c] and c not in excluded]


def ranked_forms(base: pb.ModelProto, chars: list[str], counts: dict, excluded: set) -> list[tuple[str, int, int]]:
    """(surface, count, tokens saved) of train text_vi word forms, most saved first.

    A form is an alphabetic word core as it occurs: `▁core` after a space, bare `core` after
    leading punctuation, in its own case. Savings are counted against base + the new letters."""
    m0 = pb.ModelProto()
    m0.CopyFrom(base)
    lo, _ = score_range(base)
    for c in chars:
        m0.pieces.add(piece=c, score=lo + 0.01, type=T.NORMAL)
    forms = counts["train"]["forms"]
    out = []
    for (b, core), ids in encode_forms(m0, forms).items():
        surface = core if b else "▁" + core
        if len(ids) > 1 and surface not in excluded and len(core) <= MAX_PIECE:
            out.append((surface, forms[(b, core)], forms[(b, core)] * (len(ids) - 1)))
    out.sort(key=lambda x: (-x[2], x[0]))
    return out


def best_split(surface: str, scores: dict, unk: float) -> float:
    """Best unigram path score of `surface` over `scores`, excluding the one-piece path."""
    n = len(surface)
    best = [-math.inf] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if best[i] == -math.inf:
            continue
        single = False
        for j in range(i + 1, min(n, i + MAX_PIECE) + 1):
            if i == 0 and j == n:
                continue
            s = scores.get(surface[i:j])
            if s is not None:
                single |= j == i + 1
                best[j] = max(best[j], best[i] + s)
        if not single:
            best[i + 1] = max(best[i + 1], best[i] + unk)  # an unknown character (bytes)
    return best[n]


def assign_scores(vocab: dict, new: list[str], lo: float, hi: float) -> dict:
    """Shortest first, each new piece scores MARGIN above the best other split of its surface,
    within the old NORMAL range so min (unk penalty) and max (user symbols) do not move."""
    scores = dict(vocab)
    out = {}
    for piece in sorted(new, key=len):
        alt = best_split(piece, scores, lo - 10.0)
        out[piece] = min(hi - 0.01, max(lo + 0.01, alt + MARGIN))
        scores[piece] = out[piece]
    return out


def plan(base_path: Path, n: int | None, k: int | None, counts: dict, excluded: set):
    base = load_proto(base_path)
    base_sp = processor(base)
    lo, hi = score_range(base)
    chars = chars_to_add(base_sp, counts, excluded)
    forms = ranked_forms(base, chars, counts, excluded)
    take = (n - len(chars)) if n is not None else k
    if take > len(forms):
        raise SystemExit("ran out of candidate forms")
    chosen = forms[:take]
    new = chars + [f[0] for f in chosen]
    slots = ranked_slots(base, counts)[:n] if n is not None else []
    if n is not None and len(slots) < n:
        raise SystemExit("not enough never-used slots")
    removed = {base.pieces[i].piece for i in slots}
    vocab = {p.piece: p.score for p in base.pieces if p.type == T.NORMAL and p.piece not in removed}
    scores = assign_scores(vocab, new, lo, hi)
    return base, lo, hi, chars, chosen, new, slots, scores


def write_superset(base_path: Path, work: Path, k: int, counts: dict, excluded: set) -> Path:
    base, lo, hi, chars, chosen, new, _, scores = plan(base_path, None, k, counts, excluded)
    m = pb.ModelProto()
    m.CopyFrom(base)
    for piece in new:
        m.pieces.add(piece=piece, score=scores[piece], type=T.NORMAL)
    path = work / f"superset_{k}.model"
    path.write_bytes(m.SerializeToString())
    print(json.dumps({"path": str(path), "chars": len(chars), "forms": len(chosen), "last_gain": chosen[-1][2]}))
    return path


def kind(piece: str) -> str:
    if len(piece) == 1:
        return "letter"
    if not piece.startswith("▁"):
        return "bare"
    t = piece[1:]
    return "upper" if t.isupper() else "capitalized" if t[0].isupper() else "lower"


def encode_surface(sp: spm.SentencePieceProcessor, sp_bare: spm.SentencePieceProcessor, surface: str) -> list[int]:
    return sp.encode(surface[1:]) if surface.startswith("▁") else sp_bare.encode(surface)


def encode_forms(m: pb.ModelProto, forms: dict) -> dict:
    """{(bare, core): ids}: `▁core` after a space, bare `core` after punctuation."""
    keys = list(forms)
    prefixed = iter(processor(m).encode([core for b, core in keys if not b]))
    bare = iter(processor(m, dummy_prefix=False).encode([core for b, core in keys if b]))
    return {k: next(bare) if k[0] else next(prefixed) for k in keys}


def shadowing(base: pb.ModelProto, model: pb.ModelProto, excluded: set) -> set:
    """New pieces that re-split an excluded surface (one known to change English), e.g. bare
    `và` taking over ` và` once `▁và` is excluded: those English rows would change too."""
    b, b_bare = processor(base), processor(base, dummy_prefix=False)
    m, m_bare = processor(model), processor(model, dummy_prefix=False)
    new_ids = {i for i in range(len(model.pieces)) if model.pieces[i].piece != base.pieces[i].piece}
    out = set()
    for q in excluded:
        ids = encode_surface(m, m_bare, q)
        if ids != encode_surface(b, b_bare, q):
            hit = {model.pieces[i].piece for i in ids if i in new_ids}
            if not hit:
                raise RuntimeError(f"{q!r} re-split without a new piece")
            out |= hit
    return out


def write_model(base_path: Path, work: Path, n: int, counts: dict, excluded: set) -> dict:
    shadowed: set = set()
    while True:
        base, lo, hi, chars, chosen, new, slots, scores = plan(base_path, n, None, counts, excluded | shadowed)
        model = pb.ModelProto()
        model.CopyFrom(base)
        for slot, piece in zip(slots, new, strict=True):
            model.pieces[slot].piece = piece
            model.pieces[slot].score = scores[piece]
            model.pieces[slot].type = T.NORMAL  # already NORMAL, but set: the published file has it
        more = shadowing(base, model, excluded)
        if not more:
            break
        shadowed |= more
    assert len(model.pieces) == 48000 and len({p.piece for p in model.pieces}) == 48000
    assert score_range(model) == (lo, hi), "unk penalty and user-symbol scores must not move"
    path = work / model_name(base_path, n)
    path.write_bytes(model.SerializeToString())

    # Each new piece is one token on its own surface.
    sp, sp_bare = processor(model), processor(model, dummy_prefix=False)
    not_whole = [p for p, slot in zip(new, slots) if encode_surface(sp, sp_bare, p) != [slot]]

    # Embedding init recipe: the ORIGINAL tokenizer's split of each new piece.
    orig, orig_bare = processor(base), processor(base, dummy_prefix=False)
    slot_set = set(slots)
    recipe, kinds, init_uses_slot, init_uses_byte = [], {}, 0, 0
    for slot, piece in zip(slots, new):
        ids = encode_surface(orig, orig_bare, piece)
        recipe.append({"id": slot, "piece": piece, "score": round(scores[piece], 4),
                       "replaces": base.pieces[slot].piece, "init_from_ids": ids,
                       "init_from_pieces": [base.pieces[i].piece for i in ids]})
        kinds[kind(piece)] = kinds.get(kind(piece), 0) + 1
        init_uses_slot += any(i in slot_set for i in ids)
        init_uses_byte += any(orig.is_byte(i) for i in ids)
    path.with_name(path.stem + "_init.json").write_text(json.dumps({
        "base_model": base_path.name,
        "model": path.name,
        "how": "Before overwriting row `id` of text_emb, depformer_text_emb and text_linear, set it "
               "to the mean of the ORIGINAL rows `init_from_ids`, read from a copy of the old "
               "tables: some of those ids are byte pieces or are themselves replaced slots.",
        "pieces": recipe,
    }, ensure_ascii=False, indent=1))
    forms = counts["train"]["forms"]
    one_piece = sum(forms[k] for k, ids in encode_forms(model, forms).items() if len(ids) == 1)
    return {"n": n, "path": str(path), "kinds": kinds, "shadowed": sorted(shadowed), "not_whole": not_whole[:20], "n_not_whole": len(not_whole),
            "train_words_one_piece": one_piece / sum(forms.values()),
            "train_words_in_new_pieces": sum(f[1] for f in chosen) / sum(forms.values()),
            "last_count": chosen[-1][1], "last_gain": chosen[-1][2],
            "init_uses_replaced_slot": init_uses_slot, "init_uses_byte": init_uses_byte,
            "score_range_new": [min(scores.values()), max(scores.values())]}


def model_name(base_path: Path, n: int) -> str:
    return f"{base_path.stem}_vi{n // 1000}k.model"


def english_differs(report: dict) -> bool:
    return any(row["en_diff_rows"] for split in corpus.SPLITS for row in report[split][1:])


def check(files: list[Path], args: argparse.Namespace, models: list[Path], out: Path) -> tuple[dict, set]:
    """-> the verify report, also written to `out`, and the new pieces that changed English rows."""
    report = corpus.verify(files, args.base_model, models, args.workers)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    culprits = {p for split in corpus.SPLITS for row in report[split][1:] for p in row["en_culprits"]}
    corpus.log(f"{len(culprits)} new pieces changed English rows")
    return report, culprits


def run(args: argparse.Namespace) -> None:
    base, sizes = args.base_model, args.sizes or [6000]
    work = args.work / f"sample{args.sample}" if args.sample else args.work
    candidates = work / "candidates"  # unchecked models; one that passes moves up into work
    candidates.mkdir(parents=True, exist_ok=True)
    for old in work.glob("verify_models_*.json"):  # an earlier run's rounds
        old.unlink()
    files = corpus.texts(args.work, args.sample)
    counts = corpus.count(files, base, args.workers)
    superset = write_superset(base, work, SUPERSET, counts, set())
    _, excluded = check(files, args, [superset], work / "verify_superset.json")
    for n in range(1, ROUNDS + 1):
        (work / "excluded.json").write_text(json.dumps(sorted(excluded), ensure_ascii=False))
        print("excluded new pieces:", len(excluded), flush=True)
        built = [write_model(base, candidates, size, counts, excluded) for size in sizes]
        for b in built:
            print(json.dumps(b, ensure_ascii=False), flush=True)
        report, culprits = check(files, args, [Path(b["path"]) for b in built], work / f"verify_models_{n}.json")
        if not english_differs(report):
            for b in built:
                path = Path(b["path"])
                for f in (path, path.with_name(path.stem + "_init.json")):
                    os.replace(f, work / f.name)
            corpus.log(f"round {n}: no English row differs; models and their init maps in {work}")
            return
        if not culprits:
            break  # rows differ with no new piece in them: another round cannot help
        excluded |= culprits
    raise SystemExit(f"English rows still differ after round {n}; see {work}/verify_models_{n}.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run"])
    p.add_argument("sizes", nargs="*", type=int, help="slots to replace per model, multiples of 1000 (default 6000)")
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--base-model", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--workers", type=int, default=max(1, int(cpu_budget())),
                   help="corpus pass processes (default: the container's CPU quota)")
    p.add_argument("--sample", type=int, help="evenly spaced train shards to read, all validation (default: all)")
    args = p.parse_args()
    if any(n <= 0 or n % 1000 for n in args.sizes) or len(set(args.sizes)) < len(args.sizes):
        p.error("sizes must be distinct positive multiples of 1000")
    if args.sample is not None and args.sample <= 0:
        p.error("--sample must be positive")
    run(args)


if __name__ == "__main__":
    main()
