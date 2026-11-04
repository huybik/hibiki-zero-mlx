"""The Vietnamese tokenizer swap: N NORMAL pieces English never uses become Vietnamese pieces, so
`text_card` stays 48,000 ([vi-tokenizer](../../docs/agent-memory/vi-tokenizer.md)).

  superset K  the base model with the top K new pieces appended (vocab over 48k): one English pass
              over it finds every new piece that would change an English row
  models N... 48k-vocab models with N never-used slots replaced, each with its embedding init map
              (<model>_init.json)
  run [N...]  the whole build on the box (N defaults to 6000): `corpus texts` and `count`, a 16k
              superset and its English check, then the models and their check, excluding each
              new piece that changed an English row and rebuilding until none does (5 rounds at
              most)

All files live in --work (default /workspace/vi_tokenizer): counts.pkl from `corpus count`,
excluded.json (the new pieces that changed an English row, which `corpus verify` adds to), the
models, their init maps and the verify reports. From the repo root:

  python -m training.vi_tokenizer.build run
  python -m training.vi_tokenizer.build models 4000 6000 8000
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from training.paths import DEFAULT_TOKENIZER
from training.vi_tokenizer import corpus

WORDS = Path(__file__).with_name("english_words.txt.gz")  # macOS /usr/share/dict/words (Webster's 2nd)
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
            if ord(c) > 127 and base_sp.piece_to_id(c) == base_sp.unk_id() and not en_chars[c] and c not in excluded]


def ranked_forms(base: pb.ModelProto, chars: list[str], counts: dict, excluded: set) -> list[tuple[str, int, int]]:
    """(surface, count, tokens saved) of train text_vi word forms, most saved first.

    A form is an alphabetic word core as it occurs: `▁core` after a space, bare `core` after
    leading punctuation, in its own case. Savings are counted against base + the new letters."""
    m0 = pb.ModelProto()
    m0.CopyFrom(base)
    lo, _ = score_range(base)
    for c in chars:
        m0.pieces.add(piece=c, score=lo + 0.01, type=T.NORMAL)
    sp0, sp0_bare = processor(m0), processor(m0, dummy_prefix=False)
    forms = counts["train"]["forms"]
    keys = list(forms)
    prefixed = iter(sp0.encode([core for bare, core in keys if not bare]))
    bare = iter(sp0_bare.encode([core for b, core in keys if b]))
    out = []
    for b, core in keys:
        ids = next(bare) if b else next(prefixed)
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


def shadowing(base: pb.ModelProto, model: pb.ModelProto, excluded: set) -> set:
    """New pieces that re-split an excluded surface (one known to occur in English), e.g. bare
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
            model.pieces[slot].type = T.NORMAL
        more = shadowing(base, model, excluded | shadowed)
        if not more:
            break
        shadowed |= more
    assert len(model.pieces) == 48000 and len({p.piece for p in model.pieces}) == 48000
    assert score_range(model) == (lo, hi), "unk penalty and user-symbol scores must not move"
    path = work / f"{base_path.stem}_vi{n // 1000}k.model"
    path.write_bytes(model.SerializeToString())

    # Each new piece is one token on its own surface.
    sp, sp_bare = processor(model), processor(model, dummy_prefix=False)
    not_whole = [p for p, slot in zip(new, slots)
                 if (sp.encode(p[1:]) if p.startswith("▁") else sp_bare.encode(p)) != [slot]]

    # Embedding init recipe: the ORIGINAL tokenizer's split of each new piece.
    orig, orig_bare = processor(base), processor(base, dummy_prefix=False)
    slot_set = set(slots)
    recipe, kinds, init_uses_slot, init_uses_byte = [], {}, 0, 0
    for slot, piece in zip(slots, new):
        ids = orig.encode(piece[1:]) if piece.startswith("▁") else orig_bare.encode(piece)
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
    keys = list(forms)
    prefixed = iter(sp.encode([c for b, c in keys if not b]))
    bare = iter(sp_bare.encode([c for b, c in keys if b]))
    one_piece = sum(forms[k] for k in keys if len(next(bare) if k[0] else next(prefixed)) == 1)
    return {"n": n, "path": str(path), "kinds": kinds, "shadowed": sorted(shadowed), "not_whole": not_whole[:20], "n_not_whole": len(not_whole),
            "train_words_one_piece": one_piece / sum(forms.values()),
            "train_words_in_new_pieces": sum(f[1] for f in chosen) / sum(forms.values()),
            "last_count": chosen[-1][1], "last_gain": chosen[-1][2],
            "init_uses_replaced_slot": init_uses_slot, "init_uses_byte": init_uses_byte,
            "score_range_new": [min(scores.values()), max(scores.values())]}


def load(work: Path) -> tuple[dict, set]:
    with (work / "counts.pkl").open("rb") as f:
        counts = pickle.load(f)
    excluded = work / "excluded.json"
    return counts, set(json.loads(excluded.read_text())) if excluded.exists() else set()


def english_differs(report: dict) -> bool:
    return any(row["en_diff_rows"] for split in corpus.SPLITS for row in report[split][1:])


def run(args: argparse.Namespace) -> None:
    work, base = args.work, args.base_model
    corpus.texts(work)
    corpus.count(work, base, args.workers, args.sample)
    counts, excluded = load(work)
    superset = write_superset(base, work, SUPERSET, counts, excluded)
    corpus.verify(work, base, [superset], work / "verify_superset.json", args.workers, args.sample)
    for n in range(1, ROUNDS + 1):
        counts, excluded = load(work)
        print("excluded new pieces:", len(excluded), flush=True)
        models = []
        for size in args.sizes or [6000]:
            built = write_model(base, work, size, counts, excluded)
            print(json.dumps(built, ensure_ascii=False), flush=True)
            models.append(Path(built["path"]))
        report = corpus.verify(work, base, models, work / f"verify_models_{n}.json", args.workers, args.sample)
        if not english_differs(report):
            corpus.log(f"round {n}: no English row differs; models and their init maps in {work}")
            return
    raise SystemExit(f"English rows still differ after {ROUNDS} rounds; see {work}/verify_models_{ROUNDS}.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["superset", "models", "run"])
    p.add_argument("sizes", nargs="*", type=int, help="superset: K; models and run: N...")
    p.add_argument("--work", type=Path, default=corpus.WORK)
    p.add_argument("--base-model", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="run: corpus pass processes")
    p.add_argument("--sample", type=int, help="run: evenly spaced train shards to read (default: all)")
    args = p.parse_args()
    if args.step == "run":
        run(args)
        return
    if not args.sizes:
        p.error(f"{args.step} needs sizes")
    counts, excluded = load(args.work)
    print("excluded new pieces:", len(excluded))
    for size in args.sizes:
        if args.step == "superset":
            write_superset(args.base_model, args.work, size, counts, excluded)
        else:
            print(json.dumps(write_model(args.base_model, args.work, size, counts, excluded), ensure_ascii=False))


if __name__ == "__main__":
    main()
