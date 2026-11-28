"""Scores of free-running translations: chrF and BLEU, word error rate, length and repetition,
LAAL, the voice scores' means, and the health gates each set must pass (`score_sets`).
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np
import sacrebleu

from training.finetune import voice_scores
from training.finetune.eval_rows import EVAL_SET


def ascii_text(text: str) -> str:
    """Remove Latin diacritics for stable word-metric normalization."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).encode(
        "ascii", "ignore"
    ).decode("ascii")


def normalize(text: str) -> str:
    text = ascii_text(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_words = normalize(reference).split()
        edits += edit_distance(ref_words, normalize(hypothesis).split())
        total_words += len(ref_words)
    return edits / total_words if total_words else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def max_repeated_ngram(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for start in range(len(words) - n + 1):
        ngram = tuple(words[start : start + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values(), default=0)


def length_repetition_metrics(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    reference_lengths = [len(normalize(reference).split()) for reference in references]
    hypothesis_lengths = [len(normalize(hypothesis).split()) for hypothesis in hypotheses]
    length_ratios = [
        hyp_len / ref_len
        for ref_len, hyp_len in zip(reference_lengths, hypothesis_lengths, strict=True)
        if ref_len > 0
    ]
    repeated_4grams = [
        max_repeated_ngram(normalize(hypothesis).split(), 4) for hypothesis in hypotheses
    ]
    return {
        "mean_reference_words": mean([float(v) for v in reference_lengths]),
        "mean_prediction_words": mean([float(v) for v in hypothesis_lengths]),
        "mean_length_ratio": mean(length_ratios),
        "overlong_predictions": sum(
            1
            for ref_len, hyp_len in zip(reference_lengths, hypothesis_lengths, strict=True)
            if hyp_len > max(32, 2 * ref_len)
        ),
        "repeated_4gram_predictions": sum(1 for value in repeated_4grams if value >= 3),
        "max_repeated_4gram_count": max(repeated_4grams, default=0),
    }


def is_bool_text(value: Any, expected: bool) -> bool:
    return value is expected or str(value).lower() == str(expected).lower()


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    references = [str(record["reference_text"]).strip() for record in records]
    hypotheses = [str(record["prediction_text"]).strip() for record in records]
    nonempty_pairs = [
        (reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
        if hypothesis
    ]
    metrics: dict[str, Any] = {
        "num_predictions": len(records),
        "nonempty_predictions": sum(1 for hypothesis in hypotheses if hypothesis),
        "empty_predictions": sum(1 for hypothesis in hypotheses if not hypothesis),
        "exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if reference == hypothesis
        ),
        "normalized_exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if normalize(reference) == normalize(hypothesis)
        ),
        "eos_found": sum(1 for record in records if is_bool_text(record.get("eos_found"), True)),
        "eos_missing": sum(1 for record in records if is_bool_text(record.get("eos_found"), False)),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(length_repetition_metrics(references, hypotheses))
    offsets = [float(record["end_offset_s"]) for record in records if record["end_offset_s"] != ""]
    metrics["median_end_offset_s"] = float(np.median(offsets)) if offsets else float("nan")
    if "laal_s" in records[0]:  # eval.py --laal
        laals = [float(record["laal_s"]) for record in records if record["laal_s"] != ""]
        metrics["mean_laal_s"] = float(np.mean(laals)) if laals else float("nan")
    if "speaker_sim" in records[0]:  # the speech was scored: rows with under a second of it are left out
        for key in voice_scores.COLUMNS:
            values = [float(record[key]) for record in records if record[key] != ""]
            metrics[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    metrics["bleu"] = sacrebleu.corpus_bleu(hypotheses, [references]).score
    metrics["chrf"] = sacrebleu.corpus_chrf(hypotheses, [references]).score
    if nonempty_pairs:
        nonempty_refs, nonempty_hyps = zip(*nonempty_pairs, strict=True)
        metrics["nonempty_bleu"] = sacrebleu.corpus_bleu(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
        metrics["nonempty_chrf"] = sacrebleu.corpus_chrf(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
    else:
        metrics["nonempty_bleu"] = 0.0
        metrics["nonempty_chrf"] = 0.0
    return metrics


def laal(word_times: list[float], source_s: float, reference_words: int) -> float:
    """Length-Adaptive Average Lagging (Papi et al. 2022) in seconds, as the Hibiki-Zero paper
    computes it: word i's time from the start of the source less (i - 1) * gamma, where an even
    pace over the source would put it, with gamma = source_s / max(generated words, reference
    words); averaged over the words up to the first that comes at or after the source's end."""
    gamma = source_s / max(len(word_times), reference_words)
    lags: list[float] = []
    for index, time_s in enumerate(word_times):
        lags.append(time_s - index * gamma)
        if time_s >= source_s:
            break
    return float(np.mean(lags))


def generation_health(metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Gates against broken output: empty, looping or overlong. A tenth of a set's rows may be
    empty: a model that leaves 2-5% of short clips silent is not broken, and a tighter gate (8
    of 180 until 2026-11-10) failed on the sampling seed alone. The EOS count is reported, not
    gated: every checkpoint of the stream runs missed a 91% EOS gate (at best 268 of 361 FLEURS
    dev rows) while their chrF rose, so it blocked every promotion."""
    count = int(metrics["num_predictions"])
    gates = {
        "min_nonempty_predictions": count - count // 10,
        "max_repeated_4gram_predictions": math.floor(count * 12 / 128),
        "max_mean_length_ratio": 2.0,
    }
    eligible = (
        int(metrics["nonempty_predictions"]) >= gates["min_nonempty_predictions"]
        and int(metrics["repeated_4gram_predictions"])
        <= gates["max_repeated_4gram_predictions"]
        and float(metrics["mean_length_ratio"]) <= gates["max_mean_length_ratio"]
    )
    return eligible, gates


def score_sets(records: list[dict[str, str]], names: list[str]) -> dict[str, dict[str, Any]]:
    """Metrics and health gates of each set in `names` that has records, in that order. Sets are
    scored apart: corpus chrF and BLEU weigh rows by length, and the gates must hold on each."""
    scores: dict[str, dict[str, Any]] = {}
    for name in names:
        group = [record for record in records if record[EVAL_SET] == name]
        if not group:
            continue
        metrics = score_records(group)
        healthy, gates = generation_health(metrics)
        metrics.update(health_eligible=healthy, health_gates=gates)
        scores[name] = metrics
    return scores


def score_summary(metrics: dict[str, Any]) -> str:
    """One line of a set's metrics, as eval.py and train.py print them."""
    count = metrics["num_predictions"]
    laal_part = f"laal={metrics['mean_laal_s']:.1f}s " if "mean_laal_s" in metrics else ""
    voice_part = (
        f"speaker_sim={metrics['mean_speaker_sim']:.2f} hnr={metrics['mean_hnr_db']:.1f}dB "
        if "mean_speaker_sim" in metrics
        else ""
    )
    return (
        f"chrf={metrics['chrf']:.2f} bleu={metrics['bleu']:.2f} "
        f"length_ratio={metrics['mean_length_ratio']:.2f} eos={metrics['eos_found']}/{count} "
        f"empty={metrics['empty_predictions']} repeat4={metrics['repeated_4gram_predictions']} "
        f"end_offset={metrics['median_end_offset_s']:.1f}s {laal_part}{voice_part}"
        f"health={'pass' if metrics['health_eligible'] else 'fail'}"
    )
