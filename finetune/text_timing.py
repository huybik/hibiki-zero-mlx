"""Word-time English text tokens for the grounded-v2 training cache."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

ALIGNER_REPO = "facebook/wav2vec2-base-960h"
ALIGNMENT_NAME = "wav2vec2_ctc_word_v1"
DEFAULT_MIN_ALIGNMENT_SCORE = 0.5


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _spoken_group(text: str) -> str:
    """Normalize one SentencePiece word group for English CTC alignment."""
    from num2words import num2words

    text = _ascii(text)
    text = re.sub(
        r"\d+",
        lambda match: str(num2words(int(match.group(0)))).replace("-", " "),
        text,
    )
    return " ".join(re.sub(r"[^A-Za-z' ]+", " ", text).upper().split())


def sentencepiece_groups(text: str, tokenizer: Any) -> list[tuple[list[int], str]]:
    pieces = list(tokenizer.encode(text, out_type=str))
    if not pieces:
        raise ValueError("Cannot align empty English text")
    groups: list[list[str]] = []
    for piece in pieces:
        if piece.startswith("▁") or not groups:
            groups.append([piece])
        else:
            groups[-1].append(piece)
    result: list[tuple[list[int], str]] = []
    for group in groups:
        ids = [int(tokenizer.piece_to_id(piece)) for piece in group]
        spoken = _spoken_group(tokenizer.decode(ids))
        if not spoken:
            if not result:
                raise ValueError(f"Leading unalignable text group: {group!r}")
            result[-1][0].extend(ids)
            continue
        result.append((ids, spoken))
    return result


@dataclass(frozen=True)
class Alignment:
    spans: list[tuple[float, float]]
    score: float


def _ctc_token_frames(
    log_probs: Any, targets: list[int], blank_id: int
) -> tuple[list[list[int]], float]:
    """Viterbi-align one CTC emission matrix to a known token sequence."""
    import torch

    if not targets:
        raise ValueError("CTC target is empty")
    time_steps = int(log_probs.shape[0])
    states = 2 * len(targets) + 1
    if time_steps < states:
        raise ValueError(f"CTC emission is too short: {time_steps} < {states}")

    symbols = torch.full((states,), blank_id, dtype=torch.long, device=log_probs.device)
    symbols[1::2] = torch.tensor(targets, dtype=torch.long, device=log_probs.device)
    score = torch.full((states,), -torch.inf, device=log_probs.device)
    score[0] = log_probs[0, blank_id]
    score[1] = log_probs[0, targets[0]]
    back = torch.zeros((time_steps, states), dtype=torch.int8, device=log_probs.device)

    for time_index in range(1, time_steps):
        candidates = torch.stack(
            (
                score,
                torch.cat((score.new_full((1,), -torch.inf), score[:-1])),
                torch.cat((score.new_full((2,), -torch.inf), score[:-2])),
            )
        )
        can_skip = torch.zeros(states, dtype=torch.bool, device=log_probs.device)
        can_skip[3::2] = symbols[3::2] != symbols[1:-2:2]
        candidates[2, ~can_skip] = -torch.inf
        best_score, choice = candidates.max(dim=0)
        score = best_score + log_probs[time_index, symbols]
        back[time_index] = choice.to(dtype=torch.int8)

    state = states - 1 if score[-1] >= score[-2] else states - 2
    back = back.cpu()
    path = [state]
    for time_index in range(time_steps - 1, 0, -1):
        state -= int(back[time_index, state])
        path.append(state)
    path.reverse()

    frames: list[list[int]] = [[] for _ in targets]
    probability_times = []
    probability_tokens = []
    for time_index, state in enumerate(path):
        if state % 2:
            target_index = state // 2
            frames[target_index].append(time_index)
            probability_times.append(time_index)
            probability_tokens.append(targets[target_index])
    if any(not item for item in frames):
        raise RuntimeError("CTC alignment omitted a transcript token")
    probabilities = (
        log_probs[
            torch.tensor(probability_times, device=log_probs.device),
            torch.tensor(probability_tokens, device=log_probs.device),
        ]
        .exp()
        .cpu()
        .tolist()
    )
    return frames, sum(probabilities) / len(probabilities)


def _ctc_token_frames_many(
    log_probs: Any,
    targets_batch: list[list[int]],
    blank_id: int,
    output_lengths: list[int],
) -> list[tuple[list[list[int]], float] | Exception]:
    """Viterbi-align a padded CTC batch with one device loop per batch."""
    import torch

    batch_size = len(targets_batch)
    if not batch_size or len(output_lengths) != batch_size:
        raise ValueError("CTC alignment batch is empty or inconsistent")
    if any(not targets for targets in targets_batch):
        raise ValueError("CTC target is empty")
    if any(
        length < 2 * len(targets) + 1 or length > int(log_probs.shape[1])
        for targets, length in zip(targets_batch, output_lengths, strict=True)
    ):
        raise ValueError("CTC output lengths are invalid for the padded emissions")
    state_lengths = torch.tensor(
        [2 * len(targets) + 1 for targets in targets_batch],
        dtype=torch.long,
        device=log_probs.device,
    )
    max_states = int(state_lengths.max())
    symbols = torch.full(
        (batch_size, max_states), blank_id, dtype=torch.long, device=log_probs.device
    )
    for batch_index, targets in enumerate(targets_batch):
        symbols[batch_index, 1 : 2 * len(targets) : 2] = torch.tensor(
            targets, dtype=torch.long, device=log_probs.device
        )

    valid_states = torch.arange(max_states, device=log_probs.device)[None] < state_lengths[:, None]
    score = torch.full((batch_size, max_states), -torch.inf, device=log_probs.device)
    score[:, 0] = log_probs[:, 0, blank_id]
    score[:, 1] = log_probs[:, 0].gather(1, symbols[:, 1:2]).squeeze(1)
    max_time = max(output_lengths)
    back = torch.zeros(
        (max_time, batch_size, max_states), dtype=torch.int8, device=log_probs.device
    )
    can_skip = torch.zeros((batch_size, max_states), dtype=torch.bool, device=log_probs.device)
    can_skip[:, 3::2] = symbols[:, 3::2] != symbols[:, 1:-2:2]
    lengths = torch.tensor(output_lengths, dtype=torch.long, device=log_probs.device)
    last_states = state_lengths - 1
    penultimate_states = state_lengths - 2
    end_states = torch.zeros(batch_size, dtype=torch.long, device=log_probs.device)
    end_times = set(output_lengths)

    for time_index in range(1, max_time):
        candidates = torch.stack(
            (
                score,
                torch.cat((score.new_full((batch_size, 1), -torch.inf), score[:, :-1]), dim=1),
                torch.cat((score.new_full((batch_size, 2), -torch.inf), score[:, :-2]), dim=1),
            )
        )
        candidates[2, ~can_skip] = -torch.inf
        best_score, choice = candidates.max(dim=0)
        score = best_score + log_probs[:, time_index].gather(1, symbols)
        score.masked_fill_(~valid_states, -torch.inf)
        back[time_index] = choice.to(dtype=torch.int8)

        if time_index + 1 in end_times:
            last_score = score.gather(1, last_states[:, None]).squeeze(1)
            penultimate_score = score.gather(1, penultimate_states[:, None]).squeeze(1)
            final_state = torch.where(
                last_score >= penultimate_score, last_states, penultimate_states
            )
            end_states = torch.where(lengths == time_index + 1, final_state, end_states)

    back = back.cpu()
    end_states = end_states.cpu()
    probability_batches: list[int] = []
    probability_times: list[int] = []
    probability_tokens: list[int] = []
    probability_offsets = [0]
    frames_batch: list[list[list[int]] | Exception] = []
    for batch_index, (targets, output_length) in enumerate(
        zip(targets_batch, output_lengths, strict=True)
    ):
        state = int(end_states[batch_index])
        path = [state]
        for time_index in range(output_length - 1, 0, -1):
            state -= int(back[time_index, batch_index, state])
            path.append(state)
        path.reverse()

        frames: list[list[int]] = [[] for _ in targets]
        for time_index, state in enumerate(path):
            if state % 2:
                target_index = state // 2
                frames[target_index].append(time_index)
                probability_batches.append(batch_index)
                probability_times.append(time_index)
                probability_tokens.append(targets[target_index])
        if any(not item for item in frames):
            frames_batch.append(RuntimeError("CTC alignment omitted a transcript token"))
        else:
            frames_batch.append(frames)
        probability_offsets.append(len(probability_times))

    probabilities = []
    if probability_times:
        probabilities = (
            log_probs[
                torch.tensor(probability_batches, device=log_probs.device),
                torch.tensor(probability_times, device=log_probs.device),
                torch.tensor(probability_tokens, device=log_probs.device),
            ]
            .exp()
            .cpu()
            .tolist()
        )
    aligned: list[tuple[list[list[int]], float] | Exception] = []
    for index, frames in enumerate(frames_batch):
        if isinstance(frames, Exception):
            aligned.append(frames)
            continue
        left, right = probability_offsets[index : index + 2]
        aligned.append((frames, sum(probabilities[left:right]) / (right - left)))
    return aligned


class EnglishCTCAligner:
    def __init__(
        self,
        device: Any,
        min_score: float = DEFAULT_MIN_ALIGNMENT_SCORE,
        alignment_backend: str = "serial",
    ):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        if not 0 <= min_score <= 1:
            raise ValueError("Minimum CTC alignment score must be in [0, 1]")
        if alignment_backend not in ("serial", "batched"):
            raise ValueError("Alignment backend must be serial or batched")
        self.device = torch.device(device)
        self.min_score = min_score
        self.alignment_backend = alignment_backend
        self.processor = AutoProcessor.from_pretrained(ALIGNER_REPO)
        self.model = AutoModelForCTC.from_pretrained(ALIGNER_REPO).to(self.device).eval()
        tokenizer = self.processor.tokenizer
        self.vocab = tokenizer.get_vocab()
        self.blank_id = int(tokenizer.pad_token_id)
        self.delimiter = str(tokenizer.word_delimiter_token)

    def _targets(
        self, groups: list[tuple[list[int], str]]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        target_ids: list[int] = []
        ranges: list[tuple[int, int]] = []
        for group_index, (_, spoken) in enumerate(groups):
            if group_index:
                target_ids.append(self.vocab[self.delimiter])
            start = len(target_ids)
            for char in spoken.replace(" ", self.delimiter):
                if char not in self.vocab:
                    raise ValueError(f"CTC transcript contains unsupported character: {char!r}")
                target_ids.append(int(self.vocab[char]))
            ranges.append((start, len(target_ids)))
        return target_ids, ranges

    def align_many(
        self,
        waveforms_16khz: list[np.ndarray],
        grouped_text: list[list[tuple[list[int], str]]],
        batch_size: int,
        sample_budget: int = 0,
    ) -> list[Alignment | Exception]:
        import torch

        if len(waveforms_16khz) != len(grouped_text):
            raise ValueError("Waveform and transcript batches differ")
        if batch_size <= 0 or sample_budget < 0:
            raise ValueError("Alignment batch size must be positive and sample budget non-negative")
        results: list[Alignment | Exception] = []
        start = 0
        while start < len(waveforms_16khz):
            stop = start + 1
            max_samples = len(waveforms_16khz[start])
            while stop < len(waveforms_16khz) and stop - start < batch_size:
                next_max = max(max_samples, len(waveforms_16khz[stop]))
                if sample_budget and (stop - start + 1) * next_max > sample_budget:
                    break
                max_samples = next_max
                stop += 1
            wavs = waveforms_16khz[start:stop]
            groups_batch = grouped_text[start:stop]
            start = stop
            # wav2vec2-base uses group-normalized features and was not trained
            # with attention masks; layer-normalized variants require them.
            inputs = self.processor(
                wavs,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=self.model.config.feat_extract_norm == "layer",
            )
            input_values = inputs.input_values.to(self.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
            ):
                logits = self.model(input_values, attention_mask=attention_mask).logits
            input_lengths = torch.tensor([len(wav) for wav in wavs], device=self.device)
            output_lengths = self.model._get_feat_extract_output_lengths(input_lengths).tolist()
            prepared: list[tuple[int, list[int], list[tuple[int, int]], int]] = []
            batch_results: list[Alignment | Exception | None] = [None] * len(wavs)
            for local_index, (groups, output_length) in enumerate(
                zip(groups_batch, output_lengths, strict=True)
            ):
                try:
                    targets, ranges = self._targets(groups)
                    if int(output_length) < 2 * len(targets) + 1:
                        raise ValueError(
                            f"CTC emission is too short: {int(output_length)} < "
                            f"{2 * len(targets) + 1}"
                        )
                    prepared.append((local_index, targets, ranges, int(output_length)))
                except (KeyError, RuntimeError, ValueError) as exc:
                    batch_results[local_index] = exc

            if self.alignment_backend == "batched" and prepared:
                valid_indices = [item[0] for item in prepared]
                valid_lengths = [item[3] for item in prepared]
                emissions = logits[valid_indices, : max(valid_lengths)].float().log_softmax(-1)
                aligned = _ctc_token_frames_many(
                    emissions,
                    [item[1] for item in prepared],
                    self.blank_id,
                    valid_lengths,
                )
            else:
                aligned = []
                for local_index, targets, _, output_length in prepared:
                    try:
                        aligned.append(
                            _ctc_token_frames(
                                logits[local_index, :output_length].float().log_softmax(-1),
                                targets,
                                self.blank_id,
                            )
                        )
                    except (KeyError, RuntimeError, ValueError) as exc:
                        aligned.append(exc)

            for (local_index, _, ranges, output_length), alignment in zip(
                prepared, aligned, strict=True
            ):
                if isinstance(alignment, Exception):
                    batch_results[local_index] = alignment
                    continue
                token_frames, score = alignment
                try:
                    if score < self.min_score:
                        raise ValueError(
                            f"CTC alignment score {score:.3f} is below {self.min_score:.3f}"
                        )
                    denom = max(1, int(output_length))
                    spans = [
                        (
                            min(frame for frames in token_frames[left:right] for frame in frames)
                            / denom,
                            (
                                max(
                                    frame for frames in token_frames[left:right] for frame in frames
                                )
                                + 1
                            )
                            / denom,
                        )
                        for left, right in ranges
                    ]
                    if any(
                        not (0 <= left < right <= 1) or (index and left < spans[index - 1][0])
                        for index, (left, right) in enumerate(spans)
                    ):
                        raise ValueError("CTC alignment produced invalid word spans")
                    batch_results[local_index] = Alignment(spans=spans, score=score)
                except (KeyError, RuntimeError, ValueError) as exc:
                    batch_results[local_index] = exc
            if any(result is None for result in batch_results):
                raise RuntimeError("Missing CTC alignment result")
            results.extend(batch_results)  # type: ignore[arg-type]
            if self.device.type == "mps":
                del inputs, input_values, attention_mask, logits, input_lengths
                torch.mps.empty_cache()
        return results


def timed_sentencepiece_tokens(
    groups: list[tuple[list[int], str]],
    alignment: Alignment,
    raw_target_frames: int,
    delay_frames: int,
    eos_id: int,
) -> tuple[list[int], list[int]]:
    """Map word-aligned SentencePiece groups onto unique 12.5 Hz frames."""
    if len(groups) != len(alignment.spans):
        raise ValueError("Text groups and alignment spans differ")
    tokens: list[int] = []
    frames: list[int] = []
    previous = delay_frames - 1
    for (ids, _), (start_ratio, end_ratio) in zip(groups, alignment.spans, strict=True):
        start = delay_frames + int(math.floor(start_ratio * raw_target_frames))
        end = delay_frames + max(
            start - delay_frames, int(math.ceil(end_ratio * raw_target_frames)) - 1
        )
        if len(ids) == 1:
            candidates = [start]
        else:
            candidates = [
                round(start + index * max(0, end - start) / (len(ids) - 1))
                for index in range(len(ids))
            ]
        for token, frame in zip(ids, candidates, strict=True):
            frame = max(frame, previous + 1)
            tokens.append(token)
            frames.append(frame)
            previous = frame
    eos_frame = delay_frames + raw_target_frames
    if previous >= eos_frame:
        raise ValueError(
            f"Aligned text needs {previous - delay_frames + 1} frames but target has "
            f"{raw_target_frames}"
        )
    tokens.append(int(eos_id))
    frames.append(eos_frame)
    return tokens, frames
