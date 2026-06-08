"""Metrics for Vietnamese spelling/diacritic correction."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    """Levenshtein edit distance using insertion, deletion, and substitution."""
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            substitution = previous[j - 1] + int(ref_item != pred_item)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def word_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def char_tokens(text: str) -> List[str]:
    return list(normalize_text(text))


def ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


@dataclass
class MetricAccumulator:
    """Streaming corpus metrics: CER, WER, and BLEU-4."""

    max_bleu_order: int = 4
    cer_edits: int = 0
    cer_ref_units: int = 0
    wer_edits: int = 0
    wer_ref_units: int = 0
    pred_length: int = 0
    ref_length: int = 0
    matches_by_order: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    possible_matches_by_order: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def add(self, prediction: str, reference: str) -> None:
        prediction = normalize_text(prediction)
        reference = normalize_text(reference)

        pred_chars = char_tokens(prediction)
        ref_chars = char_tokens(reference)
        self.cer_edits += edit_distance(ref_chars, pred_chars)
        self.cer_ref_units += len(ref_chars)

        pred_words = word_tokens(prediction)
        ref_words = word_tokens(reference)
        self.wer_edits += edit_distance(ref_words, pred_words)
        self.wer_ref_units += len(ref_words)

        self.pred_length += len(pred_words)
        self.ref_length += len(ref_words)
        for order in range(1, self.max_bleu_order + 1):
            pred_ngram_counts = ngram_counts(pred_words, order)
            ref_ngram_counts = ngram_counts(ref_words, order)
            overlap = pred_ngram_counts & ref_ngram_counts
            self.matches_by_order[order - 1] += sum(overlap.values())
            self.possible_matches_by_order[order - 1] += max(len(pred_words) - order + 1, 0)

    def add_many(self, predictions: Iterable[str], references: Iterable[str]) -> None:
        for prediction, reference in zip(predictions, references):
            self.add(prediction, reference)

    def compute(self) -> dict[str, float]:
        precisions = []
        for matches, possible in zip(self.matches_by_order, self.possible_matches_by_order):
            if possible > 0:
                precisions.append(matches / possible)

        if not precisions or min(precisions) <= 0.0:
            geo_mean = 0.0
        else:
            geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / len(precisions))

        if self.pred_length == 0:
            brevity_penalty = 0.0
        elif self.pred_length > self.ref_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - self.ref_length / self.pred_length)

        return {
            "cer": self.cer_edits / self.cer_ref_units if self.cer_ref_units else 0.0,
            "wer": self.wer_edits / self.wer_ref_units if self.wer_ref_units else 0.0,
            "bleu": 100.0 * brevity_penalty * geo_mean,
        }


def compute_text_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    accumulator = MetricAccumulator()
    accumulator.add_many(predictions, references)
    return accumulator.compute()
