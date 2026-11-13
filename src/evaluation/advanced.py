from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import torch

"""
Advanced evaluation. Evaluator (evaluator.py) answers "how well does the
model predict the next reference token" - loss, perplexity, token
accuracy under teacher forcing. That says nothing about what the model
*writes* when left to generate on its own, which is what anyone uses it
for. This module scores real generations:

  - exact_match, character_error_rate: strict and edit-based agreement
    with the reference (right for tasks with one correct output, like
    the toy copy task).
  - bleu, chrf: n-gram overlap with the reference - BLEU on words (or
    characters), chrF on character n-grams, which is kinder to
    morphology and tokenization differences. Overlap metrics reward
    surface similarity, not meaning, so they are a proxy.
  - judge: a pluggable callable for qualitative scoring - another model
    or a rule-based grader; pass any (source, reference, hypothesis) ->
    score in [0, 1]. No judge model is bundled.
  - BenchmarkSuite/run_benchmarks: named held-out sets scored
    separately, so a regression on one kind of input isn't hidden by an
    average.
  - calibration_error: expected calibration error of the model's
    next-token probabilities - when it says it is 90% sure, is it right
    90% of the time? Perplexity can look fine while confidence is badly
    off.
"""


# ==================================================
# Text metrics
# ==================================================

def _tokens(text: str, tokenize: str) -> list[str]:

    if tokenize == "word":
        return text.split()

    if tokenize == "char":
        return list(text)

    raise ValueError("tokenize must be 'word' or 'char'")


def _ngrams(tokens: list[str], n: int) -> Counter:

    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def exact_match(hypothesis: str, reference: str) -> float:

    return float(hypothesis.strip() == reference.strip())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertions, deletions, substitutions)."""

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )

        previous = current

    return previous[-1]


def character_error_rate(hypothesis: str, reference: str) -> float:
    """Edit distance / reference length; 0 is perfect, can exceed 1."""

    return edit_distance(hypothesis, reference) / max(len(reference), 1)


def bleu(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
    tokenize: str = "word",
) -> float:
    """
    Corpus BLEU in [0, 1] (Papineni et al., 2002): the geometric mean of
    clipped n-gram precisions for n = 1..max_n, times a brevity penalty
    that punishes outputs shorter than the reference. n-gram counts are
    pooled over the whole corpus, and orders above 1 use add-one
    smoothing so a short corpus with no 4-gram match scores low instead
    of exactly zero.
    """

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")

    matches = [0] * max_n
    totals = [0] * max_n
    hypothesis_length = 0
    reference_length = 0

    for hypothesis, reference in zip(hypotheses, references):

        hyp = _tokens(hypothesis, tokenize)
        ref = _tokens(reference, tokenize)

        hypothesis_length += len(hyp)
        reference_length += len(ref)

        for n in range(1, max_n + 1):

            hyp_counts = _ngrams(hyp, n)
            ref_counts = _ngrams(ref, n)

            matches[n - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in hyp_counts.items()
            )
            totals[n - 1] += sum(hyp_counts.values())

    if hypothesis_length == 0 or matches[0] == 0:
        return 0.0

    log_precision = 0.0

    for n in range(max_n):

        if n == 0:
            precision = matches[0] / totals[0]
        else:
            precision = (matches[n] + 1) / (totals[n] + 1)

        log_precision += math.log(precision) / max_n

    brevity_penalty = (
        1.0
        if hypothesis_length >= reference_length
        else math.exp(1 - reference_length / hypothesis_length)
    )

    return brevity_penalty * math.exp(log_precision)


def chrf(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """
    chrF in [0, 1] (Popovic, 2015): F-beta score of character n-gram
    precision and recall, averaged over n = 1..max_n (spaces ignored).
    beta = 2 weights recall twice as much as precision.
    """

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")

    precisions, recalls = [], []

    for n in range(1, max_n + 1):

        matched = hyp_total = ref_total = 0

        for hypothesis, reference in zip(hypotheses, references):

            hyp_counts = _ngrams(list(hypothesis.replace(" ", "")), n)
            ref_counts = _ngrams(list(reference.replace(" ", "")), n)

            matched += sum(
                min(count, ref_counts[gram]) for gram, count in hyp_counts.items()
            )
            hyp_total += sum(hyp_counts.values())
            ref_total += sum(ref_counts.values())

        precisions.append(matched / hyp_total if hyp_total else 0.0)
        recalls.append(matched / ref_total if ref_total else 0.0)

    precision = sum(precisions) / max_n
    recall = sum(recalls) / max_n

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


# ==================================================
# Scoring a generator on held-out pairs
# ==================================================

@dataclass
class GenerationReport:
    metrics: dict[str, float]
    samples: list[dict[str, str]] = field(default_factory=list)


def evaluate_generation(
    generator,
    pairs: list[tuple[str, str]],
    *,
    max_new_tokens: int = 128,
    tokenize: str = "word",
    judge: Callable[[str, str, str], float] | None = None,
    num_samples: int = 5,
) -> GenerationReport:
    """
    Greedily generates a hypothesis for every (source, reference) pair
    and scores them all. `judge`, if given, adds a "judge_score" metric
    (its mean); the first `num_samples` generations are kept so results
    can be read, not just averaged.
    """

    hypotheses = [
        generator.generate_text(
            source,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        for source, _ in pairs
    ]

    references = [reference for _, reference in pairs]

    count = max(len(pairs), 1)

    metrics = {
        "exact_match": sum(
            exact_match(h, r) for h, r in zip(hypotheses, references)
        ) / count,
        "character_error_rate": sum(
            character_error_rate(h, r) for h, r in zip(hypotheses, references)
        ) / count,
        "bleu": bleu(hypotheses, references, tokenize=tokenize),
        "chrf": chrf(hypotheses, references),
        "length_ratio": (
            sum(len(h) for h in hypotheses) / max(sum(len(r) for r in references), 1)
        ),
    }

    if judge is not None:
        metrics["judge_score"] = sum(
            judge(source, reference, hypothesis)
            for (source, reference), hypothesis in zip(pairs, hypotheses)
        ) / count

    samples = [
        {"source": source, "reference": reference, "hypothesis": hypothesis}
        for (source, reference), hypothesis in list(zip(pairs, hypotheses))[:num_samples]
    ]

    return GenerationReport(metrics=metrics, samples=samples)


@dataclass
class BenchmarkSuite:
    name: str
    pairs: list[tuple[str, str]]


def run_benchmarks(
    generator,
    suites: list[BenchmarkSuite],
    **evaluation_options,
) -> dict[str, dict[str, float]]:
    """Scores each suite separately; returns {suite name: metrics}."""

    return {
        suite.name: evaluate_generation(
            generator,
            suite.pairs,
            **evaluation_options,
        ).metrics
        for suite in suites
    }


# ==================================================
# Calibration
# ==================================================

@torch.no_grad()
def calibration_error(
    model,
    dataloader,
    pad_token_id: int,
    device: str | torch.device = "cpu",
    num_bins: int = 10,
) -> dict[str, float]:
    """
    Expected calibration error of next-token predictions under teacher
    forcing. Predictions are grouped by confidence (the probability of
    the top token) into num_bins bins; ECE is the average gap between a
    bin's confidence and its actual accuracy, weighted by how many
    predictions fall in it. 0 = perfectly calibrated. Also returns the
    mean confidence and accuracy it is built from.
    """

    device = torch.device(device)

    model.eval()

    bin_counts = torch.zeros(num_bins)
    bin_confidence = torch.zeros(num_bins)
    bin_correct = torch.zeros(num_bins)

    for batch in dataloader:

        source_ids = batch["source_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]

        logits = model(
            source_ids=source_ids,
            target_ids=decoder_input_ids,
            source_key_padding_mask=source_ids == pad_token_id,
            target_key_padding_mask=decoder_input_ids == pad_token_id,
        ).logits

        confidence, prediction = torch.softmax(logits.float(), dim=-1).max(dim=-1)

        keep = labels != pad_token_id

        confidence = confidence[keep].cpu()
        correct = (prediction == labels)[keep].cpu().float()

        bins = (confidence * num_bins).long().clamp(max=num_bins - 1)

        bin_counts.index_add_(0, bins, torch.ones_like(confidence))
        bin_confidence.index_add_(0, bins, confidence)
        bin_correct.index_add_(0, bins, correct)

    total = bin_counts.sum().clamp(min=1.0)

    gap = (bin_confidence - bin_correct).abs()

    return {
        "ece": (gap.sum() / total).item(),
        "mean_confidence": (bin_confidence.sum() / total).item(),
        "accuracy": (bin_correct.sum() / total).item(),
    }
