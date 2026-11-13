from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.generator.generator import Generator

"""
Inference-time compute: improving output by spending more compute at
generation time rather than by training a bigger model.

Greedy decoding commits to the single most likely token at every step.
These helpers instead draw several candidate outputs, then pick among
them:

  - best_of_n: sample N candidates, score each, return the best. The
    default score is the model's own length-normalised log-probability
    of the candidate; pass any `verifier` to score differently (a
    reference-based check, a second model, a rule). Quality rises with N
    because the best of many draws is better than a typical draw, as
    long as the scorer actually prefers good outputs - a weak verifier
    caps the gain - and cost grows linearly with N.
  - majority_vote (self-consistency): when a task has a single correct
    answer, the answer most candidates agree on is more reliable than any
    one sample.
  - verify_and_vote: filter by verifier first, then vote among survivors.

Candidates are lists of token ids starting at <BOS> and ending at the
first <EOS> (kept), like Generator.generate() but with the junk it emits
after <EOS> for already-finished rows removed.
"""


@dataclass
class CandidateSet:
    candidates: list[list[int]]
    scores: list[float]

    @property
    def best_index(self) -> int:
        return max(range(len(self.scores)), key=self.scores.__getitem__)

    @property
    def best(self) -> list[int]:
        return self.candidates[self.best_index]


def truncate_at_eos(token_ids: list[int], eos_token_id: int) -> list[int]:

    if eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id) + 1]

    return token_ids


def sample_candidates(
    generator: Generator,
    source_ids: torch.Tensor,
    num_samples: int,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[list[int]]:
    """source_ids: (1, S). Returns num_samples sampled candidates."""

    if source_ids.size(0) != 1:
        raise ValueError("source_ids must hold a single example, shape (1, S)")

    generated = generator.generate(
        source_ids.repeat(num_samples, 1),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
    )

    eos_token_id = generator.tokenizer.eos_token_id

    return [
        truncate_at_eos(row.tolist(), eos_token_id)
        for row in generated
    ]


@torch.no_grad()
def score_candidates(
    generator: Generator,
    source_ids: torch.Tensor,
    candidates: list[list[int]],
    length_penalty: float = 1.0,
) -> list[float]:
    """
    Model log-probability of each candidate given the source, divided by
    length**length_penalty (1.0 = mean per-token log-prob; 0.0 = raw
    sum, which favours short outputs).
    """

    pad_token_id = generator.tokenizer.pad_token_id
    device = generator.device

    generator.model.eval()

    target = pad_sequence(
        [torch.tensor(c, dtype=torch.long) for c in candidates],
        batch_first=True,
        padding_value=pad_token_id,
    ).to(device)

    source = source_ids.to(device).repeat(len(candidates), 1)

    decoder_input = target[:, :-1]
    labels = target[:, 1:]

    logits = generator.model(
        source_ids=source,
        target_ids=decoder_input,
        source_key_padding_mask=source == pad_token_id,
        target_key_padding_mask=decoder_input == pad_token_id,
    ).logits

    log_probs = -nn.functional.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=pad_token_id,
        reduction="none",
    )

    lengths = (labels != pad_token_id).sum(dim=1).clamp(min=1)

    return (log_probs.sum(dim=1) / lengths.float() ** length_penalty).tolist()


def best_of_n(
    generator: Generator,
    source_ids: torch.Tensor,
    num_samples: int = 8,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    verifier: Callable[[torch.Tensor, list[int]], float] | None = None,
    length_penalty: float = 1.0,
) -> CandidateSet:
    """
    Samples num_samples candidates and scores them - with `verifier`
    (called as verifier(source_ids, candidate_ids) -> float, higher is
    better) if given, otherwise with the model's own log-probability.
    """

    candidates = sample_candidates(
        generator,
        source_ids,
        num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    if verifier is None:
        scores = score_candidates(
            generator,
            source_ids,
            candidates,
            length_penalty=length_penalty,
        )
    else:
        scores = [float(verifier(source_ids, c)) for c in candidates]

    return CandidateSet(candidates, scores)


def majority_vote(
    candidates: list[list[int]],
    scores: list[float] | None = None,
) -> tuple[list[int], int]:
    """
    The most common candidate and how many times it appeared. Ties go to
    the one with the higher score (if scores are given), else the one
    seen first.
    """

    counts = Counter(tuple(c) for c in candidates)

    def rank(item):
        sequence, count = item
        best_score = (
            max(s for c, s in zip(candidates, scores) if tuple(c) == sequence)
            if scores is not None
            else 0.0
        )
        return (count, best_score)

    sequence, count = max(counts.items(), key=rank)

    return list(sequence), count


def verify_and_vote(
    generator: Generator,
    source_ids: torch.Tensor,
    verifier: Callable[[torch.Tensor, list[int]], float],
    num_samples: int = 8,
    *,
    threshold: float = 0.5,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[list[int], int]:
    """
    Self-verification: keep only candidates the verifier scores at or
    above `threshold`, then majority-vote among them. If none pass, falls
    back to voting over all candidates, so there is always an answer.
    """

    result = best_of_n(
        generator,
        source_ids,
        num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        verifier=verifier,
    )

    passed = [
        (c, s) for c, s in zip(result.candidates, result.scores)
        if s >= threshold
    ]

    if not passed:
        return majority_vote(result.candidates, result.scores)

    return majority_vote([c for c, _ in passed], [s for _, s in passed])
