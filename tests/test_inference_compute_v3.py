"""
Lite smoke tests for inference-time compute helpers: candidate sampling
and truncation, log-probability scoring, best-of-N selection with the
default and a custom verifier, majority vote and verify-and-vote.
"""

from __future__ import annotations

import torch

from src.generator.generator import Generator
from src.generator.inference_compute import (
    best_of_n,
    majority_vote,
    sample_candidates,
    score_candidates,
    truncate_at_eos,
    verify_and_vote,
)
from src.models.v3.datamodels_v3 import (
    DecoderConfigV3,
    EncoderConfigV3,
    TransformerConfigV3,
)
from src.models.v3.transformer_v3 import TransformerV3
from src.tokenizers.v0.tokenizer_v0 import CharTokenizer


TOKENIZER = CharTokenizer("abcxy")


def _generator() -> Generator:

    torch.manual_seed(0)

    parameters = dict(
        max_seq_len=24, d_model=32, num_heads=4, d_ff=64,
        num_layers=2, num_kv_heads=2, dropout_param=0.0,
        num_experts=2, top_k=1,
    )

    model = TransformerV3(
        TransformerConfigV3(
            encoder_config=EncoderConfigV3(vocab_size=TOKENIZER.vocab_size, **parameters),
            decoder_config=DecoderConfigV3(target_vocab_size=TOKENIZER.vocab_size, **parameters),
        )
    ).eval()

    generator = object.__new__(Generator)
    generator.model = model
    generator.tokenizer = TOKENIZER
    generator.device = torch.device("cpu")

    return generator


SOURCE = torch.tensor([TOKENIZER.encode("abcabc")])


def test_truncate_at_eos_keeps_the_eos():

    assert truncate_at_eos([1, 5, 6, 3, 9, 9], eos_token_id=3) == [1, 5, 6, 3]
    assert truncate_at_eos([1, 5, 6], eos_token_id=3) == [1, 5, 6]


def test_sampled_candidates_start_with_bos_and_stop_at_eos():

    generator = _generator()

    candidates = sample_candidates(generator, SOURCE, 6, max_new_tokens=10)

    eos = TOKENIZER.eos_token_id

    assert len(candidates) == 6

    for candidate in candidates:
        assert candidate[0] == TOKENIZER.bos_token_id
        assert eos not in candidate[:-1]


def test_score_matches_a_manual_log_probability():

    generator = _generator()

    candidate = [TOKENIZER.bos_token_id] + TOKENIZER.encode("abx") + [TOKENIZER.eos_token_id]

    (score,) = score_candidates(generator, SOURCE, [candidate])

    with torch.no_grad():
        logits = generator.model(
            source_ids=SOURCE,
            target_ids=torch.tensor([candidate[:-1]]),
        ).logits[0]

    log_probs = torch.log_softmax(logits, dim=-1)
    manual = sum(log_probs[i, t] for i, t in enumerate(candidate[1:]))

    assert abs(score - manual.item() / 4) < 1e-5


def test_scoring_does_not_depend_on_batch_padding():

    generator = _generator()

    short = [TOKENIZER.bos_token_id, 6, TOKENIZER.eos_token_id]
    long = [TOKENIZER.bos_token_id, 5, 6, 7, 5, TOKENIZER.eos_token_id]

    together = score_candidates(generator, SOURCE, [short, long])
    alone = score_candidates(generator, SOURCE, [short])

    assert abs(together[0] - alone[0]) < 1e-5


def test_best_of_n_returns_the_highest_scoring_candidate():

    generator = _generator()

    result = best_of_n(generator, SOURCE, num_samples=8, max_new_tokens=8)

    assert len(result.candidates) == len(result.scores) == 8
    assert result.scores[result.best_index] == max(result.scores)
    assert result.best == result.candidates[result.best_index]


def test_best_of_n_accepts_a_custom_verifier():

    generator = _generator()

    # prefer the longest candidate
    result = best_of_n(
        generator, SOURCE, num_samples=8, max_new_tokens=8,
        verifier=lambda source, candidate: len(candidate),
    )

    assert len(result.best) == max(len(c) for c in result.candidates)


def test_majority_vote_picks_the_most_common_and_breaks_ties_by_score():

    assert majority_vote([[1, 2], [3], [1, 2]]) == ([1, 2], 2)
    assert majority_vote([[1], [2]], scores=[0.1, 0.9]) == ([2], 1)


def test_verify_and_vote_filters_then_votes_and_falls_back():

    generator = _generator()

    answer, count = verify_and_vote(
        generator, SOURCE, verifier=lambda s, c: 1.0, num_samples=6, max_new_tokens=6,
    )

    assert answer[0] == TOKENIZER.bos_token_id and 1 <= count <= 6

    fallback, _ = verify_and_vote(
        generator, SOURCE, verifier=lambda s, c: 0.0, num_samples=6, max_new_tokens=6,
    )

    assert fallback[0] == TOKENIZER.bos_token_id
