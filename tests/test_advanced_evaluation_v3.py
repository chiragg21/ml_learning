"""
Lite smoke tests for advanced evaluation: the text metrics against hand-
computed values, generation scoring with an optional judge, benchmark
suites, and calibration error on a model that is and isn't confident.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.data.seq2seq import TokenizedSeq2SeqDataset, make_collate_fn
from src.evaluation.advanced import (
    BenchmarkSuite,
    bleu,
    calibration_error,
    character_error_rate,
    chrf,
    edit_distance,
    evaluate_generation,
    exact_match,
    run_benchmarks,
)
from src.tokenizers.v0.tokenizer_v0 import CharTokenizer


def test_exact_match_ignores_surrounding_whitespace():

    assert exact_match(" abc ", "abc") == 1.0
    assert exact_match("abd", "abc") == 0.0


def test_edit_distance_and_character_error_rate():

    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("abcd", "abc") == pytest.approx(1 / 3)


def test_bleu_is_one_for_identical_text_and_zero_without_overlap():

    text = ["the cat sat on the mat today"]

    assert bleu(text, text) == pytest.approx(1.0)
    assert bleu(["xyz qrs"], text) == 0.0


def test_bleu_brevity_penalty_punishes_short_output():

    reference = ["the cat sat on the mat"]

    full = bleu(["the cat sat on the mat"], reference)
    short = bleu(["the cat sat"], reference)

    assert short < full


def test_bleu_matches_a_hand_computed_unigram_score():

    # 3 of 4 hypothesis words match; same length -> no brevity penalty
    score = bleu(["a b c x"], ["a b c d"], max_n=1)

    assert score == pytest.approx(0.75)


def test_bleu_can_work_on_characters():

    assert bleu(["abcdef"], ["abcdef"], tokenize="char") == pytest.approx(1.0)

    with pytest.raises(ValueError):
        bleu(["a"], ["a"], tokenize="syllable")


def test_chrf_orders_closer_text_higher():

    reference = ["the quick brown fox"]

    assert chrf(reference, reference) == pytest.approx(1.0)
    assert chrf(["the quick brown fax"], reference) < 1.0
    assert chrf(["the quick brown fax"], reference) > chrf(["zzz"], reference)
    assert chrf(["zzz"], reference) == 0.0


def test_metrics_validate_length_mismatch():

    with pytest.raises(ValueError):
        bleu(["a"], ["a", "b"])

    with pytest.raises(ValueError):
        chrf(["a"], ["a", "b"])


class _EchoGenerator:
    """Stands in for Generator: 'translates' by upper-casing."""

    def generate_text(self, text, **kwargs):
        return text.upper()


def test_evaluate_generation_scores_perfect_and_imperfect_output():

    pairs = [("ab", "AB"), ("cd", "CD"), ("ef", "xx")]

    report = evaluate_generation(_EchoGenerator(), pairs, tokenize="char")

    assert report.metrics["exact_match"] == pytest.approx(2 / 3)
    assert report.metrics["character_error_rate"] == pytest.approx(1 / 3)
    assert 0 < report.metrics["bleu"] < 1
    assert report.metrics["length_ratio"] == pytest.approx(1.0)
    assert report.samples[2]["hypothesis"] == "EF"
    assert "judge_score" not in report.metrics


def test_evaluate_generation_uses_a_judge():

    report = evaluate_generation(
        _EchoGenerator(),
        [("ab", "AB"), ("cd", "zz")],
        judge=lambda source, reference, hypothesis: float(hypothesis == reference),
    )

    assert report.metrics["judge_score"] == pytest.approx(0.5)


def test_run_benchmarks_scores_each_suite_separately():

    results = run_benchmarks(
        _EchoGenerator(),
        [
            BenchmarkSuite("easy", [("a", "A")]),
            BenchmarkSuite("hard", [("b", "Z")]),
        ],
    )

    assert results["easy"]["exact_match"] == 1.0
    assert results["hard"]["exact_match"] == 0.0


class _FixedModel(torch.nn.Module):
    """Predicts one fixed token with probability `confidence`."""

    def __init__(self, vocab_size, token, confidence):
        super().__init__()

        probs = torch.full((vocab_size,), (1 - confidence) / (vocab_size - 1))
        probs[token] = confidence

        self.register_buffer("logits_row", probs.log())

    def forward(self, source_ids, target_ids, **kwargs):

        batch, length = target_ids.shape

        class Output:
            logits = self.logits_row.expand(batch, length, -1)

        return Output


def _loader():

    tokenizer = CharTokenizer("ab")

    dataset = TokenizedSeq2SeqDataset([("ab", "ab")] * 8, tokenizer)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=make_collate_fn(tokenizer.pad_token_id)
    )

    return tokenizer, loader


def test_calibration_error_is_large_for_an_overconfident_model():

    tokenizer, loader = _loader()

    # always 99% sure of <BOS>, which is right for none of the labels
    model = _FixedModel(tokenizer.vocab_size, tokenizer.bos_token_id, 0.99)

    result = calibration_error(model, loader, tokenizer.pad_token_id)

    assert result["accuracy"] == 0.0
    assert result["ece"] == pytest.approx(0.99, abs=0.02)


def test_calibration_error_is_small_for_an_honest_model():

    tokenizer, loader = _loader()

    # labels are a, b, <EOS>: always predicting 'a' is right 1/3 of the
    # time, so 1/3 confidence is exactly calibrated
    model = _FixedModel(tokenizer.vocab_size, tokenizer.encode("a")[0], 1 / 3)

    result = calibration_error(model, loader, tokenizer.pad_token_id)

    assert result["ece"] < 0.02
    assert math.isclose(result["accuracy"], 1 / 3, abs_tol=0.01)
