"""
Lite smoke tests for v2's data-pipeline pieces (sequence packing,
streaming, length bucketing) and for KV-cached generation, on tiny
hand-made data. End-to-end training with them is covered in
test_smoke_training.py.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.seq2seq import (
    BucketBatchSampler,
    PackedSeq2SeqDataset,
    StreamingSeq2SeqDataset,
    TokenizedSeq2SeqDataset,
    make_collate_fn,
)
from src.generator.generator import Generator
from src.models.v2.datamodels_v2 import (
    DecoderConfigV2,
    EncoderConfigV2,
    TransformerConfigV2,
)
from src.models.v2.transformer_v2 import TransformerV2
from src.tokenizers.v0.tokenizer_v0 import CharTokenizer


PAIRS = [
    ("abc" * (i % 4 + 1), "xy" * (i % 3 + 1))
    for i in range(40)
]

TOKENIZER = CharTokenizer("abcxy")


def _dataset() -> TokenizedSeq2SeqDataset:
    return TokenizedSeq2SeqDataset(PAIRS, TOKENIZER)


# ==================================================
# Packing
# ==================================================

def test_packing_respects_both_budgets_and_keeps_every_example():

    packed = PackedSeq2SeqDataset(_dataset(), max_source_len=20, max_target_len=14)

    assert len(packed) < len(PAIRS)

    for row in packed.rows:
        assert len(row["source_ids"]) <= 20
        assert len(row["target_ids"]) <= 14

    assert sum(row["source_segment_ids"].max().item() for row in packed.rows) == len(PAIRS)


def test_packed_collate_pads_ids_and_marks_padding_with_segment_zero():

    packed = PackedSeq2SeqDataset(_dataset(), max_source_len=20, max_target_len=14)

    batch = make_collate_fn(TOKENIZER.pad_token_id)([packed[0], packed[-1]])

    padded = batch["source_ids"] == TOKENIZER.pad_token_id

    assert torch.equal(padded, batch["source_segment_ids"] == 0)
    assert batch["target_segment_ids"].shape == batch["target_ids"].shape


def test_unpacked_collate_has_no_segment_ids():

    batch = make_collate_fn(0)([_dataset()[0]])

    assert "source_segment_ids" not in batch


# ==================================================
# Streaming
# ==================================================

def test_streaming_yields_every_example_once_per_epoch():

    stream = StreamingSeq2SeqDataset(PAIRS, TOKENIZER, shuffle_buffer=8)

    first = [tuple(e["source_ids"].tolist()) for e in stream]
    second = [tuple(e["source_ids"].tolist()) for e in stream]

    expected = sorted(tuple(e["source_ids"].tolist()) for e in _dataset().examples)

    assert sorted(first) == expected
    assert sorted(second) == expected
    assert first != second


def test_streaming_workers_split_the_data_without_overlap():

    stream = StreamingSeq2SeqDataset(PAIRS, TOKENIZER, shuffle_buffer=8)

    loader = DataLoader(
        stream,
        batch_size=4,
        num_workers=2,
        collate_fn=make_collate_fn(TOKENIZER.pad_token_id),
    )

    assert sum(batch["source_ids"].shape[0] for batch in loader) == len(PAIRS)


def test_streaming_skips_examples_that_are_too_long():

    stream = StreamingSeq2SeqDataset(PAIRS, TOKENIZER, max_source_len=6)

    assert all(len(e["source_ids"]) <= 6 for e in stream)
    assert 0 < sum(1 for _ in stream) < len(PAIRS)


# ==================================================
# Length bucketing
# ==================================================

def test_bucketing_covers_every_index_once_and_reduces_padding():

    lengths = [(i * 7) % 50 + 1 for i in range(200)]

    batches = list(BucketBatchSampler(lengths, batch_size=8, chunk_factor=5))

    assert sorted(i for batch in batches for i in batch) == list(range(200))
    assert len(batches) == len(BucketBatchSampler(lengths, batch_size=8))

    def padding(groups):
        return sum(
            max(lengths[i] for i in g) * len(g) - sum(lengths[i] for i in g)
            for g in groups
        )

    shuffled = [list(range(i, min(i + 8, 200))) for i in range(0, 200, 8)]

    assert padding(batches) < padding(shuffled)


def test_bucketing_reshuffles_every_epoch():

    sampler = BucketBatchSampler(list(range(100)), batch_size=4)

    assert list(sampler) != list(sampler)


# ==================================================
# KV-cached generation
# ==================================================

def test_cached_generation_matches_uncached_generation():

    torch.manual_seed(0)

    parameters = dict(
        max_seq_len=24,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_kv_heads=2,
        window_size=4,
        dropout_param=0.0,
    )

    model = TransformerV2(
        TransformerConfigV2(
            encoder_config=EncoderConfigV2(vocab_size=TOKENIZER.vocab_size, **parameters),
            decoder_config=DecoderConfigV2(
                target_vocab_size=TOKENIZER.vocab_size, **parameters
            ),
        )
    ).eval()

    generator = object.__new__(Generator)
    generator.model = model
    generator.tokenizer = TOKENIZER
    generator.device = torch.device("cpu")

    source = torch.tensor([TOKENIZER.encode("abcabc"), TOKENIZER.encode("abc") + [0] * 3])

    cached = generator.generate(source, max_new_tokens=15, use_cache=True)
    uncached = generator.generate(source, max_new_tokens=15, use_cache=False)

    assert torch.equal(cached, uncached)
