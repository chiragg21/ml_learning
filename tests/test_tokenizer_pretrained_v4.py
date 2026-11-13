"""
Lite smoke test for the pretrained tokenizer (v4). Real pretrained
tokenizers come from the Hugging Face Hub, so these tests save a tiny
byte-level BPE to a local tokenizer.json and load it back through the
`source` path option - same code path, no network needed. End-to-end
training with this tokenizer is covered in test_smoke_training.py.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.tokenizers.factory import TokenizerFactory
from src.tokenizers.v4.tokenizer_v4 import PretrainedTokenizerV4


CORPUS = [
    "the cat sat on the mat",
    "the dog played in the garden",
    "cats and dogs are playing, again and again.",
] * 20


@pytest.fixture
def tokenizer_file(tmp_path) -> str:

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(CORPUS, trainer)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    return str(path)


def test_pretrained_v4_missing_special_tokens_are_appended(tokenizer_file):

    original_size = Tokenizer.from_file(tokenizer_file).get_vocab_size()

    tokenizer = PretrainedTokenizerV4("", source=tokenizer_file)

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    }

    assert None not in special_ids
    assert len(special_ids) == 4
    assert tokenizer.vocab_size == original_size + 4


def test_pretrained_v4_encode_decode_round_trip(tokenizer_file):

    tokenizer = PretrainedTokenizerV4("", source=tokenizer_file)

    token_ids = tokenizer.encode(
        "the dog played, again.",
        add_bos=True,
        add_eos=True,
    )

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids) == "the dog played, again."


def test_pretrained_v4_can_reuse_an_existing_token(tokenizer_file):

    original_size = Tokenizer.from_file(tokenizer_file).get_vocab_size()

    tokenizer = PretrainedTokenizerV4(
        "",
        source=tokenizer_file,
        eos_token="<|endoftext|>",
    )

    assert tokenizer.eos_token_id == 0
    assert tokenizer.vocab_size == original_size + 3


def test_pretrained_v4_unknown_requested_token_raises(tokenizer_file):

    with pytest.raises(ValueError):
        PretrainedTokenizerV4(
            "",
            source=tokenizer_file,
            eos_token="<not-a-token>",
        )


def test_pretrained_v4_survives_factory_save_and_load(tokenizer_file, tmp_path):

    tokenizer = PretrainedTokenizerV4("", source=tokenizer_file)

    path = tmp_path / "saved" / "tokenizer.pkl"

    TokenizerFactory.save(tokenizer, path)
    restored = TokenizerFactory.load(path)

    text = "cats and dogs"

    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.pad_token_id == tokenizer.pad_token_id
