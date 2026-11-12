"""
Lite smoke test for the Unigram tokenizer (v3): trains on a tiny corpus
and checks the interface the rest of the pipeline relies on (special token
ids, BOS/EOS handling, encode/decode round trip). End-to-end training with
this tokenizer is covered in test_smoke_training.py.
"""

from __future__ import annotations

from src.tokenizers.factory import TokenizerFactory
from src.tokenizers.v3.tokenizer_v3 import UnigramTokenizerV3


CORPUS = "\n".join(
    [
        "the cat sat on the mat",
        "the dog played in the garden",
        "cats and dogs are playing, again and again.",
    ]
    * 20
)


def test_unigram_v3_special_tokens_are_distinct():

    tokenizer = UnigramTokenizerV3(CORPUS, vocab_size=100)

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    }

    assert None not in special_ids
    assert len(special_ids) == 4


def test_unigram_v3_encode_decode_round_trip():

    tokenizer = UnigramTokenizerV3(CORPUS, vocab_size=100)

    token_ids = tokenizer.encode(
        "the dog played, again.",
        add_bos=True,
        add_eos=True,
    )

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids) == "the dog played, again."


def test_unigram_v3_unseen_character_maps_to_unk():

    tokenizer = UnigramTokenizerV3(CORPUS, vocab_size=100)

    assert tokenizer.unk_token_id in tokenizer.encode("the 世")


def test_unigram_v3_is_registered_and_picklable(tmp_path):

    tokenizer = UnigramTokenizerV3(CORPUS, vocab_size=100)

    path = tmp_path / "tokenizer.pkl"
    TokenizerFactory.save(tokenizer, path)
    restored = TokenizerFactory.load(path)

    assert TokenizerFactory._registry["unigram_v3"] is UnigramTokenizerV3
    assert restored.encode("the cat") == tokenizer.encode("the cat")
