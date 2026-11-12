from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

"""
Byte-level BPE tokenizer (v1) - trained on our own corpus, not pretrained.

How BPE works, in one paragraph: start with every individual symbol as
its own token, then repeatedly find the most frequent *adjacent pair* of
tokens across the training corpus and merge it into one new token.
Repeat until the vocabulary reaches the target size. The result: common
whole words ("the", "and") end up as single tokens, common word-pieces
("-ing", "-tion") end up as single tokens, and anything never seen
before still encodes fine by falling back toward smaller, more common
pieces - nothing is ever truly "unknown".

Why *byte-level* BPE specifically, rather than running BPE directly over
unicode characters (which is what v0's CharTokenizer uses as its base
unit): CharTokenizer's vocabulary is exactly however many distinct
characters appear in whatever corpus built it - 341 for the OPUS-100
en-fr data, inflated by rare unicode symbols. Byte-level BPE instead
starts from UTF-8's 256 possible byte values as its fixed base alphabet,
so the base vocabulary never depends on what's in the corpus, and any
input text whatsoever - any language, emoji, typo - can always be
losslessly represented by falling back to raw bytes if nothing larger
matches. This is the same scheme GPT-2/GPT-3/RoBERTa use.

"Trained", not "pretrained": the actual merge rules below are learned
from *this* dataset's own corpus, at a vocabulary size we choose -
keeping the embedding table sized for a small from-scratch model,
rather than importing a 30k+ vocabulary sized for someone else's much
bigger pretrained model.
"""


class BPETokenizerV1:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        text: str,
        vocab_size: int = 8000,
    ):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        self._tokenizer = self._train(text, vocab_size)

    def _train(self, text: str, vocab_size: int) -> Tokenizer:

        tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
        )

        # One training "document" per line, so BPE's pair-frequency
        # counting doesn't merge tokens across unrelated sentence
        # boundaries that happened to sit next to each other in the
        # corpus string.
        lines = [line for line in text.split("\n") if line]

        tokenizer.train_from_iterator(lines, trainer)

        return tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.token_to_id(self.UNK_TOKEN)

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self.BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self.EOS_TOKEN)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:

        token_ids = self._tokenizer.encode(text).ids

        if add_bos:
            token_ids = [self.bos_token_id] + token_ids

        if add_eos:
            token_ids = token_ids + [self.eos_token_id]

        return token_ids

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )


if __name__ == "__main__":
    text = "hello world\nbonjour le monde\nhello again, world"
    tokenizer = BPETokenizerV1(text, vocab_size=300)

    print(tokenizer.vocab_size)
    encoded = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    print(encoded)
    print(tokenizer.decode(encoded))
