from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import UnigramTrainer

"""
Unigram tokenizer (v3) - the algorithm behind SentencePiece, trained on
our own corpus, not pretrained.

How Unigram differs from BPE (v1) and WordPiece (v2), in one paragraph:
both of those *grow* a vocabulary from single characters by merging
pairs. Unigram goes the other way: it starts from a large candidate
vocabulary (all frequent substrings) and repeatedly *prunes* it, at each
round dropping the pieces whose removal hurts the corpus likelihood the
least, until the target size is reached. Each piece carries a learned
probability, and at encoding time the tokenizer picks the segmentation
of a word whose pieces have the highest total probability (found with
the Viterbi algorithm), rather than replaying merge rules or matching
greedily. So the same word is always split the single most likely way,
and the model can express several segmentations of a word, which is
what makes subword regularization possible.

Like SentencePiece, whitespace is treated as an ordinary symbol instead
of a separator: spaces are rewritten to "▁" before tokenizing, so a
piece that starts a word carries its own "▁" marker and decoding is just
concatenating pieces and turning "▁" back into a space. That makes
decode() reproduce the input text exactly (unlike WordPiece, which
re-joins words with single spaces), as long as every character was in
the vocabulary. Like WordPiece it works on unicode characters rather
than bytes, so a character never seen in training encodes to <UNK>.
"""


class UnigramTokenizerV3:

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

        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.decoder = MetaspaceDecoder()

        # Unlike BPE/WordPiece, the Unigram model is built from the
        # trainer's output, so the trainer (not the model) must be told
        # which special token stands for unknown pieces.
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.UNK_TOKEN,
        )

        # One training "document" per line, so substring statistics
        # aren't polluted by unrelated sentences that happened to sit
        # next to each other in the corpus string.
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
    tokenizer = UnigramTokenizerV3(text, vocab_size=100)

    print(tokenizer.vocab_size)
    encoded = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    print(encoded)
    print(tokenizer.decode(encoded))
