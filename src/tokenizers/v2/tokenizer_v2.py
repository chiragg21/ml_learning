from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

"""
WordPiece tokenizer (v2) - trained on our own corpus, not pretrained.

How WordPiece differs from BPE (v1), in one paragraph: both start from
single characters and grow a vocabulary by merging neighbouring pieces,
but BPE always merges the most *frequent* pair, while WordPiece merges
the pair that most increases the likelihood of the training corpus
(roughly: pair count divided by the product of the two parts' counts),
so it prefers pairs whose parts rarely appear apart. The other visible
difference is at encoding time: BPE replays its merges in order, while
WordPiece greedily matches the *longest* vocabulary entry from the left
of each word, then continues with the remainder. Pieces that continue a
word are marked with a "##" prefix ("playing" -> "play", "##ing"), which
is how the decoder knows where to glue pieces back together. This is the
tokenizer BERT uses.

Unlike v1's byte-level BPE, WordPiece operates on unicode characters, not
bytes, so its base alphabet depends on the corpus and a character never
seen in training encodes to <UNK>. Decoding is also lossy about
whitespace: text is split into words and punctuation before matching, so
decode() re-joins them with single spaces and then cleans up spacing
around punctuation, rather than reproducing the input byte-for-byte.
"""


class WordPieceTokenizerV2:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    CONTINUING_SUBWORD_PREFIX = "##"

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

        tokenizer = Tokenizer(WordPiece(unk_token=self.UNK_TOKEN))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.decoder = WordPieceDecoder(
            prefix=self.CONTINUING_SUBWORD_PREFIX,
        )

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            continuing_subword_prefix=self.CONTINUING_SUBWORD_PREFIX,
        )

        # One training "document" per line, so pair statistics aren't
        # polluted by unrelated sentences that happened to sit next to
        # each other in the corpus string.
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
    tokenizer = WordPieceTokenizerV2(text, vocab_size=300)

    print(tokenizer.vocab_size)
    encoded = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    print(encoded)
    print(tokenizer.decode(encoded))
