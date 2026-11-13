from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

"""
Pretrained tokenizer (v4) - loads a tokenizer someone else already
trained (GPT-2's byte-level BPE, BERT's WordPiece, a SentencePiece
Unigram, ...) instead of training one on our corpus.

Why use one: a vocabulary learned from a huge, varied corpus segments
real text far better than one fitted to a small training set, and it is
what any model initialised from or compared against a public checkpoint
has to use. The `corpus` the factory passes in is ignored, because
nothing is trained here.

`source` is either a Hugging Face Hub id (e.g. "gpt2", downloaded and
cached on first use, so it needs network access) or a path to a local
`tokenizer.json`. Whatever algorithm the file describes, its own
normalizer, pre-tokenizer, model and decoder are used unchanged.

The rest of the pipeline expects <PAD>/<UNK>/<BOS>/<EOS> ids, which
pretrained vocabularies don't reliably have (GPT-2 has only
<|endoftext|>; byte-level BPEs have no unknown token at all). By
default any of the four that are missing are appended to the vocabulary
as new ids, so `vocab_size` can end up slightly larger than the
original. Pass `pad_token`/`unk_token`/`bos_token`/`eos_token` to reuse
an existing token instead (e.g. `eos_token="<|endoftext|>"`). Encoding
never adds the pretrained post-processor's own markers (BERT's
[CLS]/[SEP]), so BOS/EOS are only ever added by `add_bos`/`add_eos`.
"""


class PretrainedTokenizerV4:

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        text: str,
        source: str,
        pad_token: str | None = None,
        unk_token: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
    ):
        self.source = source

        self._tokenizer = self._load(source)

        self._pad_token = self._resolve(pad_token, self.PAD_TOKEN)
        self._unk_token = self._resolve(unk_token, self.UNK_TOKEN)
        self._bos_token = self._resolve(bos_token, self.BOS_TOKEN)
        self._eos_token = self._resolve(eos_token, self.EOS_TOKEN)

        # Kept as a list for parity with the trained tokenizers.
        self.special_tokens = [
            self._pad_token,
            self._unk_token,
            self._bos_token,
            self._eos_token,
        ]

    @staticmethod
    def _load(source: str) -> Tokenizer:

        if Path(source).is_file():
            return Tokenizer.from_file(source)

        return Tokenizer.from_pretrained(source)

    def _resolve(
        self,
        requested: str | None,
        default: str,
    ) -> str:

        token = requested or default

        if self._tokenizer.token_to_id(token) is not None:
            return token

        # An explicitly requested token that isn't in the vocabulary is
        # a typo in the config, not something to silently create.
        if requested is not None:
            raise ValueError(
                f"Token '{requested}' is not in the vocabulary of '{self.source}'."
            )

        self._tokenizer.add_special_tokens([token])

        return token

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._pad_token)

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._unk_token)

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._bos_token)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id(self._eos_token)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:

        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids

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

        # Only the four pipeline ids count as special here. Reused
        # tokens like <|endoftext|> are in `special_tokens` too, so they
        # are dropped, which is what generation wants.
        if skip_special_tokens:
            skipped = set(self._special_ids())
            token_ids = [t for t in token_ids if t not in skipped]

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )

    def _special_ids(self) -> list[int]:

        return [
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ]
