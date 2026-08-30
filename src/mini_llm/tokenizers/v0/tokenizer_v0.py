from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

class CharTokenizer:
    """
    Simple character level tokenizer.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
            self,
            text: str,
        ):
        self.vocab = self._build_vocabulary(text)
        self.special_tokens = []

    def _build_vocabulary(
            self, 
            text: str,
        )-> Vocabulary:

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]

        characters = sorted(set(text))

        tokens = self.special_tokens + characters

        token_to_id = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        id_to_token = {
            idx: token
            for token, idx in token_to_id.items() 
        }

        return Vocabulary(
            token_to_id=token_to_id,
            id_to_token=id_to_token
        )

    @property
    def vocab_size(self)->int:
        return len(self.vocab.token_to_id)

    @property
    def pad_token_id(self)->int:
        return self.vocab.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self)->int:
            return self.vocab.token_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self)->int:
                return self.vocab.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self)->int:
            return self.vocab.token_to_id[self.EOS_TOKEN]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
          
        token_ids = []

        if add_bos:
              token_ids.append(self.bos_token_id)

        token_ids.extend(
              self.vocab.token_to_id.get(
                    character,
                    self.unk_token_id,
              )
              for character in text
        )

        if add_eos:
              token_ids.append(self.eos_token_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True
    )->str:

        if isinstance(token_ids, torch.Tensor):
              token_ids = token_ids.tolist()

        tokens = []

        for token_id in token_ids:
              token = self.vocab.id_to_token[token_id]

              if skip_special_tokens and token in self.special_tokens:
                continue

              tokens.append(token)

        return "".join(tokens)


if __name__ == "__main__":
     text = "Hello World!"
     tokenizer = CharTokenizer(text)

     print(tokenizer.vocab_size, '\n', tokenizer.vocab, '\n', tokenizer.encode(text+' z'))
     