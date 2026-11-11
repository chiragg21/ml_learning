from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenizedSeq2SeqDataset(Dataset):
    """
    Wraps a raw (source_text, target_text) pair dataset, tokenizing both
    sides with the experiment's tokenizer. Works with any DataFactory entry
    that yields text pairs and any tokenizer registered in TokenizerFactory.

    Tokenizes everything once up front rather than per __getitem__ call -
    with DataLoader re-iterating the same dataset every epoch, encoding on
    the fly would otherwise re-tokenize every sample on every epoch.
    """

    def __init__(self, raw_pairs, tokenizer):

        self.examples = [
            {
                "source_ids": torch.tensor(
                    tokenizer.encode(source_text),
                    dtype=torch.long,
                ),
                "target_ids": torch.tensor(
                    tokenizer.encode(
                        target_text,
                        add_bos=True,
                        add_eos=True,
                    ),
                    dtype=torch.long,
                ),
            }
            for source_text, target_text in raw_pairs
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def make_collate_fn(pad_token_id: int):
    """
    Returns a collate_fn that right-pads variable-length source/target
    sequences within a batch, keyed to the given tokenizer's pad token id.
    """

    def collate(batch):

        source_ids = pad_sequence(
            [item["source_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        target_ids = pad_sequence(
            [item["target_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
        }

    return collate
