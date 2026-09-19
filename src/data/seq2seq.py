from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenizedSeq2SeqDataset(Dataset):
    """
    Wraps a raw (source_text, target_text) pair dataset, tokenizing both
    sides with the experiment's tokenizer. Works with any DataFactory entry
    that yields text pairs and any tokenizer registered in TokenizerFactory.
    """

    def __init__(self, raw_pairs, tokenizer):
        self.raw_pairs = raw_pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.raw_pairs)

    def __getitem__(self, index):

        source_text, target_text = self.raw_pairs[index]

        source_ids = self.tokenizer.encode(source_text)

        target_ids = self.tokenizer.encode(
            target_text,
            add_bos=True,
            add_eos=True,
        )

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


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
