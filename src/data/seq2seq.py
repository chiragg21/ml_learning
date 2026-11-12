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

    max_source_len/max_target_len (if given) drop any example whose
    *tokenized* length exceeds the model's max_seq_len, after tokenizing -
    this is the correctness guarantee a character-count pre-filter (e.g.
    the filter_by_length data-pipeline step) can't actually provide.
    Character count is not a safe proxy for token count with every
    tokenizer: byte-level BPE's base alphabet is UTF-8 *bytes*, not
    characters, so a multi-byte character (e.g. an accented French
    letter) can tokenize into more than one token if its byte pair
    wasn't merged by the trained vocabulary - meaning token count can
    exceed character count, which is exactly what crashed a real
    training run on this exact dataset/tokenizer combination.
    """

    def __init__(
        self,
        raw_pairs,
        tokenizer,
        max_source_len: int | None = None,
        max_target_len: int | None = None,
    ):

        self.examples = []
        dropped = 0

        for source_text, target_text in raw_pairs:

            source_ids = tokenizer.encode(source_text)

            target_ids = tokenizer.encode(
                target_text,
                add_bos=True,
                add_eos=True,
            )

            too_long = (
                (max_source_len is not None and len(source_ids) > max_source_len)
                or (max_target_len is not None and len(target_ids) > max_target_len)
            )

            if too_long:
                dropped += 1
                continue

            self.examples.append({
                "source_ids": torch.tensor(source_ids, dtype=torch.long),
                "target_ids": torch.tensor(target_ids, dtype=torch.long),
            })

        if dropped:
            print(
                f"TokenizedSeq2SeqDataset: dropped {dropped}/{len(raw_pairs)} "
                "pairs whose tokenized length exceeded max_source_len/max_target_len."
            )

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
