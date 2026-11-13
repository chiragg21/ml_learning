from __future__ import annotations

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, Sampler, get_worker_info


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

    @property
    def lengths(self) -> list[int]:
        return [
            len(example["source_ids"]) + len(example["target_ids"])
            for example in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ==================================================
# Sequence packing
# ==================================================

class PackedSeq2SeqDataset(Dataset):
    """
    Packs several short examples into each row, so a batch carries far
    fewer <PAD> tokens: a row holds as many consecutive examples as fit
    within max_source_len on the source side AND max_target_len on the
    target side (the row ends when either budget would overflow).

    Each row also carries segment ids - 1, 2, ... for the examples in it,
    0 for padding - which the model turns into a block-diagonal attention
    mask so examples cannot see each other, and which the trainer uses to
    ignore the one label per boundary that would otherwise ask the model
    to predict the next example's <BOS>.

    Packing is static: rows are formed once, in dataset order, and
    DataLoader shuffles whole rows. Re-packing every epoch would mix
    examples more, at the cost of doing this work each time.
    """

    def __init__(
        self,
        dataset: TokenizedSeq2SeqDataset,
        max_source_len: int | None,
        max_target_len: int | None,
    ):

        if max_source_len is None or max_target_len is None:
            raise ValueError(
                "Packing needs max_source_len and max_target_len "
                "(the encoder/decoder max_seq_len)."
            )

        self.rows = []

        current: list[dict] = []
        source_used = 0
        target_used = 0

        for example in dataset.examples:

            source_len = len(example["source_ids"])
            target_len = len(example["target_ids"])

            overflows = (
                source_used + source_len > max_source_len
                or target_used + target_len > max_target_len
            )

            if current and overflows:
                self.rows.append(self._pack(current))
                current, source_used, target_used = [], 0, 0

            current.append(example)
            source_used += source_len
            target_used += target_len

        if current:
            self.rows.append(self._pack(current))

        self.num_examples = len(dataset)

        print(
            f"PackedSeq2SeqDataset: packed {self.num_examples} examples "
            f"into {len(self.rows)} rows.",
            flush=True,
        )

    @staticmethod
    def _pack(examples: list[dict]) -> dict:

        def segments(key: str) -> torch.Tensor:
            return torch.cat([
                torch.full((len(example[key]),), index + 1, dtype=torch.long)
                for index, example in enumerate(examples)
            ])

        return {
            "source_ids": torch.cat([e["source_ids"] for e in examples]),
            "target_ids": torch.cat([e["target_ids"] for e in examples]),
            "source_segment_ids": segments("source_ids"),
            "target_segment_ids": segments("target_ids"),
        }

    @property
    def lengths(self) -> list[int]:
        return [
            len(row["source_ids"]) + len(row["target_ids"])
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


# ==================================================
# Streaming
# ==================================================

class StreamingSeq2SeqDataset(IterableDataset):
    """
    Tokenizes lazily, as the DataLoader asks for examples, instead of
    encoding the whole corpus up front: start-up is instant and only the
    examples in flight sit in memory as tensors. Each DataLoader worker
    takes every num_workers-th pair, so workers never repeat work, and a
    small shuffle buffer (examples are drawn at random from the last
    `shuffle_buffer` seen) stands in for a full shuffle, which a stream
    can't do.

    Pairs that tokenize past max_source_len/max_target_len are skipped,
    so the true length can be smaller than len(); len() is the pair
    count, which only feeds batch-count estimates (progress logging and
    the LR schedule's total steps).
    """

    def __init__(
        self,
        raw_pairs,
        tokenizer,
        max_source_len: int | None = None,
        max_target_len: int | None = None,
        shuffle_buffer: int = 1000,
        seed: int = 0,
    ):

        self.raw_pairs = raw_pairs
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def _encode(self, pair) -> dict | None:

        source_text, target_text = pair

        source_ids = self.tokenizer.encode(source_text)

        target_ids = self.tokenizer.encode(
            target_text,
            add_bos=True,
            add_eos=True,
        )

        if (
            (self.max_source_len is not None and len(source_ids) > self.max_source_len)
            or (self.max_target_len is not None and len(target_ids) > self.max_target_len)
        ):
            return None

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    def __iter__(self):

        worker = get_worker_info()
        worker_id, num_workers = (
            (worker.id, worker.num_workers) if worker is not None else (0, 1)
        )

        # Different order every epoch, identical across workers' shares.
        rng = random.Random(f"{self.seed}-{self.epoch}-{worker_id}")
        self.epoch += 1

        buffer: list[dict] = []

        for pair in self.raw_pairs[worker_id::num_workers]:

            example = self._encode(pair)

            if example is None:
                continue

            if len(buffer) < self.shuffle_buffer:
                buffer.append(example)
                continue

            slot = rng.randrange(len(buffer))
            yield buffer[slot]
            buffer[slot] = example

        rng.shuffle(buffer)

        yield from buffer

    def __len__(self) -> int:
        return len(self.raw_pairs)


# ==================================================
# Length bucketing
# ==================================================

class BucketBatchSampler(Sampler):
    """
    Batches examples of similar length together, so each batch pads to
    roughly the length of all its members instead of its one longest
    outlier. Pure shuffling would make that padding large; sorting
    everything would make every epoch identical. The compromise:
    shuffle, cut into chunks of batch_size * chunk_factor examples, sort
    each chunk by length, slice it into batches, then shuffle the batches.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        chunk_factor: int = 50,
        seed: int = 0,
    ):

        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = batch_size * chunk_factor
        self.seed = seed
        self.epoch = 0

    def __iter__(self):

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng.shuffle(indices)

        batches = []

        for start in range(0, len(indices), self.chunk_size):

            chunk = sorted(
                indices[start : start + self.chunk_size],
                key=lambda index: self.lengths[index],
            )

            batches.extend(
                chunk[i : i + self.batch_size]
                for i in range(0, len(chunk), self.batch_size)
            )

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size)


class Seq2SeqCollator:
    """
    Right-pads variable-length source/target sequences within a batch,
    keyed to the given tokenizer's pad token id.

    A plain class rather than make_collate_fn's old closure-returning
    function: DataLoader(num_workers>0) ships the collate_fn to each
    worker process, and a nested function can't be pickled for that -
    Python 3.14 made this fail loudly by switching POSIX's default
    multiprocessing start method from fork (no pickling needed, just
    inherits memory) to forkserver (pickles everything), where the old
    closure raised PicklingError on every multi-worker run.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):

        source_ids = pad_sequence(
            [item["source_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        target_ids = pad_sequence(
            [item["target_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        collated = {
            "source_ids": source_ids,
            "target_ids": target_ids,
        }

        # Present only for packed rows; 0 marks padding.
        if "source_segment_ids" in batch[0]:

            for key in ("source_segment_ids", "target_segment_ids"):

                collated[key] = pad_sequence(
                    [item[key] for item in batch],
                    batch_first=True,
                    padding_value=0,
                )

        return collated


def make_collate_fn(pad_token_id: int) -> Seq2SeqCollator:
    return Seq2SeqCollator(pad_token_id)


def build_datasets(config, tokenizer, train_pairs, val_pairs):
    """
    Tokenizes the raw pairs into (train, validation) datasets, applying
    the data-pipeline options in config.data.extra:

      streaming: bool        - tokenize the training set lazily
      shuffle_buffer: int    - streaming shuffle buffer size (default 1000)
      pack_sequences: bool   - pack short examples into shared rows

    Validation is never streamed or packed, so its metrics stay
    comparable across settings. Streaming and packing are exclusive
    (packing needs to see all lengths up front).
    """

    options = config.data.extra

    max_source_len = config.model.encoder.parameters.get("max_seq_len")
    max_target_len = config.model.decoder.parameters.get("max_seq_len")

    if options.get("streaming") and options.get("pack_sequences"):
        raise ValueError("data.extra: streaming and pack_sequences are exclusive.")

    if options.get("streaming"):

        train_dataset = StreamingSeq2SeqDataset(
            train_pairs,
            tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            shuffle_buffer=options.get("shuffle_buffer", 1000),
            seed=config.runtime.seed,
        )

    else:

        train_dataset = TokenizedSeq2SeqDataset(
            train_pairs,
            tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )

        if options.get("pack_sequences"):
            train_dataset = PackedSeq2SeqDataset(
                train_dataset,
                max_source_len,
                max_target_len,
            )

    validation_dataset = (
        TokenizedSeq2SeqDataset(
            val_pairs,
            tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )
        if val_pairs
        else None
    )

    return train_dataset, validation_dataset
