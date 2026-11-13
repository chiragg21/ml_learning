from __future__ import annotations

import math

import torch
from torch import nn


class Evaluator:

    def __init__(
        self,
        model,
        device: str | torch.device = "cpu",
        pad_token_id: int = 0,
        mixed_precision: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):

        self.model = model
        self.device = torch.device(device)
        self.pad_token_id = pad_token_id
        self.mixed_precision = mixed_precision
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ) -> dict[str, float]:

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        for batch in dataloader:

            source_ids = batch[
                "source_ids"
            ].to(self.device)

            target_ids = batch[
                "target_ids"
            ].to(self.device)

            # Teacher forcing: decoder sees all but the last target token,
            # and is scored against all but the first (the shifted labels).
            decoder_input_ids = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            source_key_padding_mask = (
                source_ids == self.pad_token_id
            )

            target_key_padding_mask = (
                decoder_input_ids == self.pad_token_id
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.mixed_precision,
            ):

                output = self.model(
                    source_ids=source_ids,
                    target_ids=decoder_input_ids,
                    source_key_padding_mask=source_key_padding_mask,
                    target_key_padding_mask=target_key_padding_mask,
                )

            # Loss in fp32 whatever the forward pass ran in.
            logits = output.logits.float()

            vocab_size = logits.size(-1)

            token_loss = (
                nn.functional.cross_entropy(
                    logits.reshape(
                        -1,
                        vocab_size,
                    ),
                    labels.reshape(-1),
                    ignore_index=self.pad_token_id,
                    reduction="sum",
                )
            )

            predictions = logits.argmax(
                dim=-1
            )

            mask = (
                labels
                != self.pad_token_id
            )

            correct_tokens += (
                (
                    predictions == labels
                )
                & mask
            ).sum().item()

            tokens = mask.sum().item()

            total_loss += token_loss.item()

            total_tokens += tokens

        if total_tokens == 0:

            raise RuntimeError(
                "No valid tokens found during evaluation."
            )

        loss = (
            total_loss / total_tokens
        )

        perplexity = math.exp(
            min(loss, 20)
        )

        accuracy = (
            correct_tokens
            / total_tokens
        )

        return {
            "loss": loss,
            "perplexity": perplexity,
            "token_accuracy": accuracy,
        }