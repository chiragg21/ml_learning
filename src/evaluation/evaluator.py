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
    ):

        self.model = model
        self.device = torch.device(device)
        self.pad_token_id = pad_token_id

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

            logits = self.model(
                source_ids=source_ids,
                target_ids=target_ids,
            )

            vocab_size = logits.size(-1)

            token_loss = (
                nn.functional.cross_entropy(
                    logits.reshape(
                        -1,
                        vocab_size,
                    ),
                    target_ids.reshape(-1),
                    ignore_index=self.pad_token_id,
                    reduction="sum",
                )
            )

            predictions = logits.argmax(
                dim=-1
            )

            mask = (
                target_ids
                != self.pad_token_id
            )

            correct_tokens += (
                (
                    predictions == target_ids
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