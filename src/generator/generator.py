from __future__ import annotations

import torch

from src.inference.base import LoadedExperiment


class Generator(LoadedExperiment):

    # ==================================================
    # Generation
    # ==================================================

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:

        source_ids = source_ids.to(
            self.device
        )

        batch_size = source_ids.size(0)

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        source_key_padding_mask = (
            source_ids == pad_token_id
        )

        target_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            output = self.model(
                source_ids=source_ids,
                target_ids=target_ids,
                source_key_padding_mask=source_key_padding_mask,
            )

            next_token_logits = (
                output.logits[:, -1, :]
            )

            next_token_logits = (
                next_token_logits
                / temperature
            )

            next_token = (
                next_token_logits
                .argmax(dim=-1)
                .unsqueeze(-1)
            )

            target_ids = torch.cat(
                [
                    target_ids,
                    next_token,
                ],
                dim=1,
            )

            finished = (
                next_token.squeeze(-1)
                == eos_token_id
            )

            if finished.all():
                break

        return target_ids

    # ==================================================
    # Text in, text out
    # ==================================================

    def generate_text(
        self,
        text: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> str:

        source_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            dtype=torch.long,
        )

        target_ids = self.generate(
            source_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return self.tokenizer.decode(target_ids[0])
