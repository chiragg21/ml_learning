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
        do_sample: bool = False,
        top_k: int | None = None,
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
                / temperature
            )

            if do_sample:

                if top_k is not None:

                    # Zero out everything outside the top_k logits before
                    # sampling, so temperature can make output more varied
                    # without the model ever picking a near-random,
                    # garbage-low-probability character.
                    top_values, top_indices = next_token_logits.topk(
                        top_k, dim=-1
                    )

                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    ).scatter_(-1, top_indices, top_values)

                probabilities = torch.softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(
                    probabilities, num_samples=1
                )

            else:

                # argmax is unaffected by dividing by temperature (it
                # never changes which logit is largest) - greedy decoding
                # is deterministic regardless of temperature, by design.
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
        do_sample: bool = False,
        top_k: int | None = None,
    ) -> str:

        source_ids = torch.tensor(
            [self.tokenizer.encode(text)],
            dtype=torch.long,
        )

        target_ids = self.generate(
            source_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
        )

        return self.tokenizer.decode(target_ids[0])
