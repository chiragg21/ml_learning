from __future__ import annotations

import torch

from artifacts.manager import ExperimentManager
from models.factory import ModelFactory


class Generator:

    def __init__(
        self,
        config,
        *,
        experiment_manager=None,
        device="auto",
    ):

        self.config = config

        self.experiment = (
            experiment_manager
            or ExperimentManager(
                config,
                create_directories=False,
            )
        )

        self.device = self._resolve_device(
            device
        )

        # =============================================
        # Model
        # =============================================

        self.model = ModelFactory.create(
            config
        )

        if not self.experiment.model_path.exists():

            raise FileNotFoundError(
                "Trained model not found at: "
                f"{self.experiment.model_path}"
            )

        self.model.load_model(
            self.experiment.model_path,
            device=self.device,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    # ==================================================
    # Generation
    # ==================================================

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ):

        source_ids = source_ids.to(
            self.device
        )

        batch_size = source_ids.size(0)

        target_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            logits = self.model(
                source_ids=source_ids,
                target_ids=target_ids,
            )

            next_token_logits = (
                logits[:, -1, :]
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

    @staticmethod
    def _resolve_device(device):

        if device == "auto":

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)