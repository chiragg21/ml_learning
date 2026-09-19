from __future__ import annotations

import torch

from artifacts.manager import ExperimentManager
from src.models.factory import ModelFactory
from src.tokenizers.factory import TokenizerFactory


class LoadedExperiment:
    """
    Loads a completed experiment's saved tokenizer and trained model weights
    from its config. Shared base for Predictor and Generator so the
    load-experiment/build-model/load-weights sequence isn't duplicated
    across inference entry points.
    """

    def __init__(
        self,
        config,
        *,
        experiment_manager=None,
        device: str | torch.device = "auto",
    ):

        self.config = config

        self.experiment = (
            experiment_manager
            or ExperimentManager(
                config,
                create_directories=False,
            )
        )

        self.device = self._resolve_device(device)

        if not self.experiment.tokenizer_path.exists():

            raise FileNotFoundError(
                "Trained tokenizer not found at: "
                f"{self.experiment.tokenizer_path}"
            )

        self.tokenizer = TokenizerFactory.load(
            self.experiment.tokenizer_path
        )

        self.model = ModelFactory.create(
            config,
            tokenizer=self.tokenizer,
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

        self.model.to(self.device)

        self.model.eval()

    @staticmethod
    def _resolve_device(device):

        if device == "auto":

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)
