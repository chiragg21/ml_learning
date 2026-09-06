from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from artifacts.manager import ExperimentManager
from models.factory import ModelFactory
from evaluation.evaluator import Evaluator


class Predictor:

    def __init__(
        self,
        config,
        *,
        experiment_manager=None,
        device="auto",
    ):

        self.config = config

        # =============================================
        # Experiment
        # =============================================

        self.experiment = (
            experiment_manager
            or ExperimentManager(
                config,
                create_directories=False,
            )
        )

        # =============================================
        # Device
        # =============================================

        self.device = self._resolve_device(
            device
        )

        # =============================================
        # Model
        # =============================================

        self.model = ModelFactory.create(
            config
        )

        # =============================================
        # Load trained model
        # =============================================

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
    # Prediction
    # ==================================================

    @torch.no_grad()
    def predict(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ):

        source_ids = source_ids.to(
            self.device
        )

        target_ids = target_ids.to(
            self.device
        )

        return self.model(
            source_ids=source_ids,
            target_ids=target_ids,
        )

    # ==================================================
    # Evaluation
    # ==================================================

    def evaluate(
        self,
        dataset,
        batch_size=None,
    ):

        if batch_size is None:

            batch_size = (
                self.config.training.batch_size
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        evaluator = Evaluator(
            model=self.model,
            device=self.device,
        )

        metrics = evaluator.evaluate(
            dataloader
        )

        # ---------------------------------------------
        # Save evaluation results
        # ---------------------------------------------

        self.experiment.artifacts.save_json(
            self.experiment.evaluation_path(),
            metrics,
        )

        return metrics

    # ==================================================
    # Device
    # ==================================================

    @staticmethod
    def _resolve_device(device):

        if device == "auto":

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)