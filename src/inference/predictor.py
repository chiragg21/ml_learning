from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.seq2seq import make_collate_fn
from src.evaluation.evaluator import Evaluator
from src.inference.base import LoadedExperiment


class Predictor(LoadedExperiment):

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
            collate_fn=make_collate_fn(
                self.tokenizer.pad_token_id
            ),
        )

        evaluator = Evaluator(
            model=self.model,
            device=self.device,
            pad_token_id=self.tokenizer.pad_token_id,
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
