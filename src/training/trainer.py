from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from artifacts.manager import ExperimentManager
from models.factory import ModelFactory
from training.checkpoint import CheckpointManager


class Trainer:

    def __init__(
        self,
        config,
        train_dataset,
        validation_dataset=None,
        *,
        experiment_manager=None,
        mlflow_logger=None,
    ):

        self.config = config

        # =============================================
        # Experiment
        # =============================================

        self.experiment = (
            experiment_manager
            or ExperimentManager(config)
        )

        self.experiment.save_config()

        # =============================================
        # Runtime
        # =============================================

        self.device = self._resolve_device()

        self._set_seed(
            config.runtime.seed
        )

        # =============================================
        # Model
        # =============================================

        self.model = ModelFactory.create(
            config
        )

        self.model.to(
            self.device
        )

        # =============================================
        # Data
        # =============================================

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.runtime.num_workers,
        )

        self.validation_loader = None

        if validation_dataset is not None:

            self.validation_loader = DataLoader(
                validation_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.runtime.num_workers,
            )

        # =============================================
        # Training
        # =============================================

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        self.optimizer = (
            self._create_optimizer()
        )

        self.scheduler = (
            self._create_scheduler()
        )

        self.checkpoint_manager = (
            CheckpointManager()
        )

        self.global_step = 0

        # =============================================
        # MLflow
        # =============================================

        self.mlflow_logger = mlflow_logger