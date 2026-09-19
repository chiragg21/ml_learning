from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.utils.hashing import get_config_hash


class ArtifactManager:
    """
    Responsible only for resolving and managing artifact locations.
    """

    def __init__(
        self,
        root_dir: str | Path = "artifacts",
    ):
        self.root_dir = Path(root_dir)

    # ==================================================
    # Experiment
    # ==================================================

    def experiment_dir(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.root_dir
            / "experiments"
            / experiment_hash
        )

    def create_experiment_dirs(
        self,
        experiment_hash: str,
    ) -> None:

        directories = [
            self.experiment_dir(experiment_hash),
            self.model_dir(experiment_hash),
            self.tokenizer_dir(experiment_hash),
            self.checkpoint_dir(experiment_hash),
            self.evaluation_dir(experiment_hash),
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ==================================================
    # Config
    # ==================================================

    def config_path(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "config.json"
        )

    # ==================================================
    # Model
    # ==================================================

    def model_dir(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "model"
        )

    def model_path(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.model_dir(experiment_hash)
            / "model.pt"
        )

    # ==================================================
    # Tokenizer
    # ==================================================

    def tokenizer_dir(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "tokenizer"
        )

    def tokenizer_path(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.tokenizer_dir(experiment_hash)
            / "tokenizer.pkl"
        )

    # ==================================================
    # Checkpoints
    # ==================================================

    def checkpoint_dir(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "checkpoints"
        )

    def checkpoint_path(
        self,
        experiment_hash: str,
        name: str,
    ) -> Path:

        return (
            self.checkpoint_dir(experiment_hash)
            / f"{name}.pt"
        )

    # ==================================================
    # Evaluation
    # ==================================================

    def evaluation_dir(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "evaluation"
        )

    def evaluation_path(
        self,
        experiment_hash: str,
        name: str = "results.json",
    ) -> Path:

        return (
            self.evaluation_dir(experiment_hash)
            / name
        )

    # ==================================================
    # Metadata
    # ==================================================

    def metadata_path(
        self,
        experiment_hash: str,
    ) -> Path:

        return (
            self.experiment_dir(experiment_hash)
            / "metadata.json"
        )

    # ==================================================
    # JSON utilities
    # ==================================================

    @staticmethod
    def save_json(
        path: str | Path,
        data: dict[str, Any],
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    @staticmethod
    def load_json(
        path: str | Path,
    ) -> dict[str, Any]:

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)


class ExperimentManager:
    """
    Represents one experiment.

    Responsible for:
        - experiment identity
        - configuration hash
        - artifact resolution
        - saving/loading experiment config
    """

    def __init__(
        self,
        config,
        *,
        artifact_manager: ArtifactManager | None = None,
        create_directories: bool = True,
    ):

        self.config = config

        # ---------------------------------------------
        # Experiment identity
        # ---------------------------------------------

        self.hash = get_config_hash(
            config
        )

        # ---------------------------------------------
        # Artifact manager
        # ---------------------------------------------

        self.artifacts = (
            artifact_manager
            or ArtifactManager()
        )

        if create_directories:

            self.artifacts.create_experiment_dirs(
                self.hash
            )

    # ==================================================
    # Experiment information
    # ==================================================

    @property
    def experiment_hash(self) -> str:

        return self.hash

    @property
    def experiment_dir(self) -> Path:

        return self.artifacts.experiment_dir(
            self.hash
        )

    # ==================================================
    # Artifact paths
    # ==================================================

    @property
    def config_path(self) -> Path:

        return self.artifacts.config_path(
            self.hash
        )

    @property
    def model_path(self) -> Path:

        return self.artifacts.model_path(
            self.hash
        )

    @property
    def tokenizer_dir(self) -> Path:

        return self.artifacts.tokenizer_dir(
            self.hash
        )

    @property
    def tokenizer_path(self) -> Path:

        return self.artifacts.tokenizer_path(
            self.hash
        )

    @property
    def checkpoint_dir(self) -> Path:

        return self.artifacts.checkpoint_dir(
            self.hash
        )

    @property
    def evaluation_dir(self) -> Path:

        return self.artifacts.evaluation_dir(
            self.hash
        )

    @property
    def metadata_path(self) -> Path:

        return self.artifacts.metadata_path(
            self.hash
        )

    # ==================================================
    # Checkpoint
    # ==================================================

    def checkpoint_path(
        self,
        name: str,
    ) -> Path:

        return self.artifacts.checkpoint_path(
            self.hash,
            name,
        )

    # ==================================================
    # Evaluation
    # ==================================================

    def evaluation_path(
        self,
        name: str = "results.json",
    ) -> Path:

        return self.artifacts.evaluation_path(
            self.hash,
            name,
        )

    # ==================================================
    # Config persistence
    # ==================================================

    def save_config(self) -> None:

        self.artifacts.save_json(
            self.config_path,
            asdict(self.config),
        )

    def load_saved_config(self) -> dict[str, Any]:

        return self.artifacts.load_json(
            self.config_path
        )

    # ==================================================
    # Completion status
    # ==================================================

    def is_complete(self) -> bool:

        if not self.metadata_path.exists():
            return False

        metadata = self.artifacts.load_json(
            self.metadata_path
        )

        return metadata.get("status") == "completed"

    def mark_complete(
        self,
        *,
        global_step: int,
        metrics: dict[str, Any],
    ) -> None:

        self.artifacts.save_json(
            self.metadata_path,
            {
                "status": "completed",
                "global_step": global_step,
                "metrics": metrics,
            },
        )