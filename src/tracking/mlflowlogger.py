from __future__ import annotations

from typing import Any

import mlflow


class MLflowLogger:

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ):

        if tracking_uri is not None:
            mlflow.set_tracking_uri(
                tracking_uri
            )

        mlflow.set_experiment(
            experiment_name
        )

        self.run = None

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def start_run(
        self,
        run_name: str | None = None,
    ):

        self.run = mlflow.start_run(
            run_name=run_name
        )

        return self.run

    def end_run(self):

        mlflow.end_run()

        self.run = None

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    def log_params(
        self,
        params: dict[str, Any],
    ):

        mlflow.log_params(params)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ):

        mlflow.log_metrics(
            metrics,
            step=step,
        )

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ):

        mlflow.log_metric(
            name,
            value,
            step=step,
        )

    # --------------------------------------------------
    # Artifacts
    # --------------------------------------------------

    def log_artifact(
        self,
        path: str,
        artifact_path: str | None = None,
    ):

        mlflow.log_artifact(
            path,
            artifact_path=artifact_path,
        )

    def log_artifacts(
        self,
        directory: str,
        artifact_path: str | None = None,
    ):

        mlflow.log_artifacts(
            directory,
            artifact_path=artifact_path,
        )

    # --------------------------------------------------
    # Tags
    # --------------------------------------------------

    def set_tags(
        self,
        tags: dict[str, str],
    ):

        mlflow.set_tags(tags)
        

    def log_experiment_config(
        self,
        config,
        experiment_hash: str,
    ):

        self.log_params({
            "schema_version":
                config.schema_version,

            "transformer":
                config.model.transformer.name,

            "transformer_version":
                config.model.transformer.version,

            "encoder_version":
                config.model.encoder.version,

            "decoder_version":
                config.model.decoder.version,

            "tokenizer":
                config.tokenizer.name,

            "tokenizer_version":
                config.tokenizer.version,

            "data":
                config.data.name,

            "data_version":
                config.data.version,

            "config_hash":
                experiment_hash,
        })