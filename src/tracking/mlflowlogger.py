from __future__ import annotations

from typing import Any

import mlflow

from src.utils.hashing import config_to_dict


def _flatten_params(
    prefix: str,
    value: Any,
    out: dict[str, str],
) -> None:

    if isinstance(value, dict):

        for key, val in value.items():

            _flatten_params(
                f"{prefix}.{key}" if prefix else str(key),
                val,
                out,
            )

        return

    out[prefix] = str(value)


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
        """
        Flattens the whole config into dot-separated mlflow params, so new
        config fields (new component types, new hyperparameters, ...) show
        up automatically without touching this method.
        """

        flat_params: dict[str, str] = {}

        _flatten_params(
            "",
            config_to_dict(config),
            flat_params,
        )

        flat_params["config_hash"] = experiment_hash

        self.log_params(flat_params)