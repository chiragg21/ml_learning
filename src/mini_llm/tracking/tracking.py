import mlflow


def setup_mlflow(
    tracking_uri: str = "http://127.0.0.1:5000",
    experiment_name: str = "mini-llm",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)