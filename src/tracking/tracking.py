import mlflow

# A local SQLite store needs no separately-running server - works the same
# on a home machine, Colab, or Kaggle without extra setup. Point this at
# an http(s):// URL instead if you do want a shared/running tracking server.
# (Not plain "file:./mlruns": newer MLflow versions put the filesystem
# store into maintenance mode and refuse to use it by default.)
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def setup_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = "mini-llm",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)