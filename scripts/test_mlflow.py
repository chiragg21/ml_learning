import mlflow
from mini_llm.tracking.tracking import setup_mlflow

def main():
    setup_mlflow()

    with mlflow.start_run(run_name="mlflow setup test"):
        