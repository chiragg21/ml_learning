from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict
from pathlib import Path

from artifacts.manager import ExperimentManager
from src.data.factory import DataFactory
from src.data.seq2seq import TokenizedSeq2SeqDataset
from src.tokenizers.factory import TokenizerFactory
from src.tracking.mlflowlogger import MLflowLogger
from src.tracking.system import get_environment_info
from src.tracking.tracking import setup_mlflow
from src.training.trainer import Trainer


def load_config(config_path: str):

    path = Path(config_path)

    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.CONFIG


def build_tokenizer(config, experiment: ExperimentManager):

    if experiment.tokenizer_path.exists():
        return TokenizerFactory.load(experiment.tokenizer_path)

    _, _, corpus = DataFactory.build(config)

    tokenizer = TokenizerFactory.build(config, corpus)

    TokenizerFactory.save(
        tokenizer,
        experiment.tokenizer_path,
    )

    return tokenizer


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if this config's experiment already completed.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    experiment = ExperimentManager(config)

    if experiment.is_complete() and not args.force:

        print(
            f"Experiment {experiment.experiment_hash} already completed "
            f"at {experiment.experiment_dir} - skipping. Use --force to retrain."
        )

        return

    tokenizer = build_tokenizer(config, experiment)

    train_pairs, val_pairs, _ = DataFactory.build(config)

    train_dataset = TokenizedSeq2SeqDataset(train_pairs, tokenizer)

    validation_dataset = (
        TokenizedSeq2SeqDataset(val_pairs, tokenizer)
        if val_pairs
        else None
    )

    setup_mlflow(experiment_name=config.data.name)
    mlflow_logger = MLflowLogger(experiment_name=config.data.name)

    with mlflow_logger.start_run(run_name=experiment.experiment_hash):

        mlflow_logger.set_tags({
            key: str(value)
            for key, value in asdict(get_environment_info()).items()
        })

        trainer = Trainer(
            config,
            train_dataset,
            tokenizer,
            validation_dataset=validation_dataset,
            experiment_manager=experiment,
            mlflow_logger=mlflow_logger,
        )

        metrics = trainer.train()

    print(f"Training complete. Experiment hash: {experiment.experiment_hash}")
    print(metrics)


if __name__ == "__main__":
    main()
