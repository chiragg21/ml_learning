"""
Lite smoke tests: run a full config -> data -> tokenizer -> model -> train
pass end-to-end on tiny synthetic data, on CPU, in a few seconds. These are
NOT testing model quality - they exist to catch integration bugs (shape
mismatches, dtype errors, config wiring) immediately, before they surface
hours into a real training run on borrowed GPU time.

Pattern for new versions: add a toy config under configs/ (copy an
existing toy_copy_*.py and bump name/version to your new model/tokenizer),
then add one function here that trains it for a couple of steps and
asserts the loss is finite. See VERSIONS.md and CLAUDE.md for the
version-by-version workflow this supports.
"""

from __future__ import annotations

import math

from artifacts.manager import ArtifactManager, ExperimentManager
from scripts.train import build_tokenizer, load_config
from src.data.factory import DataFactory
from src.data.seq2seq import TokenizedSeq2SeqDataset
from src.training.trainer import Trainer


def _run_smoke_training(config_path: str, tmp_path) -> dict:

    config = load_config(config_path)

    artifact_manager = ArtifactManager(root_dir=tmp_path / "artifacts")
    experiment = ExperimentManager(config, artifact_manager=artifact_manager)

    tokenizer = build_tokenizer(config, experiment)

    train_pairs, val_pairs, _ = DataFactory.build(config)

    max_source_len = config.model.encoder.parameters.get("max_seq_len")
    max_target_len = config.model.decoder.parameters.get("max_seq_len")

    train_dataset = TokenizedSeq2SeqDataset(
        train_pairs,
        tokenizer,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
    )

    validation_dataset = (
        TokenizedSeq2SeqDataset(
            val_pairs,
            tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )
        if val_pairs
        else None
    )

    trainer = Trainer(
        config,
        train_dataset,
        tokenizer,
        validation_dataset=validation_dataset,
        experiment_manager=experiment,
    )

    return trainer.train()


def test_toy_copy_v0_trains_end_to_end(tmp_path):

    metrics = _run_smoke_training("configs/toy_copy_v0.py", tmp_path)

    assert math.isfinite(metrics["train_loss"])


def test_toy_copy_v1_trains_end_to_end(tmp_path):

    metrics = _run_smoke_training("configs/toy_copy_v1.py", tmp_path)

    assert math.isfinite(metrics["train_loss"])


def test_toy_copy_wordpiece_v2_trains_end_to_end(tmp_path):

    metrics = _run_smoke_training("configs/toy_copy_wordpiece_v2.py", tmp_path)

    assert math.isfinite(metrics["train_loss"])


def test_toy_copy_v2_trains_end_to_end(tmp_path):

    metrics = _run_smoke_training("configs/toy_copy_v2.py", tmp_path)

    assert math.isfinite(metrics["train_loss"])
