from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EXPERIMENTS_DIR = Path("artifacts/experiments")
MLFLOW_DB = Path("mlflow.db")


def _is_completed(experiment_dir: Path) -> bool:

    metadata_path = experiment_dir / "metadata.json"

    if not metadata_path.exists():
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return metadata.get("status") == "completed"


def _find_experiment_dir(experiment_hash: str | None) -> Path:

    if experiment_hash is not None:

        experiment_dir = EXPERIMENTS_DIR / experiment_hash

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"No experiment directory at {experiment_dir}"
            )

        return experiment_dir

    completed = [
        experiment_dir
        for experiment_dir in EXPERIMENTS_DIR.iterdir()
        if experiment_dir.is_dir() and _is_completed(experiment_dir)
    ]

    if not completed:
        raise FileNotFoundError(
            f"No completed experiments found under {EXPERIMENTS_DIR}. "
            "Pass --hash explicitly if you want to export an in-progress one."
        )

    return max(completed, key=lambda path: path.stat().st_mtime)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Bundle one experiment's final model/tokenizer/config "
            "(plus the MLflow log database) into a single zip, ready to "
            "download off Colab/Kaggle. Intermediate checkpoints are "
            "excluded - see CheckpointManager's retention and this "
            "script's own checkpoints/ exclusion."
        )
    )

    parser.add_argument(
        "--hash",
        default=None,
        help="Experiment hash to export (default: most recently completed one).",
    )

    parser.add_argument(
        "--output",
        default="export_bundle",
        help="Output zip path, without the .zip extension (default: export_bundle).",
    )

    args = parser.parse_args()

    experiment_dir = _find_experiment_dir(args.hash)

    staging_dir = Path(f"{args.output}_staging")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Skip checkpoints/: those hold model + optimizer state for
    # resuming training, not for using the trained model, and each one
    # is roughly 3x the model's size - bundling all of them turned an
    # export into gigabytes nobody needed once training completed.
    shutil.copytree(
        experiment_dir,
        staging_dir / "experiments" / experiment_dir.name,
        ignore=shutil.ignore_patterns("checkpoints"),
    )

    if MLFLOW_DB.exists():

        shutil.copy(
            MLFLOW_DB,
            staging_dir / MLFLOW_DB.name,
        )

    archive_path = shutil.make_archive(
        args.output,
        "zip",
        staging_dir,
    )

    shutil.rmtree(staging_dir)

    print(f"Exported experiment {experiment_dir.name} -> {archive_path}")


if __name__ == "__main__":
    main()
