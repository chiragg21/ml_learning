#!/usr/bin/env bash
# One-shot setup for a fresh Colab/Kaggle runtime: clone the repo (skipped
# if you're already inside it), install the handful of deps that aren't
# preinstalled, download the training data, train, then immediately
# export and (on Colab) download the trained model/tokenizer/mlflow logs -
# all in one script run, so there's no gap where the runtime could be
# changed/disconnected before anything is saved off the VM.
#
# Usage (paste into a Colab/Kaggle code cell, or run from a terminal):
#   !bash bootstrap_colab.sh
#
# Override the config/dataset without editing this file:
#   !CONFIG=configs/toy_copy_v0.py bash bootstrap_colab.sh   # fast pipeline smoke test
#   !CONFIG=configs/wikitext_v0.py DATASET=wikitext bash bootstrap_colab.sh

set -euo pipefail

REPO_URL="https://github.com/chiragg21/ml_learning.git"
CONFIG="${CONFIG:-configs/translation_v0.py}"
DATASET="${DATASET:-translation}"

# If we're not already inside the repo (no pyproject.toml with this
# project's name), clone it fresh and cd in.
if [ ! -f "pyproject.toml" ] || ! grep -q 'name = "ml-learning"' pyproject.toml; then

    if [ ! -d "ml_learning" ]; then
        git clone "$REPO_URL" ml_learning
    fi

    cd ml_learning
fi

# torch is normally preinstalled with GPU support on Colab/Kaggle - avoid
# clobbering that build by only installing it if it's missing.
python -c "import torch" 2>/dev/null || pip install -q torch

pip install -q mlflow datasets

python -m scripts.download_data --dataset "$DATASET"

python -m scripts.train --config "$CONFIG"

# Export immediately after training, in the same script run - waiting
# for a separate manual step here is exactly what risks losing the
# trained model if the runtime gets disconnected/changed in between.
python -m scripts.export_artifacts

# On Colab, trigger an actual browser download of the exported zip so
# nothing further needs to be done by hand. On Kaggle (or anywhere else),
# the zip is just left in the working directory - Kaggle's own Output
# panel picks up files under /kaggle/working automatically.
python -c "
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download('export_bundle.zip')
"
