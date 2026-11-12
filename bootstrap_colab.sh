#!/usr/bin/env bash
# One-shot setup for a fresh Colab/Kaggle runtime: clone the repo (skipped
# if you're already inside it), install the handful of deps that aren't
# preinstalled, download the training data, train, then immediately
# export the trained model/tokenizer/mlflow logs into one zip - all in
# one script run, so there's no gap where the runtime could be
# changed/disconnected before anything is saved off the VM.
#
# This script itself CANNOT trigger the browser download: it runs as a
# shell subprocess (`!bash bootstrap_colab.sh`), and Colab's
# google.colab.files.download() only works when called directly from a
# real notebook code cell, in-kernel - not from any subprocess. See the
# printed instructions at the end for the one extra cell that does that.
#
# Usage (paste into a Colab/Kaggle code cell, or run from a terminal):
#   !bash bootstrap_colab.sh
#
# Override the config/dataset without editing this file:
#   !CONFIG=configs/toy_copy_v0.py bash bootstrap_colab.sh   # fast pipeline smoke test
#   !CONFIG=configs/wikitext_v0.py DATASET=wikitext bash bootstrap_colab.sh
#   !CONFIG=configs/translation_v0_scaled.py bash bootstrap_colab.sh   # 500k pairs, BPE tokenizer, d_model=384

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

pip install -q mlflow datasets tokenizers

python -m scripts.download_data --dataset "$DATASET"

python -m scripts.train --config "$CONFIG"

# Export immediately after training, in the same script run - waiting
# for a separate manual step here is exactly what risks losing the
# trained model if the runtime gets disconnected/changed in between.
python -m scripts.export_artifacts

echo ""
echo "Zip ready at: $(pwd)/export_bundle.zip"
echo "On Kaggle: it will already show up in the Output panel."
echo "On Colab: run this in a genuine code cell (not a !-prefixed shell"
echo "cell - files.download() only works called directly, in-kernel):"
echo ""
echo "  from google.colab import files"
echo "  files.download(\"$(pwd)/export_bundle.zip\")"
