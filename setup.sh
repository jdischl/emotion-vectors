#!/usr/bin/env bash
# Setup script for RunPod GPU instance
set -euo pipefail

echo "=== Emotion Vectors — RunPod Setup ==="

# Install dependencies
echo "[1/3] Installing Python dependencies..."
pip install torch transformers accelerate scikit-learn numpy matplotlib seaborn tqdm jsonlines umap-learn

# HuggingFace login (needed for gated models)
echo "[2/3] HuggingFace authentication..."
if huggingface-cli whoami &>/dev/null; then
    echo "Already logged in to HuggingFace."
else
    echo "Please enter your HuggingFace token (needs access to google/gemma-4-31B-it):"
    huggingface-cli login
fi

# Create data directories
echo "[3/3] Creating data directories..."
mkdir -p data/{stories,activations,vectors} outputs

echo ""
echo "=== Setup complete ==="
echo "Run the pipeline:"
echo "  python 01_generate_stories.py"
echo "  python 02_extract_activations.py"
echo "  python 03_compute_vectors.py"
echo "  python 04_validate_probes.py"
echo "  python 05_steer_and_eval.py"
