#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod with a /workspace network volume.
# Idempotent: safe to re-run on the same pod or on a new pod reusing the volume.
#
# Usage:
#   # On the pod (after ssh -A root@<pod>):
#   curl -fsSL https://raw.githubusercontent.com/jdischl/emotion-vectors/main/bootstrap.sh | bash
#   # or, if repo already cloned:
#   bash /workspace/emotion-vectors/bootstrap.sh
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:jdischl/emotion-vectors.git}"
REPO_DIR="${REPO_DIR:-/workspace/emotion-vectors}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"

echo "=== emotion-vectors bootstrap ==="
echo "repo:      $REPO_DIR"
echo "hf cache:  $HF_CACHE"
echo

# 1. Env vars (export now + persist to ~/.bashrc)
export HF_HUB_CACHE="$HF_CACHE"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_CACHE"
for line in "export HF_HUB_CACHE=$HF_CACHE" "export HF_HUB_DISABLE_XET=1"; do
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done

# 2. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "[bootstrap] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc \
        || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# 3. Clone or pull repo (uses SSH agent forwarding — no key on pod)
if [ -d "$REPO_DIR/.git" ]; then
    echo "[bootstrap] repo exists — pulling"
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[bootstrap] cloning repo to $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 4. Install Python deps
echo "[bootstrap] uv sync"
uv sync

# 5. Hugging Face auth (for gated Llama weights)
if ! uv run huggingface-cli whoami &>/dev/null; then
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "[bootstrap] logging into HF via \$HF_TOKEN"
        uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "[bootstrap] no HF session — run 'uv run huggingface-cli login' or set HF_TOKEN"
    fi
fi

echo
echo "=== bootstrap complete ==="
echo "next:"
echo "  cd $REPO_DIR"
echo "  tmux new -s chat"
echo "  uv run python 06_chat_interface.py --share"
