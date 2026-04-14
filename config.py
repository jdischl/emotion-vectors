"""
Central configuration for the emotion vectors project.

All hyperparameters, model settings, and path conventions live here so that
individual pipeline scripts stay focused on logic rather than magic numbers.
"""

import json
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STORIES_DIR = DATA_DIR / "stories"
ACTIVATIONS_DIR = DATA_DIR / "activations"
VECTORS_DIR = DATA_DIR / "vectors"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMOTIONS_PATH = PROJECT_ROOT / "emotions.json"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-31B-it"
FALLBACK_MODEL_ID = "google/gemma-4-E4B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Layer targeting
# ---------------------------------------------------------------------------
# We sample the residual stream at 25%, 50%, and 75% depth.
# For Gemma 4's hybrid attention, we snap to the nearest *global* attention
# layer (full attention every 6 layers, at indices where idx % 6 == 5) so
# that captured activations reflect the full context window.
TARGET_LAYER_PERCENTAGES = [0.25, 0.50, 0.75]

# Gemma 4 places global (full) attention at every 6th layer.  The pattern
# for a 60-layer model is: 5, 11, 17, 23, 29, 35, 41, 47, 53, 59.
_GLOBAL_ATTENTION_STRIDE = 6


def get_text_config(model_or_config):
    """Return the text component config for any model.

    Gemma 4 is a multimodal model; its text-tower config lives at
    ``model.config.text_config``.  All other models expose attributes
    directly on ``model.config``.  This helper normalises both cases.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", cfg)


def get_target_layers(model) -> list[int]:
    """Return layer indices for activation capture, snapped to global attention layers.

    The function reads num_hidden_layers from the model config, computes the
    raw 25%/50%/75% positions, and rounds each to the nearest layer whose
    index satisfies  idx % 6 == 5  (the global-attention positions in Gemma 4).
    """
    text_cfg = get_text_config(model)
    if hasattr(text_cfg, "num_hidden_layers"):
        n_layers = text_cfg.num_hidden_layers
    else:
        n_layers = len(model.model.layers)

    # Build set of global attention layer indices
    global_layers = [i for i in range(n_layers) if i % _GLOBAL_ATTENTION_STRIDE == _GLOBAL_ATTENTION_STRIDE - 1]

    targets = []
    for pct in TARGET_LAYER_PERCENTAGES:
        raw = int(n_layers * pct)
        # Snap to nearest global attention layer
        best = min(global_layers, key=lambda gl: abs(gl - raw))
        targets.append(best)

    # Deduplicate while preserving order (possible for very small models)
    seen = set()
    unique = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Story generation
# ---------------------------------------------------------------------------
NUM_STORIES_PER_EMOTION = 50
NUM_NEUTRAL_STORIES = 100  # 2x per-emotion count
BATCH_SIZE = 8
GENERATION_TEMPERATURE = 0.9
GENERATION_TOP_P = 0.95
GENERATION_MAX_NEW_TOKENS = 256

# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
STEERING_ALPHAS = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_emotions() -> list[dict]:
    """Load the emotion vocabulary from emotions.json."""
    with open(EMOTIONS_PATH) as f:
        return json.load(f)


def ensure_dirs():
    """Create all required data and output directories."""
    for d in [STORIES_DIR, ACTIVATIONS_DIR, VECTORS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
