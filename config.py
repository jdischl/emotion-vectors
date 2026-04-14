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
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # ungated mirror
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Layer targeting
# ---------------------------------------------------------------------------
# We sample the residual stream at 25%, 50%, and 75% depth.
# For Llama 3.1 8B (32 layers, all global attention): layers 8, 16, 24.
TARGET_LAYER_PERCENTAGES = [0.25, 0.50, 0.75]

# Models with hybrid attention (e.g. Gemma 4) place global attention at
# every Nth layer; we snap targets to these layers.  For models with
# uniform global attention (Llama, Mistral), stride=1 means no snapping.
_HYBRID_ATTENTION_STRIDES = {
    "gemma4": 6,
}


def get_text_config(model_or_config):
    """Return the text component config for any model.

    Gemma 4 is a multimodal model; its text-tower config lives at
    ``model.config.text_config``.  All other models expose attributes
    directly on ``model.config``.  This helper normalises both cases.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", cfg)


def get_decoder_layers(model):
    """Return the transformer decoder layer list for any model.

    Tries common HuggingFace model structures in order:
      1. model.model.layers          — standard causal LM (Llama, Gemma 1-3, ...)
      2. model.language_model.model.layers — multimodal models (PaliGemma, Gemma 4, ...)
      3. model.model.language_model.model.layers — another multimodal variant
      4. model.transformer.h         — GPT-2 style

    Raises a descriptive AttributeError if none match, listing the model's
    top-level attributes so the caller can diagnose the right path.
    """
    candidates = [
        lambda m: m.model.layers,                        # standard causal LM
        lambda m: m.model.language_model.layers,         # Gemma4ForConditionalGeneration
        lambda m: m.language_model.model.layers,         # PaliGemma / LLaVA
        lambda m: m.model.language_model.model.layers,   # other multimodal variants
        lambda m: m.model.text_model.layers,             # BLIP-style
        lambda m: m.transformer.h,                       # GPT-2
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    top_attrs = [a for a in dir(model) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot locate transformer decoder layers for {type(model).__name__}. "
        f"Top-level attributes: {top_attrs}"
    )


def get_target_layers(model) -> list[int]:
    """Return layer indices for activation capture at 25%, 50%, 75% depth.

    For models with hybrid attention (e.g. Gemma 4), snaps to the nearest
    global-attention layer.  For models with uniform attention (Llama,
    Mistral), returns the raw percentage positions directly.
    """
    text_cfg = get_text_config(model)
    if hasattr(text_cfg, "num_hidden_layers"):
        n_layers = text_cfg.num_hidden_layers
    else:
        n_layers = len(get_decoder_layers(model))

    # Determine attention stride: hybrid models have stride > 1
    model_type = getattr(text_cfg, "model_type", "")
    stride = _HYBRID_ATTENTION_STRIDES.get(model_type, 1)

    if stride > 1:
        # Snap to global attention layers (e.g. Gemma 4: every 6th layer)
        global_layers = [i for i in range(n_layers) if i % stride == stride - 1]
        targets = []
        for pct in TARGET_LAYER_PERCENTAGES:
            raw = int(n_layers * pct)
            best = min(global_layers, key=lambda gl: abs(gl - raw))
            targets.append(best)
    else:
        # All layers are equivalent — use raw percentages
        targets = [int(n_layers * pct) for pct in TARGET_LAYER_PERCENTAGES]

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
