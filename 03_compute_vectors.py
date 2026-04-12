#!/usr/bin/env python3
"""
Step 3 — Compute mean-difference emotion vectors and analyse the emotion space.

For each emotion and each target layer, we compute:
    emotion_vector = mean(emotion_activations) - mean(neutral_activations)
then normalise to unit length.  This "mean-difference" approach is the
simplest way to find a linear direction in activation space that separates
one concept from baseline.

We then analyse the resulting vector space:
    - Pairwise cosine similarity matrix between all emotion vectors
    - PCA projection coloured by valence
    - Correlation with expected valence/arousal distances

Methodology reference:
    Anthropic (2026) §2.3 — "We computed emotion directions as the
    difference between the mean activation for passages about that
    emotion and the mean activation over neutral passages."

Usage:
    python 03_compute_vectors.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import config
from utils.visualization import plot_similarity_heatmap, plot_emotion_space


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute emotion vectors")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Layer indices to process (default: all in activations dir)")
    return p.parse_args()


def load_activations(layer_dir: Path, name: str) -> torch.Tensor:
    """Load a (num_stories, d_model) activation tensor."""
    path = layer_dir / f"{name}.pt"
    return torch.load(path, weights_only=True)


def compute_emotion_vector(
    emotion_acts: torch.Tensor,
    neutral_mean: torch.Tensor,
) -> torch.Tensor:
    """Compute a unit-length mean-difference vector.

    If the difference is near-zero (emotion mean ≈ neutral mean), returns
    a zero vector rather than producing NaN from division by zero.
    """
    vec = emotion_acts.mean(dim=0) - neutral_mean
    norm = vec.norm()
    if norm < 1e-8:
        return torch.zeros_like(vec)
    return vec / norm


def cosine_similarity_matrix(vectors: torch.Tensor) -> np.ndarray:
    """Compute pairwise cosine similarity for a (num_vectors, d_model) matrix."""
    # Normalise rows to unit length
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vectors / norms
    return (normed @ normed.T).numpy()


def expected_similarity(emotions: list[dict]) -> np.ndarray:
    """Compute expected pairwise similarity from valence/arousal coordinates.

    Two emotions close in (valence, arousal) space should have high
    cosine similarity if the model's representations mirror human
    psychological dimensions.
    """
    coords = np.array([[e["valence"], e["arousal"]] for e in emotions])
    # Euclidean distance → convert to similarity (1 - normalised distance)
    from scipy.spatial.distance import squareform, pdist
    dists = squareform(pdist(coords, metric="euclidean"))
    max_dist = dists.max() if dists.max() > 0 else 1.0
    return 1.0 - dists / max_dist


def main():
    args = parse_args()
    config.ensure_dirs()
    emotions = config.load_emotions()
    emotion_names = [e["name"] for e in emotions]

    # Discover available layers
    if args.layers:
        layer_indices = args.layers
    else:
        layer_indices = sorted(
            int(d.name) for d in config.ACTIVATIONS_DIR.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    if not layer_indices:
        print("ERROR: No activation directories found. Run 02_extract_activations.py first.")
        return

    print(f"Processing layers: {layer_indices}")
    print(f"Emotions: {emotion_names}")
    print()

    # Compute expected similarity from valence/arousal for correlation analysis
    expected_sim = expected_similarity(emotions)

    for layer_idx in layer_indices:
        print(f"=== Layer {layer_idx} ===")
        layer_dir = config.ACTIVATIONS_DIR / str(layer_idx)
        vec_dir = config.VECTORS_DIR / str(layer_idx)
        vec_dir.mkdir(parents=True, exist_ok=True)

        # Load neutral baseline
        neutral_acts = load_activations(layer_dir, "neutral")
        neutral_mean = neutral_acts.mean(dim=0)
        print(f"  Neutral: {neutral_acts.shape[0]} stories, dim={neutral_acts.shape[1]}")

        # Compute emotion vectors
        vectors = []
        for emo in emotions:
            name = emo["name"]
            acts = load_activations(layer_dir, name)
            vec = compute_emotion_vector(acts, neutral_mean)
            torch.save(vec, vec_dir / f"{name}.pt")
            vectors.append(vec)
            print(f"  {name}: {acts.shape[0]} stories → unit vector (norm check: {vec.norm():.4f})")

        # Stack into matrix and save
        all_vectors = torch.stack(vectors)  # (num_emotions, d_model)
        torch.save(all_vectors, vec_dir / "all_vectors.pt")

        # Cosine similarity matrix
        sim_matrix = cosine_similarity_matrix(all_vectors)

        plot_similarity_heatmap(
            sim_matrix,
            emotion_names,
            config.OUTPUTS_DIR / f"similarity_matrix_layer{layer_idx}.png",
            title=f"Emotion Vector Cosine Similarity — Layer {layer_idx}",
        )
        print(f"  Saved similarity heatmap")

        # PCA/UMAP of emotion space
        valences = [e["valence"] for e in emotions]
        plot_emotion_space(
            all_vectors.numpy(),
            emotion_names,
            valences,
            config.OUTPUTS_DIR / f"emotion_space_layer{layer_idx}.png",
            title=f"Emotion Vector Space (PCA) — Layer {layer_idx}",
        )
        print(f"  Saved PCA plot")

        # Correlation with expected valence/arousal similarity
        # Use upper-triangle values only (avoid diagonal and symmetry)
        tri_idx = np.triu_indices(len(emotions), k=1)
        observed_pairs = sim_matrix[tri_idx]
        expected_pairs = expected_sim[tri_idx]
        r, p = stats.pearsonr(observed_pairs, expected_pairs)
        print(f"  Valence/arousal correlation: r={r:.3f}, p={p:.4f}")

        # Top-5 most similar and dissimilar pairs
        pairs = []
        for i in range(len(emotions)):
            for j in range(i + 1, len(emotions)):
                pairs.append((emotion_names[i], emotion_names[j], sim_matrix[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n  Top-5 most similar pairs:")
        for a, b, s in pairs[:5]:
            print(f"    {a:>12} — {b:<12}  cos={s:+.3f}")

        print(f"  Top-5 most dissimilar pairs:")
        for a, b, s in pairs[-5:]:
            print(f"    {a:>12} — {b:<12}  cos={s:+.3f}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
