#!/usr/bin/env python3
"""
Step 3 — Compute emotion vectors with proper denoising and analyse the space.

For each emotion and each target layer, we:
    1. Compute per-emotion mean activations
    2. Compute the grand mean across all emotion means
    3. Fit PCA on neutral activations and project out the top components
       that explain 50% of variance (removes writing-style / topic confounds)
    4. Subtract the grand mean from each emotion mean (isolates what makes
       each emotion *distinctive from the average emotion*)
    5. Project out the "emotionality" direction (grand_mean - neutral_mean)
       to compensate for our valence-imbalanced 10-emotion set
    6. Normalise to unit length

Steps 3-5 follow the Anthropic methodology, with step 5 as an additional
correction for our small emotion set (10 vs. 171 in the paper).

Methodology references:
    Anthropic (2026) §2.3 — "We obtained emotion vectors by averaging these
    activations across stories corresponding to a given emotion, and
    subtracting off the mean activation across different emotions."

    Anthropic (2026) §2.3 — "We obtained model activations on a set of
    emotionally neutral transcripts and computed the top principal components
    … (enough to explain 50% of the variance). We then projected out these
    components from our emotion vectors."

Usage:
    python 03_compute_vectors.py
    python 03_compute_vectors.py --variance-threshold 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA

import config
from utils.visualization import plot_similarity_heatmap, plot_emotion_space


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute emotion vectors")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Layer indices to process (default: all in activations dir)")
    p.add_argument("--variance-threshold", type=float, default=0.5,
                    help="Fraction of neutral variance to project out (default: 0.5)")
    return p.parse_args()


def load_activations(layer_dir: Path, name: str) -> torch.Tensor:
    """Load a (num_stories, d_model) activation tensor."""
    path = layer_dir / f"{name}.pt"
    return torch.load(path, weights_only=True)


def fit_neutral_pca(neutral_acts: torch.Tensor, variance_threshold: float) -> np.ndarray:
    """Fit PCA on neutral activations, return components explaining `variance_threshold` of variance.

    Returns
    -------
    components : (n_components, d_model) array — the principal components to project out.
    """
    X = neutral_acts.numpy()
    # Fit full PCA to find how many components explain the threshold
    pca = PCA()
    pca.fit(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = min(n_components, len(cumulative))

    print(f"  PCA: {n_components} components explain {cumulative[n_components-1]:.1%} of neutral variance")
    return pca.components_[:n_components]  # (n_components, d_model)


def project_out(vectors: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Project out a set of orthonormal directions from each vector.

    Parameters
    ----------
    vectors : (n, d_model)
    directions : (k, d_model) — must be orthonormal (sklearn PCA components
        satisfy this; single unit vectors are trivially orthonormal).

    Returns
    -------
    projected : (n, d_model)
    """
    D = directions  # (k, d_model)
    projections = vectors @ D.T  # (n, k)
    return vectors - projections @ D


def normalise(vec: torch.Tensor) -> torch.Tensor:
    """Normalise to unit length, returning zero vector if near-zero."""
    norm = vec.norm()
    if norm < 1e-8:
        return torch.zeros_like(vec)
    return vec / norm


def cosine_similarity_matrix(vectors: torch.Tensor) -> np.ndarray:
    """Compute pairwise cosine similarity for a (num_vectors, d_model) matrix."""
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vectors / norms
    return (normed @ normed.T).numpy()


def expected_similarity(emotions: list[dict]) -> np.ndarray:
    """Compute expected pairwise similarity from valence/arousal coordinates."""
    coords = np.array([[e["valence"], e["arousal"]] for e in emotions])
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
    print(f"Neutral PCA variance threshold: {args.variance_threshold:.0%}")
    print()

    expected_sim = expected_similarity(emotions)

    for layer_idx in layer_indices:
        print(f"=== Layer {layer_idx} ===")
        layer_dir = config.ACTIVATIONS_DIR / str(layer_idx)
        vec_dir = config.VECTORS_DIR / str(layer_idx)
        vec_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Load all activations and compute per-emotion means ---
        neutral_acts = load_activations(layer_dir, "neutral")
        neutral_mean = neutral_acts.mean(dim=0)
        print(f"  Neutral: {neutral_acts.shape[0]} stories, dim={neutral_acts.shape[1]}")

        emotion_means = {}
        for emo in emotions:
            name = emo["name"]
            acts = load_activations(layer_dir, name)
            emotion_means[name] = acts.mean(dim=0)
            print(f"  {name}: {acts.shape[0]} stories")

        # --- Step 2: Compute grand mean across all emotion means ---
        # This is the mean-of-means (each emotion weighted equally),
        # not the mean of all individual stories.
        all_means = torch.stack(list(emotion_means.values()))  # (10, d_model)
        grand_mean = all_means.mean(dim=0)                     # (d_model,)
        print(f"  Grand mean computed across {len(emotion_means)} emotions")

        # --- Step 3: PCA confound removal from neutral data ---
        # Fit PCA on neutral activations, find components explaining 50% of
        # variance.  These capture writing-style, topic, and narrative structure
        # variance that is unrelated to emotion.
        pca_components = fit_neutral_pca(neutral_acts, args.variance_threshold)
        pca_directions = torch.tensor(pca_components, dtype=torch.float32)

        # --- Step 4: Subtract grand mean from each emotion mean ---
        # This isolates what makes each emotion distinctive from the average
        # emotion (the paper's approach), not from neutral.
        raw_vectors = {}
        for name in emotion_names:
            raw_vectors[name] = emotion_means[name] - grand_mean

        # --- Step 5: Project out neutral PCA components ---
        raw_matrix = torch.stack([raw_vectors[n] for n in emotion_names])  # (10, d_model)
        denoised = project_out(raw_matrix, pca_directions)

        # --- Step 6: Project out the "emotionality" direction ---
        # With only 10 emotions (7 negative-valence), the grand mean is skewed.
        # The direction (grand_mean - neutral_mean) captures shared "being
        # emotional" signal.  Projecting it out removes this bias.
        # We first project the emotionality direction through the same PCA
        # filter so it lives in the same subspace as our denoised vectors.
        emotionality_dir = grand_mean - neutral_mean
        emotionality_dir = project_out(emotionality_dir.unsqueeze(0), pca_directions).squeeze(0)
        emotionality_norm = emotionality_dir.norm()
        if emotionality_norm > 1e-8:
            emotionality_unit = (emotionality_dir / emotionality_norm).unsqueeze(0)  # (1, d_model)
            denoised = project_out(denoised, emotionality_unit)
            print(f"  Projected out emotionality direction (norm={emotionality_norm:.2f})")

        # --- Step 7: Normalise to unit length and save ---
        vectors = []
        for i, name in enumerate(emotion_names):
            vec = normalise(denoised[i])
            torch.save(vec, vec_dir / f"{name}.pt")
            vectors.append(vec)
            print(f"  {name}: final vector (norm check: {vec.norm():.4f})")

        all_vectors = torch.stack(vectors)  # (num_emotions, d_model)
        torch.save(all_vectors, vec_dir / "all_vectors.pt")

        # --- Analysis ---
        sim_matrix = cosine_similarity_matrix(all_vectors)

        plot_similarity_heatmap(
            sim_matrix,
            emotion_names,
            config.OUTPUTS_DIR / f"similarity_matrix_layer{layer_idx}.png",
            title=f"Emotion Vector Cosine Similarity — Layer {layer_idx}",
        )
        print(f"  Saved similarity heatmap")

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
