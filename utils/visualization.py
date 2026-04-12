"""
Plotting helpers for the emotion-vectors pipeline.

All functions save to disk and return the matplotlib Figure so callers can
display or close as needed.  We use a consistent style: white background,
no unnecessary grid lines, and colourblind-friendly palettes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str],
    save_path: str | Path,
    title: str = "Emotion Vector Cosine Similarity",
) -> plt.Figure:
    """Plot a square cosine-similarity heatmap and save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_emotion_space(
    vectors: np.ndarray,
    labels: list[str],
    valences: list[float],
    save_path: str | Path,
    title: str = "Emotion Vector Space (PCA)",
    use_umap: bool = False,
) -> plt.Figure:
    """Project emotion vectors to 2-D via PCA (or UMAP) and scatter-plot, coloured by valence."""
    from sklearn.decomposition import PCA

    if use_umap:
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(labels) - 1))
            method_name = "UMAP"
        except ImportError:
            reducer = PCA(n_components=2)
            method_name = "PCA"
    else:
        reducer = PCA(n_components=2)
        method_name = "PCA"

    coords = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=valences,
        cmap="RdYlGn",
        s=120,
        edgecolors="black",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Valence (negative → positive)")
    ax.set_xlabel(f"{method_name} dimension 1")
    ax.set_ylabel(f"{method_name} dimension 2")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    save_path: str | Path,
    title: str = "Emotion Classification — Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix and save as PNG."""
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_dose_response(
    alphas: list[float],
    mean_ratings: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Steering Dose-Response",
) -> plt.Figure:
    """Plot target-emotion intensity as a function of steering strength.

    Parameters
    ----------
    alphas : list of steering strengths
    mean_ratings : {emotion_name: [mean_rating_at_alpha_0, ...]}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for emotion, ratings in mean_ratings.items():
        ax.plot(alphas, ratings, marker="o", label=emotion)
    ax.set_xlabel("Steering strength (α)")
    ax.set_ylabel("Mean target-emotion rating (1–10)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
