#!/usr/bin/env python3
"""
Step 4 — Train linear probes to classify emotions from residual-stream activations.

If emotion concepts are linearly represented in the model's activation space,
a simple logistic regression should be able to classify which emotion a story
depicts from its activation vector alone.  This is the standard "linear probe"
validation used across interpretability research.

We train:
    1. A multinomial classifier (all emotions + neutral)
    2. Per-emotion binary classifiers (emotion-vs-rest) whose weight vectors
       are probe-derived emotion directions — we compare these to the
       mean-difference vectors from step 3.

Methodology reference:
    Anthropic (2026) §3.1 — "We validated the emotion directions by training
    linear probes … achieving >90% classification accuracy at middle layers."

Usage:
    python 04_validate_probes.py
    python 04_validate_probes.py --layers 17 29 47
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config
from utils.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear probes on emotion activations")
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(layer_dir: Path, emotions: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load activations and labels for all emotions + neutral.

    Returns (X, y_encoded, label_names) where X is (n_samples, d_model)
    and y_encoded is integer-encoded labels.
    """
    X_parts = []
    y_parts = []
    label_names = [e["name"] for e in emotions] + ["neutral"]

    for name in label_names:
        path = layer_dir / f"{name}.pt"
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping")
            continue
        acts = torch.load(path, weights_only=True).numpy()
        X_parts.append(acts)
        y_parts.extend([name] * acts.shape[0])

    X = np.concatenate(X_parts, axis=0)
    le = LabelEncoder()
    le.fit(label_names)
    y = le.transform(y_parts)
    return X, y, list(le.classes_)


def train_multinomial_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    layer_idx: int,
):
    """Train a multinomial logistic regression and report metrics."""
    print(f"\n  --- Multinomial Probe (Layer {layer_idx}) ---")
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Macro F1:  {macro_f1:.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, label_names,
        config.OUTPUTS_DIR / f"confusion_matrix_layer{layer_idx}.png",
        title=f"Emotion Classification — Layer {layer_idx} (acc={acc:.2f})",
    )
    print(f"  Saved confusion matrix plot")

    return clf, acc, macro_f1


def train_binary_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    layer_idx: int,
    emotion_names: list[str],
) -> dict:
    """Train per-emotion binary probes and compare with mean-diff vectors.

    Returns {emotion: {"accuracy": float, "f1": float, "cosine_sim": float}}.
    """
    print(f"\n  --- Binary Probes (Layer {layer_idx}) ---")

    vec_dir = config.VECTORS_DIR / str(layer_idx)
    results = {}

    for emo_name in emotion_names:
        if emo_name not in label_names:
            continue
        emo_idx = label_names.index(emo_name)

        # Binary labels: 1 = this emotion, 0 = everything else
        y_train_bin = (y_train == emo_idx).astype(int)
        y_test_bin = (y_test == emo_idx).astype(int)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train_bin)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test_bin, y_pred)
        f1 = f1_score(y_test_bin, y_pred, zero_division=0)

        # Compare probe weight vector with mean-diff vector
        probe_vec = torch.tensor(clf.coef_[0], dtype=torch.float32)
        probe_vec = probe_vec / probe_vec.norm()

        cosine_sim = float("nan")
        mean_diff_path = vec_dir / f"{emo_name}.pt"
        if mean_diff_path.exists():
            mean_diff_vec = torch.load(mean_diff_path, weights_only=True).float()
            cosine_sim = float(torch.dot(probe_vec, mean_diff_vec))

        results[emo_name] = {"accuracy": acc, "f1": f1, "cosine_sim": cosine_sim}
        print(f"    {emo_name:>12}: acc={acc:.3f}  F1={f1:.3f}  cos(probe, mean-diff)={cosine_sim:+.3f}")

    return results


def main():
    args = parse_args()
    config.ensure_dirs()
    emotions = config.load_emotions()
    emotion_names = [e["name"] for e in emotions]

    # Discover layers
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

    print(f"Layers: {layer_indices}")
    print(f"Emotions: {emotion_names}")
    print()

    best_layer = None
    best_f1 = -1
    all_results = {}

    for layer_idx in layer_indices:
        print(f"{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        layer_dir = config.ACTIVATIONS_DIR / str(layer_idx)
        X, y, label_names = load_dataset(layer_dir, emotions)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

        clf, acc, macro_f1 = train_multinomial_probe(
            X_train, y_train, X_test, y_test, label_names, layer_idx
        )

        binary_results = train_binary_probes(
            X_train, y_train, X_test, y_test, label_names, layer_idx, emotion_names
        )

        all_results[layer_idx] = {
            "multinomial_accuracy": acc,
            "multinomial_macro_f1": macro_f1,
            "binary_probes": binary_results,
        }

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_layer = layer_idx

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for idx in layer_indices:
        r = all_results[idx]
        print(f"  Layer {idx}: accuracy={r['multinomial_accuracy']:.3f}, macro-F1={r['multinomial_macro_f1']:.3f}")
    print(f"\n  Best layer: {best_layer} (macro-F1={best_f1:.3f})")
    print(f"  → Use this layer for steering in step 05")

    # Save results
    results_path = config.OUTPUTS_DIR / "probe_results.json"
    serialisable = {
        "best_layer": best_layer,
        "best_macro_f1": best_f1,
        "per_layer": {
            str(k): {
                "multinomial_accuracy": v["multinomial_accuracy"],
                "multinomial_macro_f1": v["multinomial_macro_f1"],
                "binary_probes": v["binary_probes"],
            }
            for k, v in all_results.items()
        },
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"  Saved results → {results_path}")


if __name__ == "__main__":
    main()
