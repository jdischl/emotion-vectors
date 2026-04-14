#!/usr/bin/env python3
"""
Step 2 — Extract residual-stream activations for each story.

For every story generated in step 1, we run a forward pass through Gemma 4
and capture the residual-stream hidden states at three target layers (25%,
50%, 75% depth, snapped to global-attention layers).  We average the
activation across all token positions from position 50 onward (skipping
early tokens where the model hasn't yet processed enough emotional context).

Note on tokenisation:
    Stories are fed as raw text (no chat template) even though they were
    generated with one.  This is intentional — we want activations reflecting
    the model's representation of the *emotional content*, not the chat
    framing.  Chat template tokens (system markers, turn delimiters) would
    add structural noise that dilutes the emotion signal.

Methodology reference:
    Anthropic (2026) §2.2 — "We extracted residual stream activations at
    each layer, averaging across all token positions within each story,
    beginning with the 50th token."

Usage:
    python 02_extract_activations.py
    python 02_extract_activations.py --batch-size 4     # reduce if OOM
    python 02_extract_activations.py --layers 17 29 47  # override auto-detect
"""

import argparse
import time
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from utils.hooks import ActivationCapture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract residual-stream activations")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Override auto-detected layer indices")
    return p.parse_args()


def load_stories(path: Path) -> list[str]:
    """Load story texts from a JSONL file."""
    with jsonlines.open(path) as reader:
        return [entry["story"] for entry in reader]


# Token position from which to start averaging activations.  Earlier tokens
# haven't processed enough emotional context to be informative.
_AVERAGING_START_TOKEN = 50


def extract_mean_activations(
    activations: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Average activations across token positions from position 50 onward.

    For each story in the batch, we compute the mean hidden state over all
    real (non-padding) tokens at positions >= 50.  This captures the
    emotional content spread across the full story rather than relying on
    a single position.

    Also computes the mean *norm* of activations at each layer (used later
    to normalise steering strength, per Anthropic §4).

    Parameters
    ----------
    activations : {layer_idx: tensor of shape (batch, seq_len, d_model)}
    attention_mask : tensor of shape (batch, seq_len), 1 for real tokens

    Returns
    -------
    mean_acts : {layer_idx: tensor of shape (batch, d_model)}
    mean_norms : {layer_idx: tensor of shape (batch,)}
    """
    batch_size, seq_len = attention_mask.shape

    # Build a mask that is 1 only for real tokens at positions >= 50
    position_mask = torch.zeros_like(attention_mask)
    if seq_len > _AVERAGING_START_TOKEN:
        position_mask[:, _AVERAGING_START_TOKEN:] = attention_mask[:, _AVERAGING_START_TOKEN:]
    else:
        # Story is shorter than 50 tokens — fall back to all real tokens
        position_mask = attention_mask

    # Count of tokens per sequence for averaging  (batch,)
    token_counts = position_mask.sum(dim=1).clamp(min=1)

    mean_acts = {}
    mean_norms = {}
    for layer_idx, acts in activations.items():
        # acts: (batch, seq_len, d_model), mask: (batch, seq_len)
        masked = acts * position_mask.unsqueeze(-1)          # zero out padding / early tokens
        summed = masked.sum(dim=1)                           # (batch, d_model)
        averaged = (summed / token_counts.unsqueeze(-1))     # (batch, d_model)
        mean_acts[layer_idx] = averaged.cpu().float()

        # Mean norm of individual token activations (for steering normalisation)
        per_token_norms = acts.norm(dim=-1)                  # (batch, seq_len)
        masked_norms = per_token_norms * position_mask
        mean_norms[layer_idx] = (masked_norms.sum(dim=1) / token_counts).cpu().float()

    return mean_acts, mean_norms


def process_file(
    model,
    tokenizer,
    story_path: Path,
    layer_indices: list[int],
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Extract activations for all stories in a single JSONL file.

    Returns
    -------
    acts : {layer_idx: tensor of shape (num_stories, d_model)}
    norms : {layer_idx: tensor of shape (num_stories,)}  — per-story mean residual stream norms
    """
    stories = load_stories(story_path)
    if not stories:
        return {}, {}

    # Accumulators: {layer_idx: [list of (sub_batch, d_model) tensors]}
    all_acts: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
    all_norms: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
    current_bs = batch_size

    # Use a manual index instead of range() so that OOM-reduced batch sizes
    # don't cause items to be silently skipped (the step must track current_bs).
    i = 0
    pbar = tqdm(total=len(stories), desc=f"  {story_path.stem}", leave=False)
    while i < len(stories):
        batch_stories = stories[i : i + current_bs]

        try:
            inputs = tokenizer(
                batch_stories,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with ActivationCapture(model, layer_indices) as cap:
                with torch.no_grad():
                    model(**inputs)
                raw_acts = cap.get()

            batch_acts, batch_norms = extract_mean_activations(
                raw_acts, inputs["attention_mask"]
            )
            for idx in layer_indices:
                all_acts[idx].append(batch_acts[idx])
                all_norms[idx].append(batch_norms[idx])

            pbar.update(len(batch_stories))
            i += len(batch_stories)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_bs > 1:
                current_bs = max(1, current_bs // 2)
                torch.cuda.empty_cache()
                print(f"    OOM — retrying with batch_size={current_bs}")
            else:
                raise

    pbar.close()

    # Concatenate across batches
    acts = {idx: torch.cat(tensors, dim=0) for idx, tensors in all_acts.items()}
    norms = {idx: torch.cat(tensors, dim=0) for idx, tensors in all_norms.items()}
    return acts, norms


def main():
    args = parse_args()
    config.ensure_dirs()

    print(f"Model: {args.model_id}")
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=config.DTYPE,
        device_map="auto",
    )
    model.eval()

    layer_indices = args.layers or config.get_target_layers(model)
    print(f"Target layers: {layer_indices}")
    print(f"Hidden dim:    {config.get_text_config(model).hidden_size}")
    print()

    # Create output dirs for each layer
    for idx in layer_indices:
        (config.ACTIVATIONS_DIR / str(idx)).mkdir(parents=True, exist_ok=True)

    # Process all story files
    story_files = sorted(config.STORIES_DIR.glob("*.jsonl"))
    if not story_files:
        print("ERROR: No story files found. Run 01_generate_stories.py first.")
        return

    t_start = time.time()

    # Accumulate norms across all files to compute a global mean per layer
    global_norms: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    for story_path in tqdm(story_files, desc="Extracting activations"):
        emotion_name = story_path.stem

        # Check if already extracted
        already_done = all(
            (config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}.pt").exists()
            for idx in layer_indices
        )
        if already_done:
            print(f"  {emotion_name}: already extracted, skipping")
            # Still load norms if the norm file exists for global averaging
            for idx in layer_indices:
                norm_path = config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}_norms.pt"
                if norm_path.exists():
                    global_norms[idx].append(torch.load(norm_path, weights_only=True))
            continue

        acts_by_layer, norms_by_layer = process_file(
            model, tokenizer, story_path, layer_indices, args.batch_size
        )

        for idx in layer_indices:
            if idx in acts_by_layer:
                out_path = config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}.pt"
                torch.save(acts_by_layer[idx], out_path)

                norm_path = config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}_norms.pt"
                torch.save(norms_by_layer[idx], norm_path)
                global_norms[idx].append(norms_by_layer[idx])

        n = next(iter(acts_by_layer.values())).shape[0] if acts_by_layer else 0
        d = next(iter(acts_by_layer.values())).shape[1] if acts_by_layer else 0
        print(f"  {emotion_name}: {n} stories × {d}-dim activations saved")

        # Free VRAM between emotion groups
        torch.cuda.empty_cache()

    # Save global mean residual stream norm per layer (for steering normalisation)
    for idx in layer_indices:
        if global_norms[idx]:
            mean_norm = torch.cat(global_norms[idx]).mean().item()
            norm_path = config.ACTIVATIONS_DIR / str(idx) / "mean_residual_norm.pt"
            torch.save(torch.tensor(mean_norm), norm_path)
            print(f"  Layer {idx} mean residual stream norm: {mean_norm:.2f}")

    elapsed = time.time() - t_start
    print(f"\nDone. Extraction completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
