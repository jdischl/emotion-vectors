#!/usr/bin/env python3
"""
Step 2 — Extract residual-stream activations for each story.

For every story generated in step 1, we run a forward pass through Gemma 4
and capture the residual-stream hidden states at three target layers (25%,
50%, 75% depth, snapped to global-attention layers).  We extract the
activation at the *last non-padding token*, which—in a decoder-only
transformer—has attended to the full story and is the most information-rich
position.

Note on tokenisation:
    Stories are fed as raw text (no chat template) even though they were
    generated with one.  This is intentional — we want activations reflecting
    the model's representation of the *emotional content*, not the chat
    framing.  Chat template tokens (system markers, turn delimiters) would
    add structural noise that dilutes the emotion signal.

Methodology reference:
    Anthropic (2026) §2.2 — "We recorded the residual stream activations at
    the midpoint of the network for each passage."

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


def extract_last_token_activations(
    activations: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Index into full sequence activations to get the last real (non-pad) token per sequence.

    Parameters
    ----------
    activations : {layer_idx: tensor of shape (batch, seq_len, d_model)}
    attention_mask : tensor of shape (batch, seq_len), 1 for real tokens

    Returns
    -------
    {layer_idx: tensor of shape (batch, d_model)}
    """
    # Position of the last non-padding token in each sequence
    last_pos = attention_mask.sum(dim=1) - 1  # (batch,)

    result = {}
    for layer_idx, acts in activations.items():
        # Gather the activation at the last real token for each item in the batch
        batch_indices = torch.arange(acts.size(0), device=acts.device)
        result[layer_idx] = acts[batch_indices, last_pos].cpu().float()
    return result


def process_file(
    model,
    tokenizer,
    story_path: Path,
    layer_indices: list[int],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Extract activations for all stories in a single JSONL file.

    Returns {layer_idx: tensor of shape (num_stories, d_model)}.
    """
    stories = load_stories(story_path)
    if not stories:
        return {}

    # Accumulators: {layer_idx: [list of (sub_batch, d_model) tensors]}
    all_acts: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}
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

            last_token_acts = extract_last_token_activations(
                raw_acts, inputs["attention_mask"]
            )
            for idx in layer_indices:
                all_acts[idx].append(last_token_acts[idx])

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
    return {idx: torch.cat(tensors, dim=0) for idx, tensors in all_acts.items()}


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
        torch_dtype=config.DTYPE,
        device_map="auto",
    )
    model.eval()

    layer_indices = args.layers or config.get_target_layers(model)
    print(f"Target layers: {layer_indices}")
    print(f"Hidden dim:    {model.config.hidden_size}")
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

    for story_path in tqdm(story_files, desc="Extracting activations"):
        emotion_name = story_path.stem

        # Check if already extracted
        already_done = all(
            (config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}.pt").exists()
            for idx in layer_indices
        )
        if already_done:
            print(f"  {emotion_name}: already extracted, skipping")
            continue

        acts_by_layer = process_file(model, tokenizer, story_path, layer_indices, args.batch_size)

        for idx, tensor in acts_by_layer.items():
            out_path = config.ACTIVATIONS_DIR / str(idx) / f"{emotion_name}.pt"
            torch.save(tensor, out_path)

        n = next(iter(acts_by_layer.values())).shape[0] if acts_by_layer else 0
        d = next(iter(acts_by_layer.values())).shape[1] if acts_by_layer else 0
        print(f"  {emotion_name}: {n} stories × {d}-dim activations saved")

        # Free VRAM between emotion groups
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nDone. Extraction completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
