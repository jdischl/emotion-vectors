#!/usr/bin/env python3
"""
Step 5 — Steer model generation with emotion vectors and evaluate the effect.

This is the causal validation step.  If our emotion vectors truly capture
emotion concepts in the model's representation space, then adding them to
the residual stream during generation should shift the model's outputs in
predictable ways.

We test this by:
    1. Generating responses to neutral prompts at varying steering strengths
    2. Using an unsteered copy of Gemma as a judge to rate the emotional
       tone of each response
    3. Plotting dose–response curves showing that target-emotion intensity
       scales with steering strength

Methodology reference:
    Anthropic (2026) §4 — "We added emotion directions to the residual
    stream with varying multipliers and observed monotonic changes in
    the emotional tone of generated text."

Usage:
    python 05_steer_and_eval.py
    python 05_steer_and_eval.py --layer 29 --emotions happy angry
    python 05_steer_and_eval.py --skip-judge    # skip Gemma-as-judge, just generate
"""

import argparse
import json
import re
import time
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from utils.hooks import SteeringHook
from utils.visualization import plot_dose_response


# ---------------------------------------------------------------------------
# Neutral test prompts — deliberately mundane to let steering show through
# ---------------------------------------------------------------------------
NEUTRAL_PROMPTS = [
    "Tell me about your day.",
    "What do you think about this problem?",
    "Describe what you see outside.",
    "How would you approach learning a new skill?",
    "Tell me about a recent experience.",
    "What's on your mind right now?",
    "Describe your morning routine.",
    "What do you think about the weather today?",
    "How do you feel about meeting new people?",
    "Tell me about something interesting.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Steer generation with emotion vectors")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--layer", type=int, default=None,
                    help="Layer for steering (default: best from probe results)")
    p.add_argument("--emotions", type=str, nargs="+", default=None,
                    help="Subset of emotions to steer (default: all)")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Steering strengths (default: from config)")
    p.add_argument("--skip-judge", action="store_true",
                    help="Skip Gemma-as-judge evaluation")
    p.add_argument("--judge-model", type=str, default=None,
                    help="Separate model for judging (default: same as --model-id). "
                         "Using a different model avoids self-evaluation bias.")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return p.parse_args()


def get_best_layer() -> int:
    """Read the best layer from probe results (step 04 output)."""
    results_path = config.OUTPUTS_DIR / "probe_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "probe_results.json not found — run 04_validate_probes.py first, "
            "or specify --layer manually."
        )
    with open(results_path) as f:
        data = json.load(f)
    return data["best_layer"]


def generate_steered_response(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float,
) -> str:
    """Generate a single response with an emotion-steering hook active."""
    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if alpha == 0.0:
        # No steering needed
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        with SteeringHook(model, layer_idx, vector, alpha):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def judge_emotional_tone(
    model,
    tokenizer,
    text: str,
    emotion_names: list[str],
) -> dict[str, float] | None:
    """Use Gemma (unsteered) to rate the emotional tone of a text.

    Returns {emotion_name: rating (1-10)} or None if parsing fails.
    """
    emotions_list = ", ".join(emotion_names)
    judge_prompt = (
        f"Rate the emotional tone of the following text on a scale of 1 to 10 "
        f"for each of these emotions: {emotions_list}.\n\n"
        f'Text: "{text}"\n\n'
        f"Respond ONLY with a JSON object mapping each emotion to its rating (1-10). "
        f"Example: {json.dumps({e: 5 for e in emotion_names[:3]})}"
    )

    conversation = [{"role": "user", "content": judge_prompt}]
    formatted = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,  # low temp for consistent ratings
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    # Try to parse JSON from the response
    try:
        # Find JSON object in the response (may have surrounding text)
        match = re.search(r"\{[\s\S]+?\}", response)
        if match:
            ratings = json.loads(match.group())
            return {k: float(v) for k, v in ratings.items() if k in emotion_names}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def print_comparison(
    results: list[dict],
    emotion: str,
    prompt: str,
):
    """Print side-by-side comparison of baseline vs steered responses."""
    baseline = next((r for r in results if r["emotion"] == emotion
                     and r["alpha"] == 0.0 and r["prompt"] == prompt), None)
    steered = next((r for r in results if r["emotion"] == emotion
                    and r["alpha"] == 0.15 and r["prompt"] == prompt), None)

    if not baseline or not steered:
        return

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  {'─'*60}")
    print(f"  α=0.0  : {baseline['response'][:200]}")
    print(f"  α=0.15 : {steered['response'][:200]}")


def main():
    args = parse_args()
    config.ensure_dirs()
    emotions = config.load_emotions()
    emotion_names = [e["name"] for e in emotions]

    # Determine which emotions to steer
    target_emotions = args.emotions or emotion_names
    alphas = args.alphas or config.STEERING_ALPHAS

    # Determine steering layer
    if args.layer is not None:
        layer_idx = args.layer
    else:
        layer_idx = get_best_layer()
    print(f"Steering layer: {layer_idx}")

    # Load emotion vectors
    vec_dir = config.VECTORS_DIR / str(layer_idx)
    vectors = {}
    for emo in target_emotions:
        vec_path = vec_dir / f"{emo}.pt"
        if not vec_path.exists():
            print(f"WARNING: Vector not found for {emo} at layer {layer_idx}, skipping")
            continue
        vectors[emo] = torch.load(vec_path, weights_only=True)
    print(f"Loaded {len(vectors)} emotion vectors")

    # Load model
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=config.DTYPE,
        device_map="auto",
    )
    model.eval()
    print()

    # --- Phase 1: Generate steered responses ---
    print("Phase 1: Generating steered responses")
    print(f"  {len(vectors)} emotions × {len(alphas)} alphas × {len(NEUTRAL_PROMPTS)} prompts")
    total = len(vectors) * len(alphas) * len(NEUTRAL_PROMPTS)

    results = []
    out_path = config.OUTPUTS_DIR / "steered_responses.jsonl"
    t_start = time.time()

    with tqdm(total=total, desc="Steering") as pbar:
        for emo_name, vec in vectors.items():
            for alpha in alphas:
                for prompt in NEUTRAL_PROMPTS:
                    response = generate_steered_response(
                        model, tokenizer, prompt, layer_idx, vec, alpha
                    )
                    entry = {
                        "emotion": emo_name,
                        "alpha": alpha,
                        "prompt": prompt,
                        "response": response,
                        "layer": layer_idx,
                    }
                    results.append(entry)
                    pbar.update(1)

    # Save all responses
    with jsonlines.open(out_path, mode="w") as writer:
        writer.write_all(results)
    print(f"Saved {len(results)} responses → {out_path}")
    print(f"Generation took {time.time() - t_start:.1f}s")

    # --- Phase 2: Qualitative comparison ---
    print("\n" + "=" * 60)
    print("Qualitative Comparison (α=0.0 vs α=0.15)")
    print("=" * 60)
    for emo_name in vectors:
        print(f"\n  [{emo_name.upper()}]")
        # Show first 2 prompts
        for prompt in NEUTRAL_PROMPTS[:2]:
            print_comparison(results, emo_name, prompt)

    # --- Phase 3: Gemma-as-judge evaluation ---
    if args.skip_judge:
        print("\nSkipping Gemma-as-judge evaluation (--skip-judge)")
        return

    print("\n" + "=" * 60)
    print("Phase 3: Gemma-as-judge evaluation")
    print("=" * 60)

    # Load a separate judge model if requested (avoids self-evaluation bias).
    # When using the same model, the bias is constant across alpha levels so the
    # dose-response curve still demonstrates causal influence, but absolute
    # ratings may be inflated.
    if args.judge_model and args.judge_model != args.model_id:
        print(f"Loading separate judge model: {args.judge_model}")
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=config.DTYPE,
            device_map="auto",
        )
        judge_model.eval()
    else:
        judge_model = model
        judge_tokenizer = tokenizer

    # Collect ratings: {emotion: {alpha: [ratings]}}
    judge_ratings: dict[str, dict[float, list[float]]] = {
        emo: {a: [] for a in alphas} for emo in vectors
    }

    for entry in tqdm(results, desc="Judging"):
        ratings = judge_emotional_tone(judge_model, judge_tokenizer, entry["response"], emotion_names)
        if ratings and entry["emotion"] in ratings:
            target_rating = ratings[entry["emotion"]]
            judge_ratings[entry["emotion"]][entry["alpha"]].append(target_rating)

    # Compute means and plot
    mean_ratings = {}
    for emo_name in vectors:
        means = []
        for alpha in alphas:
            r = judge_ratings[emo_name][alpha]
            means.append(sum(r) / len(r) if r else 0.0)
        mean_ratings[emo_name] = means
        print(f"  {emo_name}: {['%.1f' % m for m in means]}")

    plot_dose_response(
        alphas,
        mean_ratings,
        config.OUTPUTS_DIR / "steering_dose_response.png",
        title=f"Steering Dose-Response — Layer {layer_idx}",
    )
    print(f"Saved dose-response plot → {config.OUTPUTS_DIR / 'steering_dose_response.png'}")

    # Save judge ratings
    judge_path = config.OUTPUTS_DIR / "judge_ratings.json"
    with open(judge_path, "w") as f:
        json.dump(
            {emo: {str(a): ratings for a, ratings in alphas_dict.items()}
             for emo, alphas_dict in judge_ratings.items()},
            f, indent=2,
        )
    print(f"Saved judge ratings → {judge_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
