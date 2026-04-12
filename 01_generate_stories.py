#!/usr/bin/env python3
"""
Step 1 — Generate short stories for each emotion and a neutral baseline.

For each emotion in the vocabulary, we prompt Gemma 4 to write short
(3-5 sentence) stories depicting a character experiencing that emotion.
We deliberately vary the character name, setting, and situation across
stories to avoid confounds.  The resulting stories are later fed back
through the model to extract residual-stream activations (step 2).

Methodology reference:
    Anthropic (2026) §2.1 — "We prompted the model to generate short
    fictional passages in which a character experiences each emotion."

Usage:
    python 01_generate_stories.py                     # defaults
    python 01_generate_stories.py --num-stories 10    # quick test run
    python 01_generate_stories.py --model-id google/gemma-4-E4B-it
"""

import argparse
import json
import random
import time
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from utils.prompts import build_emotion_prompt, build_neutral_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate emotion and neutral stories")
    p.add_argument("--num-stories", type=int, default=config.NUM_STORIES_PER_EMOTION,
                    help="Stories per emotion (default: %(default)s)")
    p.add_argument("--num-neutral", type=int, default=None,
                    help="Neutral stories (default: 2 × --num-stories)")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
) -> list[str]:
    """Generate responses for a list of prompts, with OOM retry at half batch size."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Build chat-formatted inputs
        conversations = [
            [{"role": "user", "content": p}] for p in batch_prompts
        ]
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]

        # Process this batch, halving sub-batch size on OOM.
        # We track which items have been processed to avoid duplicates on retry.
        processed = 0
        current_bs = len(texts)

        while processed < len(texts):
            sub_texts = texts[processed : processed + current_bs]

            try:
                inputs = tokenizer(
                    sub_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
                        temperature=config.GENERATION_TEMPERATURE,
                        top_p=config.GENERATION_TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Decode only the new tokens (skip the prompt)
                for j, output in enumerate(outputs):
                    prompt_len = inputs["input_ids"][j].shape[0]
                    story = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                    results.append(story.strip())

                processed += len(sub_texts)

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    torch.cuda.empty_cache()
                    print(f"  OOM — retrying with batch_size={current_bs}")
                else:
                    raise

    return results


def generate_stories_for_emotion(
    model,
    tokenizer,
    emotion: str,
    num_stories: int,
    batch_size: int,
    rng: random.Random,
) -> list[dict]:
    """Generate stories for one emotion, return list of {emotion, story, prompt}."""
    prompts = [build_emotion_prompt(emotion, rng=rng) for _ in range(num_stories)]
    stories = generate_batch(model, tokenizer, prompts, batch_size)
    return [
        {"emotion": emotion, "story": story, "prompt": prompt}
        for prompt, story in zip(prompts, stories)
    ]


def generate_neutral_stories(
    model,
    tokenizer,
    num_stories: int,
    batch_size: int,
    rng: random.Random,
) -> list[dict]:
    """Generate emotionally neutral stories."""
    prompts = [build_neutral_prompt(rng=rng) for _ in range(num_stories)]
    stories = generate_batch(model, tokenizer, prompts, batch_size)
    return [
        {"emotion": "neutral", "story": story, "prompt": prompt}
        for prompt, story in zip(prompts, stories)
    ]


def main():
    args = parse_args()
    num_neutral = args.num_neutral or 2 * args.num_stories
    rng = random.Random(args.seed)

    config.ensure_dirs()
    emotions = config.load_emotions()

    print(f"Model:          {args.model_id}")
    print(f"Stories/emotion: {args.num_stories}")
    print(f"Neutral stories: {num_neutral}")
    print(f"Emotions:        {[e['name'] for e in emotions]}")
    print()

    # Load model and tokenizer
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
    print(f"Model loaded on {model.device if hasattr(model, 'device') else 'multiple devices'}")
    print()

    t_start = time.time()

    # Generate emotion stories
    for emo in tqdm(emotions, desc="Emotions"):
        name = emo["name"]
        out_path = config.STORIES_DIR / f"{name}.jsonl"

        if out_path.exists():
            existing = sum(1 for _ in open(out_path))
            if existing >= args.num_stories:
                print(f"  {name}: {existing} stories already exist, skipping")
                continue

        entries = generate_stories_for_emotion(
            model, tokenizer, name, args.num_stories, args.batch_size, rng
        )
        with jsonlines.open(out_path, mode="w") as writer:
            writer.write_all(entries)
        print(f"  {name}: wrote {len(entries)} stories → {out_path}")

    # Generate neutral stories
    neutral_path = config.STORIES_DIR / "neutral.jsonl"
    if neutral_path.exists() and sum(1 for _ in open(neutral_path)) >= num_neutral:
        print(f"  neutral: stories already exist, skipping")
    else:
        print("Generating neutral stories...")
        entries = generate_neutral_stories(
            model, tokenizer, num_neutral, args.batch_size, rng
        )
        with jsonlines.open(neutral_path, mode="w") as writer:
            writer.write_all(entries)
        print(f"  neutral: wrote {len(entries)} stories → {neutral_path}")

    elapsed = time.time() - t_start
    total = args.num_stories * len(emotions) + num_neutral
    print(f"\nDone. Generated {total} stories in {elapsed:.1f}s ({elapsed/total:.2f}s/story)")


if __name__ == "__main__":
    main()
