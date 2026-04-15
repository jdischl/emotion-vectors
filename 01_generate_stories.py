#!/usr/bin/env python3
"""
Step 1 — Generate short stories for each emotion and neutral dialogues.

For each emotion in the vocabulary, we generate stories across all 100 fixed
scenario topics from the Anthropic paper.  Using fixed topics across all
emotions controls for topic confounds — every emotion gets the same
scenarios, so the vectors capture emotion rather than topic.

Neutral baseline content uses a dialogue format (Person/AI conversations)
with strict emotional neutrality requirements.

If pre-generated data already exists in data/stories/ (e.g. generated
offline with a different model), this script can be skipped entirely.

Methodology reference:
    Anthropic (2026) §2.1, Appendix B

Usage:
    python 01_generate_stories.py                     # defaults
    python 01_generate_stories.py --num-stories 1     # 1 story per topic (100 total per emotion)
    python 01_generate_stories.py --model-id google/gemma-4-E4B-it
"""

import argparse
import random
import time
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from utils.prompts import (
    build_emotional_story_prompt,
    build_neutral_dialogue_prompt,
    load_topics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate emotion stories and neutral dialogues")
    p.add_argument("--num-stories", type=int, default=1,
                    help="Stories per (topic, emotion) pair (default: 1). "
                         "Total per emotion = num_stories × 100 topics")
    p.add_argument("--num-neutral", type=int, default=None,
                    help="Neutral dialogues (default: 2 × total stories per emotion)")
    p.add_argument("--model-id", type=str, default=config.MODEL_ID)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def generate_single(
    model,
    tokenizer,
    prompt: str,
    batch_size: int,
) -> str:
    """Generate a single response to a prompt, with OOM retry."""
    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    current_bs = 1  # single generation
    while True:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
                    temperature=config.GENERATION_TEMPERATURE,
                    top_p=config.GENERATION_TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  OOM on generation — clearing cache and retrying")
            else:
                raise


def parse_stories_from_response(response: str) -> list[str]:
    """Parse multiple stories from a single model response.

    The prompt asks the model to format stories as [story 1], [story 2], etc.
    We split on these markers or on double newlines as a fallback.
    """
    import re

    # Try splitting on [story N] markers
    parts = re.split(r"\[story\s*\d+\]", response, flags=re.IGNORECASE)
    stories = [p.strip() for p in parts if p.strip() and len(p.strip()) > 50]

    if stories:
        return stories

    # Fallback: split on double-newline paragraphs
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip() and len(p.strip()) > 50]
    return paragraphs if paragraphs else [response.strip()]


def main():
    args = parse_args()
    config.ensure_dirs()
    emotions = config.load_emotions()
    topics = load_topics()

    total_per_emotion = args.num_stories * len(topics)
    num_neutral = args.num_neutral or 2 * total_per_emotion

    print(f"Model:              {args.model_id}")
    print(f"Topics:             {len(topics)}")
    print(f"Stories/topic:      {args.num_stories}")
    print(f"Total/emotion:      {total_per_emotion}")
    print(f"Neutral dialogues:  {num_neutral}")
    print(f"Emotions:           {[e['name'] for e in emotions]}")
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
    print()

    t_start = time.time()

    # Generate emotion stories: iterate over emotions × topics
    for emo in emotions:
        name = emo["name"]
        out_path = config.STORIES_DIR / f"{name}.jsonl"

        if out_path.exists():
            with open(out_path) as f:
                existing = sum(1 for _ in f)
            if existing >= total_per_emotion:
                print(f"  {name}: {existing} stories already exist, skipping")
                continue

        print(f"  Generating stories for: {name}")
        entries = []

        for topic in tqdm(topics, desc=f"    {name}", leave=False):
            prompt = build_emotional_story_prompt(name, topic, n_stories=args.num_stories)
            response = generate_single(model, tokenizer, prompt, args.batch_size)
            stories = parse_stories_from_response(response)

            for story in stories[:args.num_stories]:
                entries.append({
                    "emotion": name,
                    "story": story,
                    "topic": topic,
                    "prompt": prompt,
                })

        with jsonlines.open(out_path, mode="w") as writer:
            writer.write_all(entries)
        print(f"    {name}: wrote {len(entries)} stories → {out_path}")

    # Generate neutral dialogues
    neutral_path = config.STORIES_DIR / "neutral.jsonl"
    _n_neutral = 0
    if neutral_path.exists():
        with open(neutral_path) as f:
            _n_neutral = sum(1 for _ in f)
    if _n_neutral >= num_neutral:
        print(f"  neutral: dialogues already exist, skipping")
    else:
        print("  Generating neutral dialogues...")
        entries = []
        # Spread neutral dialogues across topics
        dialogues_per_topic = max(1, num_neutral // len(topics))

        for topic in tqdm(topics, desc="    neutral", leave=False):
            prompt = build_neutral_dialogue_prompt(topic, n_dialogues=dialogues_per_topic)
            response = generate_single(model, tokenizer, prompt, args.batch_size)
            # Each dialogue is the full response for this topic
            entries.append({
                "emotion": "neutral",
                "story": response,  # "story" field for pipeline compatibility
                "topic": topic,
                "prompt": prompt,
            })

        with jsonlines.open(neutral_path, mode="w") as writer:
            writer.write_all(entries)
        print(f"    neutral: wrote {len(entries)} dialogues → {neutral_path}")

    elapsed = time.time() - t_start
    total = total_per_emotion * len(emotions) + num_neutral
    print(f"\nDone. Generated content in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
