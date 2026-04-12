"""
Story-generation prompt templates with built-in variety.

We use diverse character names, settings, and situations to prevent the model
from learning surface-level confounds (e.g. always associating a particular
name with a particular emotion).  The "show don't tell" constraint forces
behavioural descriptions rather than emotion labels, producing richer and
more naturalistic activation patterns.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Diversity pools
# ---------------------------------------------------------------------------

CHARACTER_NAMES = [
    "Amara", "Ben", "Chen Wei", "Diana", "Eduardo", "Fatima", "Gabriel",
    "Hana", "Isaac", "Jaya", "Kenji", "Lena", "Marco", "Nadia", "Omar",
    "Priya", "Quinn", "Rosa", "Samuel", "Tomoko", "Uma", "Victor",
    "Wren", "Xander", "Yuki", "Zara",
]

SETTINGS = [
    "a quiet office", "a crowded park", "a small kitchen", "a hospital waiting room",
    "a school hallway", "a busy train station", "a dimly lit café", "a sunlit garden",
    "a construction site", "an empty library", "a noisy market", "a rooftop terrace",
    "a suburban street", "a hotel lobby", "a cramped elevator", "a parking garage",
    "a forest trail", "a beachside boardwalk", "a laundromat", "a museum gallery",
]

SITUATIONS = [
    "preparing for a meeting", "waiting for a friend", "finishing a task",
    "organizing their belongings", "reading a letter", "making a phone call",
    "cooking a meal", "walking to work", "sitting in traffic",
    "cleaning up after an event", "checking their phone", "writing in a journal",
    "packing a suitcase", "browsing a store", "repairing something",
    "helping a stranger", "watching the news", "catching a bus",
    "reviewing a document", "leaving a voicemail",
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_emotion_prompt(emotion: str, *, rng: random.Random | None = None) -> str:
    """Build a prompt that asks the model to write a short story showing *emotion*.

    Each call draws a random character name, setting, and situation so that
    repeated calls for the same emotion produce varied stories.
    """
    r = rng or random.Random()
    name = r.choice(CHARACTER_NAMES)
    setting = r.choice(SETTINGS)
    situation = r.choice(SITUATIONS)

    return (
        f"Write a short story (3-5 sentences) about a character named {name} "
        f"experiencing the emotion of being {emotion} while {situation} in {setting}. "
        f"Show the emotion through their actions and thoughts, not by naming it directly."
    )


def build_neutral_prompt(*, rng: random.Random | None = None) -> str:
    """Build a prompt for a mundane, emotionally neutral story."""
    r = rng or random.Random()
    name = r.choice(CHARACTER_NAMES)
    setting = r.choice(SETTINGS)

    return (
        f"Write a short story (3-5 sentences) about a character named {name} "
        f"doing a mundane everyday activity in {setting}. "
        f"The character should have no strong emotions — just going about their routine."
    )
