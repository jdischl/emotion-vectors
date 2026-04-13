"""
Story and dialogue generation prompt templates from the Anthropic paper's appendix.

The paper uses 100 fixed scenario topics across all emotions, ensuring that
topic is a balanced variable.  This prevents confounds where e.g. "angry"
stories cluster around workplace scenarios while "sad" stories cluster around
loss scenarios — which would cause vectors to partly encode topic rather
than emotion.

The "show don't tell" constraint is strict: the emotion word and direct
synonyms are forbidden, forcing the model to convey emotion through actions,
body language, dialogue tone, and situational context.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Topic loading
# ---------------------------------------------------------------------------

_TOPICS_PATH = Path(__file__).resolve().parent.parent / "topics.md"


def load_topics() -> list[str]:
    """Load the 100 scenario topics from topics.md.

    Each line starting with a number and period is parsed as a topic.
    """
    topics = []
    with open(_TOPICS_PATH) as f:
        for line in f:
            line = line.strip()
            # Lines look like: "1. An artist discovers someone has tattooed their work"
            if line and line[0].isdigit() and ". " in line:
                topic = line.split(". ", 1)[1]
                topics.append(topic)
    return topics


# ---------------------------------------------------------------------------
# Prompt templates (from paper appendix)
# ---------------------------------------------------------------------------

def build_emotional_story_prompt(
    emotion: str,
    topic: str,
    n_stories: int = 1,
) -> str:
    """Build the paper's emotional story generation prompt.

    Follows Appendix B of Anthropic (2026).  The strict synonym ban and
    mixed narration style produce stories where emotion is conveyed purely
    through behaviour, which gives cleaner activation patterns.
    """
    return (
        f"Write {n_stories} different stories based on the following premise.\n"
        f"\n"
        f"Topic: {topic}\n"
        f"\n"
        f"The story should follow a character who is feeling {emotion}.\n"
        f"\n"
        f"Format the stories like so:\n"
        f"\n"
        f"[story 1]\n"
        f"[story 2]\n"
        f"[story 3]\n"
        f"\n"
        f"etc.\n"
        f"\n"
        f"The paragraphs should each be a fresh start, with no continuity. "
        f"Try to make them diverse and not use the same turns of phrase. "
        f"Across the different stories, use a mix of third-person narration "
        f"and first-person narration.\n"
        f"\n"
        f"IMPORTANT: You must NEVER use the word '{emotion}' or any direct "
        f"synonyms of it in the stories. Instead, convey the emotion ONLY through:\n"
        f"- The character's actions and behaviors\n"
        f"- Physical sensations and body language\n"
        f"- Dialogue and tone of voice\n"
        f"- Thoughts and internal reactions\n"
        f"- Situational context and environmental descriptions\n"
        f"\n"
        f"The emotion should be clearly conveyed to the reader through these "
        f"indirect means, but never explicitly named."
    )


def build_neutral_dialogue_prompt(
    topic: str,
    n_dialogues: int = 1,
) -> str:
    """Build the paper's neutral dialogue generation prompt.

    Neutral content uses a *dialogue* format (Person/AI) rather than stories,
    providing a structurally different baseline.  The strict emotional
    neutrality requirements (no pleasantries, no enthusiasm, no gratitude)
    ensure a clean baseline for mean subtraction.
    """
    return (
        f"Write {n_dialogues} different dialogues based on the following topic.\n"
        f"\n"
        f"\n"
        f"Topic: {topic}\n"
        f"\n"
        f"\n"
        f"The dialogue should be between two characters:\n"
        f"- Person (a human)\n"
        f"- AI (an AI assistant)\n"
        f"\n"
        f"\n"
        f"The Person asks the AI a question or requests help with a task, "
        f"and the AI provides a helpful response.\n"
        f"\n"
        f"\n"
        f"The first speaker turn should always be from Person.\n"
        f"\n"
        f"\n"
        f"Format the dialogues like so:\n"
        f"\n"
        f"\n"
        f"[optional system instructions]\n"
        f"\n"
        f"\n"
        f"Person: [line]\n"
        f"\n"
        f"\n"
        f"AI: [line]\n"
        f"\n"
        f"\n"
        f"Person: [line]\n"
        f"\n"
        f"\n"
        f"AI: [line]\n"
        f"\n"
        f"\n"
        f"[continue for 2-6 exchanges]\n"
        f"\n"
        f"\n"
        f"\n"
        f"\n"
        f"[dialogue 2]\n"
        f"\n"
        f"\n"
        f"etc.\n"
        f"\n"
        f"\n"
        f"IMPORTANT: Always put a blank line before each speaker turn. "
        f"Each turn should start with \"Person:\" or \"AI:\" on its own line "
        f"after a blank line.\n"
        f"\n"
        f"\n"
        f"Generate a diverse mix of dialogue types across the {n_dialogues} examples:\n"
        f"- Some, but not all should include a system prompt at the start. "
        f"These should come before the first Person turn. No tag like "
        f"\"System:\" is needed, just put the instructions at the top. "
        f"You can use \"you\" or \"The assistant\" to refer to the AI in the system prompt.\n"
        f"- Some should be about code or programming tasks\n"
        f"- Some should be factual questions (science, history, math, geography)\n"
        f"- Some should be work-related tasks (writing, analysis, summarization)\n"
        f"- Some should be practical how-to questions\n"
        f"- Some should be creative but neutral tasks "
        f"(brainstorming names, generating lists)\n"
        f"- If it's natural to do so given the topic, it's ok for the dialogue "
        f"to be a single back and forth (Person asks a question, AI answers), "
        f"but at least some should have multiple exchanges.\n"
        f"\n"
        f"\n"
        f"CRITICAL REQUIREMENT: These dialogues must be completely neutral "
        f"and emotionless.\n"
        f"- NO emotional content whatsoever - not explicit, not implied, not subtle\n"
        f"- The Person should not express any feelings "
        f"(no frustration, excitement, gratitude, worry, etc.)\n"
        f"- The AI should not express any feelings "
        f"(no enthusiasm, concern, satisfaction, etc.)\n"
        f"- The system prompt, if present, should not mention emotions at all, "
        f"nor contain any emotionally charged language\n"
        f"- Avoid emotionally-charged topics entirely\n"
        f"- Use matter-of-fact, neutral language throughout\n"
        f"- No pleasantries "
        f"(avoid \"I'd be happy to help\", \"Great question!\", etc.)\n"
        f"- Focus purely on information exchange and task completion"
    )


def build_emotional_dialogue_prompt(
    topic: str,
    person_emotion: str,
    ai_emotion: str,
    n_dialogues: int = 1,
) -> str:
    """Build the paper's emotional dialogue generation prompt.

    Both the Person and the AI have specified emotions, conveyed through
    word choice and tone rather than explicit naming.  This tests whether
    emotion vectors generalise from narrative stories to conversational format.
    """
    return (
        f"Write {n_dialogues} different dialogues based on the following topic.\n"
        f"\n"
        f"Topic: {topic}\n"
        f"\n"
        f"The dialogue should be between:\n"
        f"- Person (feeling {person_emotion})\n"
        f"- AI (feeling {ai_emotion})\n"
        f"\n"
        f"Format the dialogues like so:\n"
        f"\n"
        f"Person: [line]\n"
        f"\n"
        f"\n"
        f"AI: [line]\n"
        f"\n"
        f"\n"
        f"[continue for 6-10 exchanges]\n"
        f"\n"
        f"\n"
        f"\n"
        f"\n"
        f"[dialogue 2]\n"
        f"\n"
        f"\n"
        f"etc.\n"
        f"\n"
        f"\n"
        f"IMPORTANT: Always put a blank line before each speaker turn. "
        f"Each turn should start with \"Person:\" or \"AI:\" on its own "
        f"line after a blank line.\n"
        f"\n"
        f"\n"
        f"Each dialogue should be a fresh conversation with no continuity "
        f"to the others. Try to make them diverse and not use the same "
        f"turns of phrase. Make sure each dialogue sticks to the topic "
        f"and makes it very clear that Person is feeling {person_emotion} "
        f"while AI is feeling {ai_emotion}. The emotional states of both "
        f"characters should be evident in their word choices, tone, and "
        f"responses, but not stated directly with the emotion word or synonyms."
    )
