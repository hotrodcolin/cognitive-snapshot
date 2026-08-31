"""Tool 4: Self vs. Social Reference Mapper.

Measures pronoun distribution (self-focus vs. social referencing) and
approach vs. avoidance motivational framing.

Grounding: Pennebaker's pronoun research (The Secret Life of Pronouns).
High first-person singular correlates with depression and self-focus.
Approach/avoidance framing reveals motivational posture.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import spacy

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Pronoun categories
# ---------------------------------------------------------------------------
SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
GROUP_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
SOCIAL_PRONOUNS = {
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}

# ---------------------------------------------------------------------------
# Approach vs. avoidance word lists
# ---------------------------------------------------------------------------
APPROACH_WORDS = {
    "want", "hope", "build", "create", "grow", "achieve", "pursue", "explore",
    "seek", "embrace", "aspire", "strive", "aim", "desire", "engage",
    "develop", "improve", "advance", "welcome", "accept", "try", "start",
    "begin", "open", "love", "enjoy", "excited", "eager", "inspired",
}

AVOIDANCE_WORDS = {
    "avoid", "fear", "prevent", "stop", "escape", "reject", "worry",
    "retreat", "hide", "flee", "deny", "refuse", "dread", "hate",
    "afraid", "scared", "anxious", "nervous", "threatened", "defensive",
    "withdraw", "quit", "abandon", "ignore", "resist",
}

# Negated approach patterns (e.g., "don't want", "can't")
NEGATED_PATTERNS = {
    "don't want", "can't", "won't", "shouldn't", "couldn't", "wouldn't",
    "not able", "unable", "never", "no way",
}


def _find_examples(doc, word_set: set[str], limit: int = 3) -> list[str]:
    """Find sentences containing words from the given set."""
    examples = []
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        for word in word_set:
            if re.search(r"\b" + re.escape(word) + r"\b", sent_lower):
                examples.append(sent.text.strip())
                break
        if len(examples) >= limit:
            break
    return examples


def analyze_social(text: str) -> dict[str, Any]:
    """Analyze self vs. social reference and motivational framing.

    Returns pronoun distribution, pronoun density, motivation framing
    (approach vs avoidance), and evidence examples.
    """
    if not text or not text.strip():
        return {
            "pronoun_distribution": {
                "self_focus": 0.0,
                "group_identity": 0.0,
                "social_reference": 0.0,
            },
            "pronoun_density": 0.0,
            "motivation_framing": {
                "approach_count": 0,
                "avoidance_count": 0,
                "ratio": 0.5,
                "dominant": "balanced",
            },
            "evidence": {
                "self_focus_examples": [],
                "social_examples": [],
                "approach_examples": [],
                "avoidance_examples": [],
            },
        }

    nlp = _get_nlp()
    doc = nlp(text)

    # Count pronouns
    self_count = 0
    group_count = 0
    social_count = 0
    total_words = 0

    for token in doc:
        if token.is_punct or token.is_space:
            continue
        total_words += 1
        word = token.text.lower()
        if word in SELF_PRONOUNS:
            self_count += 1
        elif word in GROUP_PRONOUNS:
            group_count += 1
        elif word in SOCIAL_PRONOUNS:
            social_count += 1

    total_pronouns = self_count + group_count + social_count

    # Pronoun distribution (as % of total pronouns)
    if total_pronouns > 0:
        pronoun_dist = {
            "self_focus": round(self_count / total_pronouns, 3),
            "group_identity": round(group_count / total_pronouns, 3),
            "social_reference": round(social_count / total_pronouns, 3),
        }
    else:
        pronoun_dist = {
            "self_focus": 0.0,
            "group_identity": 0.0,
            "social_reference": 0.0,
        }

    # Pronoun density (total pronouns / total words)
    pronoun_density = round(total_pronouns / total_words, 3) if total_words > 0 else 0.0

    # Approach vs. avoidance
    text_lower = text.lower()
    approach_count = sum(
        len(re.findall(r"\b" + re.escape(w) + r"\b", text_lower))
        for w in APPROACH_WORDS
    )
    avoidance_count = sum(
        len(re.findall(r"\b" + re.escape(w) + r"\b", text_lower))
        for w in AVOIDANCE_WORDS
    )
    # Also count negated patterns as avoidance
    for pattern in NEGATED_PATTERNS:
        avoidance_count += text_lower.count(pattern)

    total_motivation = approach_count + avoidance_count
    if total_motivation > 0:
        ratio = round(approach_count / total_motivation, 3)
    else:
        ratio = 0.5

    if ratio > 0.6:
        dominant_motivation = "approach"
    elif ratio < 0.4:
        dominant_motivation = "avoidance"
    else:
        dominant_motivation = "balanced"

    # Evidence
    evidence = {
        "self_focus_examples": _find_examples(doc, SELF_PRONOUNS),
        "social_examples": _find_examples(doc, SOCIAL_PRONOUNS | GROUP_PRONOUNS),
        "approach_examples": _find_examples(doc, APPROACH_WORDS),
        "avoidance_examples": _find_examples(doc, AVOIDANCE_WORDS | NEGATED_PATTERNS),
    }

    return {
        "pronoun_distribution": pronoun_dist,
        "pronoun_density": pronoun_density,
        "motivation_framing": {
            "approach_count": approach_count,
            "avoidance_count": avoidance_count,
            "ratio": ratio,
            "dominant": dominant_motivation,
        },
        "evidence": evidence,
    }
