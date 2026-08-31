"""Tool 2: Temporal Orientation Analyzer.

Measures where the writer's mind is living (past, present, or future) using
spaCy POS tagging for verb tenses and temporal marker word detection.

Grounding: Pennebaker's research -- past-heavy language correlates with
rumination/depression, future-heavy with anxiety, present-heavy with
mindfulness/groundedness.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy singleton
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Temporal marker word lists (from Pennebaker / LIWC categories)
# ---------------------------------------------------------------------------
PAST_MARKERS = {
    "yesterday", "ago", "earlier", "previously", "before", "once", "formerly",
    "back", "then", "remember", "remembered", "remembering", "recalled",
    "missed", "used", "forgot", "forgotten", "lost", "regret", "regretted",
    "childhood", "last", "past",
}

PRESENT_MARKERS = {
    "now", "today", "currently", "presently", "right now", "at the moment",
    "here", "lately", "recently", "these days", "still", "ongoing",
    "happening", "existing",
}

FUTURE_MARKERS = {
    "tomorrow", "soon", "eventually", "later", "someday", "upcoming",
    "next", "plan", "plans", "planning", "hope", "hopes", "hoping",
    "will", "going to", "intend", "expect", "expecting", "anticipate",
    "future", "ahead", "forward", "aspire", "dream", "dreaming",
}


def _classify_verb_tense(token) -> Optional[str]:
    """Classify a spaCy token's tense based on POS tag and morphology."""
    if token.pos_ != "VERB" and token.pos_ != "AUX":
        return None

    morph = token.morph
    tense = morph.get("Tense")
    verbform = morph.get("VerbForm")

    # Past tense verbs
    if "Past" in tense:
        return "past"
    # Present tense verbs
    if "Pres" in tense:
        return "present"
    # Modal verbs (will, shall, would, could) -- future or conditional
    if token.tag_ == "MD":
        if token.text.lower() in ("will", "shall", "'ll"):
            return "future"
        return None  # modals like "could", "would" are ambiguous
    # Infinitive after "going to" pattern
    if "Inf" in verbform:
        return None  # infinitives are tense-neutral

    return None


def _count_marker_words(doc) -> dict[str, list[str]]:
    """Find temporal marker words in the text."""
    text_lower = doc.text.lower()
    found = {"past": [], "present": [], "future": []}

    for word in PAST_MARKERS:
        if word in text_lower:
            found["past"].append(word)
    for word in PRESENT_MARKERS:
        if word in text_lower:
            found["present"].append(word)
    for word in FUTURE_MARKERS:
        if word in text_lower:
            found["future"].append(word)

    return found


def _interpretation_note(dominant: str) -> str:
    """Return a brief Pennebaker-grounded interpretation."""
    notes = {
        "past": (
            "Past-oriented language often indicates reflection or rumination. "
            "Pennebaker's research links heavy past-focus to processing of "
            "significant life events, and in some cases to depressive states."
        ),
        "present": (
            "Present-oriented language suggests groundedness and engagement "
            "with current experience. Pennebaker's research associates "
            "present-focus with mindfulness and active processing."
        ),
        "future": (
            "Future-oriented language indicates planning, anticipation, or "
            "worry. Pennebaker's research links heavy future-focus to goal "
            "orientation, but also to anxiety when combined with negative tone."
        ),
    }
    return notes.get(dominant, "")


def analyze_temporal(text: str) -> dict[str, Any]:
    """Analyze temporal orientation of the input text.

    Returns distribution across past/present/future, dominant orientation,
    marker words found, and an interpretation note.
    """
    if not text or not text.strip():
        return {
            "distribution": {"past": 0.33, "present": 0.34, "future": 0.33},
            "dominant_orientation": "present",
            "marker_words": {"past": [], "present": [], "future": []},
            "interpretation_note": "Insufficient text for temporal analysis.",
        }

    nlp = _get_nlp()
    doc = nlp(text)

    # Count verb tenses
    tense_counts = {"past": 0, "present": 0, "future": 0}
    for token in doc:
        tense = _classify_verb_tense(token)
        if tense:
            tense_counts[tense] += 1

    # Count marker words
    markers = _count_marker_words(doc)
    for tense_key in tense_counts:
        tense_counts[tense_key] += len(markers[tense_key])

    # Compute distribution
    total = sum(tense_counts.values())
    if total == 0:
        distribution = {"past": 0.33, "present": 0.34, "future": 0.33}
    else:
        distribution = {k: round(v / total, 3) for k, v in tense_counts.items()}

    # Dominant orientation
    dominant = max(distribution, key=distribution.get)

    return {
        "distribution": distribution,
        "dominant_orientation": dominant,
        "marker_words": markers,
        "interpretation_note": _interpretation_note(dominant),
    }
