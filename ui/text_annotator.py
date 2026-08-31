"""Text annotation and highlighting logic for the cognitive snapshot UI.

Produces HTML-formatted text with color-coded highlights for each
analysis dimension.
"""

from __future__ import annotations

import re
from typing import Any

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Color schemes per dimension
# ---------------------------------------------------------------------------
COLORS = {
    "sentiment_positive": "#2ecc71",
    "sentiment_negative": "#e74c3c",
    "sentiment_neutral": "#95a5a6",
    "temporal_past": "#636EFA",
    "temporal_present": "#EF553B",
    "temporal_future": "#AB63FA",
    "complexity_marker": "#f39c12",
    "social_self": "#f1c40f",
    "social_group": "#1abc9c",
    "social_other": "#3498db",
    "coherence_break": "#e74c3c",
}


def _highlight(text: str, color: str, label: str = "") -> str:
    """Wrap text in a colored span."""
    title = f' title="{label}"' if label else ""
    return (
        f'<span style="background-color: {color}; padding: 1px 4px; '
        f'border-radius: 3px; color: #fff; font-weight: 500;"{title}>{text}</span>'
    )


def annotate_sentiment(text: str, tool_result: dict) -> str:
    """Highlight sentences by sentiment polarity."""
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    scores = tool_result.get("sentence_scores", [])

    parts = []
    for i, sent in enumerate(sentences):
        if i < len(scores):
            score = scores[i]
            if score > 0.15:
                parts.append(_highlight(sent, COLORS["sentiment_positive"], f"positive ({score:+.2f})"))
            elif score < -0.15:
                parts.append(_highlight(sent, COLORS["sentiment_negative"], f"negative ({score:+.2f})"))
            else:
                parts.append(_highlight(sent, COLORS["sentiment_neutral"], f"neutral ({score:+.2f})"))
        else:
            parts.append(sent)
    return " ".join(parts)


def annotate_temporal(text: str, tool_result: dict) -> str:
    """Highlight temporal marker words by tense."""
    markers = tool_result.get("marker_words", {})
    result = text
    for tense, color_key in [("past", "temporal_past"), ("present", "temporal_present"), ("future", "temporal_future")]:
        for word in markers.get(tense, []):
            pattern = re.compile(r"\b(" + re.escape(word) + r")\b", re.IGNORECASE)
            result = pattern.sub(
                lambda m, c=COLORS[color_key], t=tense: _highlight(m.group(), c, t),
                result,
            )
    return result


def annotate_complexity(text: str, tool_result: dict) -> str:
    """Highlight cognitive complexity marker words."""
    from tools.complexity import (
        SUBORDINATE_MARKERS, CAUSAL_MARKERS, DIFFERENTIATION_MARKERS,
    )
    all_markers = SUBORDINATE_MARKERS | CAUSAL_MARKERS | DIFFERENTIATION_MARKERS
    result = text
    for marker in sorted(all_markers, key=len, reverse=True):
        pattern = re.compile(r"\b(" + re.escape(marker) + r")\b", re.IGNORECASE)
        result = pattern.sub(
            lambda m: _highlight(m.group(), COLORS["complexity_marker"], "complexity marker"),
            result,
        )
    return result


def annotate_social(text: str, tool_result: dict) -> str:
    """Highlight pronouns by category."""
    from tools.social import SELF_PRONOUNS, GROUP_PRONOUNS, SOCIAL_PRONOUNS

    nlp = _get_nlp()
    doc = nlp(text)
    parts = []
    for token in doc:
        word_lower = token.text.lower()
        if word_lower in SELF_PRONOUNS:
            parts.append(_highlight(token.text, COLORS["social_self"], "self"))
        elif word_lower in GROUP_PRONOUNS:
            parts.append(_highlight(token.text, COLORS["social_group"], "group"))
        elif word_lower in SOCIAL_PRONOUNS:
            parts.append(_highlight(token.text, COLORS["social_other"], "other"))
        else:
            parts.append(token.text)
        parts.append(token.whitespace_)
    return "".join(parts)


def annotate_coherence(text: str, tool_result: dict) -> str:
    """Mark topic breaks with visual dividers."""
    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    topic_breaks = set(tool_result.get("topic_breaks", []))

    parts = []
    for i, sent in enumerate(sentences):
        if i in topic_breaks:
            parts.append(
                '<span style="display:block; border-top: 2px dashed #e74c3c; '
                'margin: 6px 0; padding-top: 4px;" '
                'title="topic break"></span>'
            )
        parts.append(sent + " ")
    return "".join(parts)


# Dispatch map for the UI
ANNOTATORS = {
    "sentiment": annotate_sentiment,
    "temporal": annotate_temporal,
    "complexity": annotate_complexity,
    "social": annotate_social,
    "coherence": annotate_coherence,
}


def annotate_text(
    text: str,
    tool_results: dict[str, dict],
    active_dimension: str,
) -> str:
    """Annotate text for a given dimension. Returns HTML string."""
    annotator = ANNOTATORS.get(active_dimension)
    if annotator and active_dimension in tool_results:
        html = annotator(text, tool_results[active_dimension])
    else:
        html = text
    return f'<div style="line-height: 1.8; font-size: 14px; padding: 12px;">{html}</div>'
