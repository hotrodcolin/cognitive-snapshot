"""Tool 3: Cognitive Complexity Scorer.

Scores how sophisticatedly the writer is reasoning using marker-based analysis
adapted from Suedfeld & Tetlock's integrative complexity framework.

High-stress periods reliably reduce cognitive complexity in writing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import spacy

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Complexity marker word lists
# ---------------------------------------------------------------------------
SUBORDINATE_MARKERS = {
    "because", "although", "if", "when", "while", "unless", "even though",
    "whereas", "since", "as long as", "provided that", "in order to",
    "so that", "given that", "assuming", "despite", "though",
}

CAUSAL_MARKERS = {
    "therefore", "consequently", "as a result", "so", "thus", "hence",
    "accordingly", "for this reason", "it follows", "leads to", "causes",
    "results in", "due to", "owing to",
}

DIFFERENTIATION_MARKERS = {
    "but", "however", "on the other hand", "yet", "whereas", "nevertheless",
    "nonetheless", "conversely", "alternatively", "in contrast", "rather",
    "instead", "on the contrary", "still", "even so",
}

INTEGRATION_MARKERS = {
    "and", "also", "moreover", "furthermore", "in addition", "additionally",
    "as well", "besides", "plus", "along with", "together with", "not only",
    "both", "equally",
}


def _count_markers(text_lower: str, marker_set: set[str]) -> int:
    """Count occurrences of markers in lowercased text."""
    count = 0
    # Count multi-word markers first (to avoid double-counting)
    multi_word = sorted(
        [m for m in marker_set if " " in m], key=len, reverse=True
    )
    remaining = text_lower
    for marker in multi_word:
        occurrences = remaining.count(marker)
        count += occurrences
        remaining = remaining.replace(marker, " " * len(marker))

    # Count single-word markers using word boundaries
    single_word = [m for m in marker_set if " " not in m]
    for marker in single_word:
        pattern = r"\b" + re.escape(marker) + r"\b"
        count += len(re.findall(pattern, remaining))

    return count


def _normalize_complexity(raw: float) -> float:
    """Normalize raw complexity ratio to 0-1 scale using sigmoid-like mapping.

    Empirically, raw complexity ratios above 0.08 indicate high complexity.
    """
    # Sigmoid centered at 0.04 (median expected value), scaled so 0.08+ -> ~0.8+
    return round(float(1.0 / (1.0 + np.exp(-80 * (raw - 0.04)))), 3)


def analyze_complexity(text: str) -> dict[str, Any]:
    """Score cognitive complexity of the input text.

    Returns complexity score (0-1), nuance ratio, average sentence length,
    marker counts, and evidence sentences.
    """
    if not text or not text.strip():
        return {
            "complexity_score": 0.0,
            "nuance_ratio": 0.5,
            "avg_sentence_length": 0.0,
            "marker_counts": {
                "subordinate": 0, "causal": 0,
                "differentiation": 0, "integration": 0,
            },
            "evidence": [],
        }

    nlp = _get_nlp()
    doc = nlp(text)
    text_lower = text.lower()
    words = [t for t in doc if not t.is_punct and not t.is_space]
    total_words = len(words)

    if total_words == 0:
        return {
            "complexity_score": 0.0,
            "nuance_ratio": 0.5,
            "avg_sentence_length": 0.0,
            "marker_counts": {
                "subordinate": 0, "causal": 0,
                "differentiation": 0, "integration": 0,
            },
            "evidence": [],
        }

    # Count markers
    sub_count = _count_markers(text_lower, SUBORDINATE_MARKERS)
    causal_count = _count_markers(text_lower, CAUSAL_MARKERS)
    diff_count = _count_markers(text_lower, DIFFERENTIATION_MARKERS)
    integ_count = _count_markers(text_lower, INTEGRATION_MARKERS)

    # Raw complexity = (subordinate + causal + differentiation) / total_words
    complex_markers = sub_count + causal_count + diff_count
    raw_complexity = complex_markers / total_words

    # Nuance ratio = differentiation / (differentiation + integration)
    if diff_count + integ_count > 0:
        nuance_ratio = diff_count / (diff_count + integ_count)
    else:
        nuance_ratio = 0.5  # balanced default

    # Average sentence length
    sentences = [sent for sent in doc.sents]
    sent_lengths = [
        len([t for t in sent if not t.is_punct and not t.is_space])
        for sent in sentences
    ]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0

    # Normalize to 0-1
    complexity_score = _normalize_complexity(raw_complexity)

    # Evidence: sentences with highest marker density
    evidence = []
    all_markers = (
        SUBORDINATE_MARKERS | CAUSAL_MARKERS | DIFFERENTIATION_MARKERS
    )
    for sent in sentences:
        sent_lower = sent.text.lower()
        sent_marker_count = sum(
            1 for m in all_markers if re.search(r"\b" + re.escape(m) + r"\b", sent_lower)
        )
        if sent_marker_count > 0:
            evidence.append((sent.text.strip(), sent_marker_count))
    evidence.sort(key=lambda x: x[1], reverse=True)
    evidence = [s for s, _ in evidence[:3]]

    return {
        "complexity_score": complexity_score,
        "nuance_ratio": round(nuance_ratio, 3),
        "avg_sentence_length": round(avg_sent_len, 1),
        "marker_counts": {
            "subordinate": sub_count,
            "causal": causal_count,
            "differentiation": diff_count,
            "integration": integ_count,
        },
        "evidence": evidence,
    }
