"""Tool 5: Coherence & Fragmentation Meter.

Measures how much the text holds together as a unified thread vs. jumps
between disconnected topics, using sentence embeddings and cosine similarity.

Grounding: Discourse coherence literature. Stream-of-consciousness writing
gets more fragmented under cognitive load or emotional distress.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_embedder = None
_nlp = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _classify_pattern(coherence: float, consistency: float) -> str:
    """Classify the coherence pattern.

    Thresholds calibrated for MiniLM consecutive cosine similarities,
    which typically range 0.0-0.5 for same-topic sentences.
    """
    if coherence >= 0.25:
        if consistency >= 0.5:
            return "focused"
        return "structured_exploration"
    if coherence >= 0.1:
        return "wandering"
    return "fragmented"


def analyze_coherence(text: str) -> dict[str, Any]:
    """Measure coherence and fragmentation of the input text.

    Returns coherence score, consistency, pattern classification,
    similarity sequence, and topic break indices.
    """
    if not text or not text.strip():
        return {
            "coherence_score": 0.0,
            "consistency": 0.0,
            "pattern": "fragmented",
            "similarity_sequence": [],
            "topic_breaks": [],
        }

    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Need at least 2 sentences for coherence analysis
    if len(sentences) < 2:
        return {
            "coherence_score": 1.0,
            "consistency": 1.0,
            "pattern": "focused",
            "similarity_sequence": [],
            "topic_breaks": [],
        }

    # Encode all sentences
    embedder = _get_embedder()
    embeddings = embedder.encode(sentences)

    # Compute consecutive cosine similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(round(sim, 3))

    # Coherence score: average consecutive similarity
    coherence_score = float(np.mean(similarities))

    # Consistency: inverse of variance (normalized to 0-1)
    variance = float(np.var(similarities))
    # Map variance to 0-1: low variance -> high consistency
    # Typical variance range is 0 to 0.1, so scale accordingly
    consistency = max(0.0, 1.0 - (variance * 10))

    # Pattern classification
    pattern = _classify_pattern(coherence_score, consistency)

    # Topic breaks: indices where similarity drops sharply
    # A "sharp drop" is a similarity below mean - 1 std
    if len(similarities) >= 3:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim - std_sim
        topic_breaks = [
            i + 1 for i, s in enumerate(similarities)
            if s < threshold and s < 0.3
        ]
    else:
        topic_breaks = []

    return {
        "coherence_score": round(coherence_score, 3),
        "consistency": round(consistency, 3),
        "pattern": pattern,
        "similarity_sequence": similarities,
        "topic_breaks": topic_breaks,
    }
