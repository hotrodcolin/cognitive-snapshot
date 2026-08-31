"""Tool 1: Sentiment & Emotional Tone Analyzer.

Detects overall emotional valence and emotional arc within the text using
the Cardiff NLP RoBERTa sentiment model (3-class: negative/neutral/positive).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import spacy
from transformers import pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_sentiment_pipeline = None
_nlp = None


def _get_sentiment_pipeline():
    """Load the RoBERTa sentiment pipeline once."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,  # return all class scores
        )
    return _sentiment_pipeline


def _get_nlp():
    """Load spaCy model once."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Label mapping: Cardiff model labels -> standardized
# ---------------------------------------------------------------------------
LABEL_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def _score_sentence(text: str) -> tuple[str, float]:
    """Return (label, score) for a single sentence. Score is -1 to 1."""
    pipe = _get_sentiment_pipeline()
    results = pipe(text[:512])[0]  # truncate to model max
    # results is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
    best = max(results, key=lambda r: r["score"])
    label = best["label"].lower()
    # Compute weighted score: sum(label_value * probability)
    weighted = sum(LABEL_MAP.get(r["label"].lower(), 0.0) * r["score"] for r in results)
    return label, weighted


def _classify_arc(scores: list[float]) -> str:
    """Classify the emotional arc from a sequence of sentence scores."""
    if len(scores) < 3:
        return "stable"
    first_half = np.mean(scores[: len(scores) // 2])
    second_half = np.mean(scores[len(scores) // 2 :])
    diff = second_half - first_half
    std = np.std(scores)
    if std > 0.5:
        return "volatile"
    if diff > 0.15:
        return "improving"
    if diff < -0.15:
        return "declining"
    return "stable"


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Analyze sentiment and emotional arc of the input text.

    Returns a dictionary with overall sentiment, per-sentence scores,
    arc direction, and evidence sentences.
    """
    if not text or not text.strip():
        return {
            "overall_sentiment": "neutral",
            "overall_score": 0.0,
            "sentence_scores": [],
            "arc_direction": "stable",
            "evidence": [],
        }

    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return {
            "overall_sentiment": "neutral",
            "overall_score": 0.0,
            "sentence_scores": [],
            "arc_direction": "stable",
            "evidence": [],
        }

    # Score each sentence
    sentence_labels = []
    sentence_scores = []
    for sent in sentences:
        label, score = _score_sentence(sent)
        sentence_labels.append(label)
        sentence_scores.append(score)

    # Overall score: mean of sentence scores
    overall_score = float(np.mean(sentence_scores))
    if overall_score > 0.15:
        overall_sentiment = "positive"
    elif overall_score < -0.15:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    # Arc direction
    arc_direction = _classify_arc(sentence_scores)

    # Evidence: top emotionally charged sentences (highest absolute score)
    scored_sents = sorted(
        zip(sentences, sentence_scores), key=lambda x: abs(x[1]), reverse=True
    )
    evidence = [s for s, _ in scored_sents[:3]]

    return {
        "overall_sentiment": overall_sentiment,
        "overall_score": round(overall_score, 3),
        "sentence_scores": [round(s, 3) for s in sentence_scores],
        "arc_direction": arc_direction,
        "evidence": evidence,
    }
