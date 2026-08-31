"""Cognitive Snapshot Agent - Analysis Tools Package."""

from tools.sentiment import analyze_sentiment
from tools.temporal import analyze_temporal
from tools.complexity import analyze_complexity
from tools.social import analyze_social
from tools.coherence import analyze_coherence

TOOL_REGISTRY = {
    "sentiment": analyze_sentiment,
    "temporal": analyze_temporal,
    "complexity": analyze_complexity,
    "social": analyze_social,
    "coherence": analyze_coherence,
}

__all__ = [
    "analyze_sentiment",
    "analyze_temporal",
    "analyze_complexity",
    "analyze_social",
    "analyze_coherence",
    "TOOL_REGISTRY",
]
