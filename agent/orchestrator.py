"""Agent orchestrator - main analysis loop.

Coordinates triage > tool execution > synthesis into a unified
cognitive snapshot result.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agent.triage import triage
from agent.synthesis import synthesize, generate_drilldown
from config import DEFAULT_TOOLS, MIN_TEXT_LENGTH
from tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


def analyze(
    text: str,
    prompt_style: int = 2,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Run the full Cognitive Snapshot Agent pipeline.

    Steps: triage (planning) > tool execution > synthesis.
    Returns a complete cognitive snapshot dictionary.
    """
    start_time = time.time()

    # Check minimum length
    word_count = len(text.split())
    if word_count < MIN_TEXT_LENGTH:
        return {
            "status": "insufficient_text",
            "message": (
                f"Please provide at least {MIN_TEXT_LENGTH} words for a meaningful "
                f"analysis. You provided {word_count}. The more you write, the "
                "richer the snapshot."
            ),
            "word_count": word_count,
        }

    # Step 1: Triage - plan which tools to run
    triage_result = triage(text, prompt_style=prompt_style)

    if triage_result.get("content_type") == "insufficient":
        return {
            "status": "insufficient_text",
            "message": triage_result["reasoning"],
            "word_count": word_count,
        }

    # Step 2: ALWAYS run all 5 tools for consistent radar charts.
    # Triage selection is logged and passed to synthesis for emphasis,
    # but execution runs the full suite every time.
    tool_results = {}
    tool_errors = {}
    for tool_name in DEFAULT_TOOLS:
        if tool_name not in TOOL_REGISTRY:
            continue
        try:
            tool_results[tool_name] = TOOL_REGISTRY[tool_name](text)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            tool_errors[tool_name] = str(e)

    # Step 3: Synthesis narrative (receives triage reasoning for emphasis)
    synthesis_narrative = synthesize(
        text,
        tool_results,
        triage_result=triage_result,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    elapsed = round(time.time() - start_time, 2)

    # Build normalized scores for radar chart (0-1 per dimension)
    scores = _extract_scores(tool_results)

    return {
        "status": "success",
        "word_count": word_count,
        "triage": triage_result,
        "tool_results": tool_results,
        "tool_errors": tool_errors,
        "scores": scores,
        "synthesis": synthesis_narrative,
        "elapsed_seconds": elapsed,
    }


def _extract_scores(tool_results: dict[str, dict]) -> dict[str, float]:
    """Extract normalized 0-1 scores for the radar chart."""
    scores = {}

    if "sentiment" in tool_results:
        # Map -1..1 to 0..1 for emotional intensity (absolute value)
        raw = tool_results["sentiment"].get("overall_score", 0)
        scores["Emotional Intensity"] = round(abs(raw), 3)

    if "temporal" in tool_results:
        # Temporal breadth: how spread across past/present/future
        dist = tool_results["temporal"]["distribution"]
        # Entropy-like measure: max breadth when evenly distributed
        values = [dist["past"], dist["present"], dist["future"]]
        max_v = max(values)
        # 1 - max gives breadth (0 = all in one tense, 0.67 = perfectly spread)
        scores["Temporal Breadth"] = round(min(1.0, (1 - max_v) * 1.5), 3)

    if "complexity" in tool_results:
        scores["Cognitive Complexity"] = tool_results["complexity"].get("complexity_score", 0)

    if "social" in tool_results:
        # Social orientation: inverse of self-focus (1 = fully social, 0 = fully self)
        self_focus = tool_results["social"]["pronoun_distribution"].get("self_focus", 0.5)
        scores["Social Orientation"] = round(1.0 - self_focus, 3)

    if "coherence" in tool_results:
        scores["Coherence"] = tool_results["coherence"].get("coherence_score", 0)

    return scores
