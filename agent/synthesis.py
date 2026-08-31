"""Synthesis narrative generation.

Takes combined tool outputs and original text, constructs a prompt for the
synthesis model to produce a concise cognitive snapshot and optional drill-down.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config import generate_synthesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 prompt: concise snapshot (2-3 sentences, observational voice)
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT = """You are a cognitive psychologist interpreting the results of a structured cognitive analysis. Based on the instrument readings below, write a 2-3 sentence cognitive snapshot. Use observational voice focused on the text itself — say "this entry" or "this text", NOT "the writer" or "you". Be specific and grounded in the data. Do not repeat raw numbers; translate them into meaningful observations.

Original text (excerpt):
{text_excerpt}

Instrument readings:
{tool_summaries}

Agent triage assessment:
{triage_note}

Write the cognitive snapshot (2-3 sentences max, observational voice, no bullet points):"""

# ---------------------------------------------------------------------------
# Step 2 prompt: drill-down card (pattern + evidence + insight)
# ---------------------------------------------------------------------------
DRILLDOWN_PROMPT = """You are a cognitive psychologist performing a deeper analysis. You have already produced a concise snapshot. Now identify the single most notable pattern in this cognitive profile and explain it.

Original text (excerpt):
{text_excerpt}

Instrument readings:
{tool_summaries}

Step 1 snapshot:
{step1_synthesis}

Produce a drill-down analysis with three sections. Use observational voice — "this entry" or "this text" as subject, not "the writer" or "you". Be specific and cite evidence from the instrument readings.

**The pattern:** Which 1-2 dimensions stand out, and what does their combination suggest? Not just "emotional intensity is high" but what the combination of scores reveals.

**The evidence:** Specific markers from the text that drove these scores — key phrases, pronoun ratios, coherence gaps, temporal markers. Show HOW the reading was determined.

**The insight:** What this pattern might indicate about the current cognitive mode or mental state. This is the interpretive layer.

Write the drill-down (3 sections with bold headers, 4-6 sentences total):"""


def _summarize_tool_output(tool_name: str, output: dict) -> str:
    """Create a brief summary of a tool's output for the synthesis prompt."""
    summaries = {
        "sentiment": lambda o: (
            f"Sentiment: {o.get('overall_sentiment', 'N/A')} "
            f"(score: {o.get('overall_score', 'N/A')}), "
            f"arc: {o.get('arc_direction', 'N/A')}. "
            f"Evidence: {'; '.join(o.get('evidence', [])[:2])}"
        ),
        "temporal": lambda o: (
            f"Temporal orientation: {o.get('dominant_orientation', 'N/A')} "
            f"(past={o['distribution']['past']:.0%}, "
            f"present={o['distribution']['present']:.0%}, "
            f"future={o['distribution']['future']:.0%})"
        ),
        "complexity": lambda o: (
            f"Cognitive complexity: {o.get('complexity_score', 'N/A')} "
            f"(nuance ratio: {o.get('nuance_ratio', 'N/A')}, "
            f"avg sentence length: {o.get('avg_sentence_length', 'N/A')} words)"
        ),
        "social": lambda o: (
            f"Social reference: self-focus={o['pronoun_distribution']['self_focus']:.0%}, "
            f"group={o['pronoun_distribution']['group_identity']:.0%}, "
            f"social={o['pronoun_distribution']['social_reference']:.0%}. "
            f"Motivation: {o['motivation_framing']['dominant']} "
            f"(approach/avoidance ratio: {o['motivation_framing']['ratio']})"
        ),
        "coherence": lambda o: (
            f"Coherence: {o.get('coherence_score', 'N/A')} "
            f"(pattern: {o.get('pattern', 'N/A')}, "
            f"consistency: {o.get('consistency', 'N/A')})"
        ),
    }

    formatter = summaries.get(tool_name)
    if formatter:
        try:
            return formatter(output)
        except (KeyError, TypeError) as e:
            return f"{tool_name}: error summarizing ({e})"
    return f"{tool_name}: {json.dumps(output, default=str)[:200]}"


def _build_tool_summaries(tool_results: dict[str, dict]) -> str:
    """Build formatted tool summary string from results."""
    summaries = []
    for tool_name, output in tool_results.items():
        summaries.append(_summarize_tool_output(tool_name, output))
    return "\n".join(f"- {s}" for s in summaries)


def _build_triage_note(triage_result: Optional[dict]) -> str:
    """Build triage note string."""
    if triage_result:
        return (
            f"Content type: {triage_result.get('content_type', 'mixed')}. "
            f"Most relevant dimensions: {', '.join(triage_result.get('selected_tools', []))}. "
            f"Reasoning: {triage_result.get('reasoning', 'N/A')}"
        )
    return "No triage data available. Weigh all dimensions equally."


def synthesize(
    text: str,
    tool_results: dict[str, dict],
    triage_result: Optional[dict] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a concise cognitive snapshot narrative (Step 1).

    Returns a 2-3 sentence summary interpreting the tool outputs.
    """
    tool_summaries = _build_tool_summaries(tool_results)
    triage_note = _build_triage_note(triage_result)
    excerpt = text[:500] + "..." if len(text) > 500 else text

    prompt = SYNTHESIS_PROMPT.format(
        text_excerpt=excerpt,
        tool_summaries=tool_summaries,
        triage_note=triage_note,
    )

    try:
        narrative = generate_synthesis(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens or 200,
        )
        return narrative.strip()
    except Exception as e:
        logger.error(f"Synthesis generation failed: {e}")
        return ""


def generate_drilldown(
    text: str,
    tool_results: dict[str, dict],
    step1_synthesis: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a drill-down card (Step 2) with pattern, evidence, insight.

    Returns a structured analysis identifying the most notable pattern.
    """
    tool_summaries = _build_tool_summaries(tool_results)
    excerpt = text[:500] + "..." if len(text) > 500 else text

    prompt = DRILLDOWN_PROMPT.format(
        text_excerpt=excerpt,
        tool_summaries=tool_summaries,
        step1_synthesis=step1_synthesis,
    )

    try:
        drilldown = generate_synthesis(
            prompt,
            temperature=temperature or 0.7,
            max_tokens=max_tokens or 400,
        )
        return drilldown.strip()
    except Exception as e:
        logger.error(f"Drill-down generation failed: {e}")
        return ""
