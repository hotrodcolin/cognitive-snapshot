"""Agent triage and planning step.

Takes raw text and a prompt style (1=minimal, 2=structured, 3=role-based),
sends to the synthesis model, and parses the response to extract content type,
selected tools, and reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import DEFAULT_TOOLS, MIN_TEXT_LENGTH, generate_synthesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Three prompt styles per SPEC.md Section 6
# ---------------------------------------------------------------------------
PROMPTS = {
    1: """Analyze this text and determine what analysis tools to use.

Text: {input_text}

Available tools: sentiment, temporal, cognitive_complexity, social_reference, coherence

Respond in JSON with:
- content_type: emotional/intellectual/narrative/mixed
- selected_tools: list of tool names to run
- reasoning: why these tools""",

    2: """You are analyzing a stream-of-consciousness text sample. Your job is to triage the input and select the most relevant cognitive analysis tools.

Consider:
- Is the text emotionally charged or intellectually focused?
- Does the writer reference past events, present experience, or future plans?
- How complex is the reasoning?
- Is the text self-focused or socially oriented?
- Does the text stay on topic or jump around?

Text: {input_text}

Available tools:
1. sentiment - emotional tone and arc detection
2. temporal - past/present/future focus
3. complexity - reasoning sophistication
4. social - self vs. other focus, approach vs. avoidance
5. coherence - topic consistency and fragmentation

Respond in JSON with:
- content_type: emotional/intellectual/narrative/mixed
- selected_tools: ordered list of most relevant tools (minimum 3)
- reasoning: 2-3 sentences explaining your selection""",

    3: """You are a cognitive psychologist reviewing a patient's free-association transcript. Your clinical training tells you that unfiltered speech reveals psychological state through specific linguistic markers: emotional valence, temporal focus, reasoning complexity, social orientation, and discourse coherence.

Based on your clinical expertise, review this transcript and determine which analytical instruments to apply, in what order, and why.

Transcript: {input_text}

Available instruments:
1. sentiment - emotional tone and arc
2. temporal - temporal focus distribution
3. complexity - integrative complexity scoring
4. social - pronoun analysis and motivational framing
5. coherence - discourse coherence measurement

Respond in JSON with:
- content_type: your clinical impression of the content type (emotional/intellectual/narrative/mixed)
- selected_tools: instruments to apply, ordered by clinical relevance
- reasoning: your clinical rationale""",
}

# Canonical tool names that the agent might refer to in various ways
TOOL_ALIASES = {
    "sentiment": "sentiment",
    "emotion": "sentiment",
    "emotional": "sentiment",
    "temporal": "temporal",
    "temporal_orientation": "temporal",
    "time": "temporal",
    "complexity": "complexity",
    "cognitive_complexity": "complexity",
    "cognitive": "complexity",
    "social": "social",
    "social_reference": "social",
    "pronoun": "social",
    "coherence": "coherence",
    "fragmentation": "coherence",
    "discourse": "coherence",
}


def _parse_triage_response(response: str) -> dict[str, Any] | None:
    """Parse the model's triage response into structured data."""
    # Try JSON extraction first
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "selected_tools" in data and "content_type" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction
    content_type = None
    ct_match = re.search(r"content_type[\"']?\s*:\s*[\"']?(\w+)", response, re.I)
    if ct_match:
        content_type = ct_match.group(1).lower()

    tools = []
    tools_match = re.search(r"selected_tools[\"']?\s*:\s*\[([^\]]+)\]", response, re.I)
    if tools_match:
        raw = tools_match.group(1)
        tools = [t.strip().strip("\"'") for t in raw.split(",")]

    reasoning = None
    reason_match = re.search(r"reasoning[\"']?\s*:\s*[\"']?(.+?)(?:[\"']?\s*[,}]|$)", response, re.I | re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip().strip("\"'")

    if tools:
        return {
            "content_type": content_type or "mixed",
            "selected_tools": tools,
            "reasoning": reasoning or "",
        }
    return None


def _normalize_tool_names(tools: list[str]) -> list[str]:
    """Map various tool name variants to canonical names."""
    normalized = []
    for t in tools:
        canonical = TOOL_ALIASES.get(t.lower().strip(), None)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def triage(text: str, prompt_style: int = 2) -> dict[str, Any]:
    """Run the triage/planning step on the input text.

    Returns a dict with content_type, selected_tools, and reasoning.
    Falls back to all tools if the model response cannot be parsed.
    """
    # Check text length
    word_count = len(text.split())
    if word_count < MIN_TEXT_LENGTH:
        return {
            "content_type": "insufficient",
            "selected_tools": [],
            "reasoning": (
                f"Text has {word_count} words, below the {MIN_TEXT_LENGTH}-word "
                "minimum for meaningful analysis."
            ),
            "error": None,
        }

    prompt_template = PROMPTS.get(prompt_style, PROMPTS[2])
    # Truncate very long text in the prompt to keep it manageable
    display_text = text[:2000] + "..." if len(text) > 2000 else text
    prompt = prompt_template.format(input_text=display_text)

    try:
        response = generate_synthesis(prompt, temperature=0.3, max_tokens=300)
    except Exception as e:
        logger.warning(f"Triage model call failed: {e}. Defaulting to all tools.")
        return {
            "content_type": "mixed",
            "selected_tools": list(DEFAULT_TOOLS),
            "reasoning": f"Model call failed ({e}). Defaulting to all tools.",
            "error": str(e),
        }

    # Parse the response
    parsed = _parse_triage_response(response)
    if parsed is None:
        logger.warning(f"Could not parse triage response. Defaulting to all tools.")
        return {
            "content_type": "mixed",
            "selected_tools": list(DEFAULT_TOOLS),
            "reasoning": f"Could not parse model response. Defaulting to all tools.",
            "raw_response": response,
            "error": "parse_failure",
        }

    # Normalize tool names
    selected = _normalize_tool_names(parsed.get("selected_tools", []))
    if len(selected) < 3:
        # Ensure at least 3 tools
        for tool in DEFAULT_TOOLS:
            if tool not in selected:
                selected.append(tool)
            if len(selected) >= 3:
                break

    return {
        "content_type": parsed.get("content_type", "mixed"),
        "selected_tools": selected,
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": response,
        "error": None,
    }
