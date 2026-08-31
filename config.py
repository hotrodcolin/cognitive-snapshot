"""Configuration and synthesis model abstraction for Cognitive Snapshot Agent."""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env if present

from anthropic import Anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------------------
SYNTHESIS_MODEL = "claude"  # "claude" (primary) or "mistral" (fallback)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-20250514"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # fallback only

# ---------------------------------------------------------------------------
# TOOL CONFIGURATION
# ---------------------------------------------------------------------------
MIN_TEXT_LENGTH = 100  # minimum words for full analysis
DEFAULT_TOOLS = ["sentiment", "temporal", "complexity", "social", "coherence"]

# ---------------------------------------------------------------------------
# GENERATION PARAMETERS
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# Anthropic client singleton
# ---------------------------------------------------------------------------
_claude_client = None


def _get_claude_client() -> Anthropic:
    """Return a cached Anthropic client."""
    global _claude_client
    if _claude_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _claude_client = Anthropic(api_key=api_key)
    return _claude_client


# ---------------------------------------------------------------------------
# Synthesis abstraction - the ONLY place the model choice matters
# ---------------------------------------------------------------------------
def generate_synthesis(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate text from the synthesis model (Claude or Mistral)."""
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    if SYNTHESIS_MODEL == "claude":
        return _call_claude(prompt, temp, tokens)
    else:
        return _call_mistral(prompt, temp, tokens)


def _call_claude(prompt: str, temperature: float, max_tokens: int) -> str:
    """Call Claude API for synthesis."""
    client = _get_claude_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_mistral(prompt: str, temperature: float, max_tokens: int) -> str:
    """Fallback: call local Mistral model. Only needed if Claude API is unavailable."""
    raise NotImplementedError(
        "Mistral fallback not configured. Set SYNTHESIS_MODEL='claude' and provide ANTHROPIC_API_KEY."
    )
