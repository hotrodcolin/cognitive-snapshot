"""Compare temperature and max_tokens settings on synthesis quality.

Tests temperature (0.3, 0.7, 1.0) and max_tokens (100, 250, 500)
on the same tool outputs to isolate the effect of generation parameters.
"""

from __future__ import annotations

from typing import Any

from agent.orchestrator import analyze
from agent.synthesis import synthesize


def run_temperature_comparison(
    text: str,
    tool_results: dict[str, dict],
    verbose: bool = True,
) -> dict[float, str]:
    """Run synthesis at three temperature settings on the same tool results."""
    temperatures = [0.3, 0.7, 1.0]
    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print("TEMPERATURE COMPARISON")
        print(f"{'='*70}")

    for temp in temperatures:
        narrative = synthesize(text, tool_results, temperature=temp, max_tokens=300)
        results[temp] = narrative

        if verbose:
            print(f"\n--- Temperature {temp} ---")
            print(narrative)
            print(f"({len(narrative.split())} words)")

    return results


def run_max_tokens_comparison(
    text: str,
    tool_results: dict[str, dict],
    verbose: bool = True,
) -> dict[int, str]:
    """Run synthesis at three max_tokens settings on the same tool results."""
    token_limits = [100, 250, 500]
    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print("MAX TOKENS COMPARISON")
        print(f"{'='*70}")

    for tokens in token_limits:
        narrative = synthesize(text, tool_results, temperature=0.7, max_tokens=tokens)
        results[tokens] = narrative

        if verbose:
            print(f"\n--- max_tokens={tokens} ---")
            print(narrative)
            print(f"({len(narrative.split())} words)")

    return results


def run_param_comparison(verbose: bool = True) -> dict[str, Any]:
    """Run both parameter comparisons on a sample text.

    First runs the pipeline to get tool results, then tests parameters
    on the synthesis step only.
    """
    text = (
        "I keep going back to that conversation we had last month. I remember "
        "feeling completely blindsided when she told me she was leaving the team. "
        "We had built something really good together, and now I wonder if I could "
        "have done something differently. Maybe if I had been more attentive to "
        "the signs, I would have seen it coming. But at the same time, I recognize "
        "that people need to pursue their own paths. I hope she finds what she is "
        "looking for, even though it leaves a gap in our work that will be hard to "
        "fill. Today I am trying to focus on what we can build going forward, but "
        "my mind keeps drifting back."
    )

    # Run tools once
    result = analyze(text, prompt_style=2)
    if result.get("status") != "success":
        print(f"Pipeline failed: {result.get('message', 'unknown error')}")
        return {}

    tool_results = result["tool_results"]

    temp_results = run_temperature_comparison(text, tool_results, verbose=verbose)
    tokens_results = run_max_tokens_comparison(text, tool_results, verbose=verbose)

    return {
        "temperature": temp_results,
        "max_tokens": tokens_results,
        "tool_results": tool_results,
    }
