"""Compare 3 prompt styles on the same text samples.

Tests whether the role-based prompt produces more thoughtful tool selection,
whether the minimal prompt misses relevant dimensions, and whether the
structured prompt is more consistent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent.triage import triage


COMPARISON_TEXT = (
    "I keep going back to that conversation we had last month. I remember "
    "feeling completely blindsided when she told me she was leaving the team. "
    "We had built something really good together, and now I wonder if I could "
    "have done something differently. Maybe if I had been more attentive to "
    "the signs, I would have seen it coming. But at the same time, I recognize "
    "that people need to pursue their own paths. I hope she finds what she is "
    "looking for, even though it leaves a gap in our work that will be hard to "
    "fill. Today I am trying to focus on what we can build going forward, but "
    "my mind keeps drifting back. I think the hardest part is not knowing "
    "whether I failed as a leader or whether this was simply inevitable."
)

PROMPT_LABELS = {
    1: "Minimal",
    2: "Structured",
    3: "Role-Based (Cognitive Psychologist)",
}


def run_prompt_comparison(
    texts: list[tuple[str, str]] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run all 3 prompt styles on the given texts.

    Returns a list of comparison dicts with results for each style.
    """
    if texts is None:
        texts = [("Reflective / Emotional", COMPARISON_TEXT)]

    results = []

    for text_label, text in texts:
        comparison = {"text_label": text_label, "styles": {}}

        if verbose:
            print(f"\n{'='*70}")
            print(f"Text: {text_label}")
            print(f"{'='*70}")

        for style in [1, 2, 3]:
            result = triage(text, prompt_style=style)
            comparison["styles"][style] = result

            if verbose:
                print(f"\n--- Prompt Style {style}: {PROMPT_LABELS[style]} ---")
                print(f"  Content type: {result.get('content_type', 'N/A')}")
                print(f"  Tools selected: {result.get('selected_tools', [])}")
                print(f"  Reasoning: {result.get('reasoning', 'N/A')[:200]}")
                if result.get("error"):
                    print(f"  Error: {result['error']}")

        results.append(comparison)

    if verbose:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        for comp in results:
            print(f"\nText: {comp['text_label']}")
            for style in [1, 2, 3]:
                r = comp["styles"][style]
                tools = r.get("selected_tools", [])
                print(f"  Style {style} ({PROMPT_LABELS[style]:30s}): "
                      f"{len(tools)} tools -> {tools}")

    return results


def comparison_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert prompt comparison results to a DataFrame."""
    rows = []
    for comp in results:
        for style in [1, 2, 3]:
            r = comp["styles"][style]
            rows.append({
                "Sample": comp["text_label"].split(":")[0] if ":" in comp["text_label"] else comp["text_label"],
                "Prompt Style": f"{style} ({PROMPT_LABELS[style]})",
                "Content Type": r.get("content_type", "N/A"),
                "Tools Selected": ", ".join(r.get("selected_tools", [])),
                "Num Tools": len(r.get("selected_tools", [])),
            })
    return pd.DataFrame(rows)
