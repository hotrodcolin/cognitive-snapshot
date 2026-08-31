"""Tool vs. No-Tool baseline comparison.

Compares the full Cognitive Snapshot Agent pipeline (structured tools + synthesis)
against raw LLM analysis to demonstrate the value of the agentic architecture.
"""

from __future__ import annotations

from typing import Any

from agent.orchestrator import analyze
from config import generate_synthesis


RAW_LLM_PROMPT = """Analyze the following text and describe the writer's cognitive state. Consider their emotional tone, temporal focus, reasoning complexity, social orientation, and overall coherence of thought.

Text: {text}

Provide a detailed analysis of the writer's cognitive state."""


def run_tool_vs_notool(
    text: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare full pipeline vs raw LLM on the same text."""
    if text is None:
        text = (
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

    results = {}

    # Method A: Full pipeline (tool-based)
    if verbose:
        print(f"\n{'='*70}")
        print("METHOD A: Full Cognitive Snapshot Agent Pipeline")
        print(f"{'='*70}")

    pipeline_result = analyze(text, prompt_style=2)
    results["pipeline"] = pipeline_result

    if verbose and pipeline_result.get("status") == "success":
        print(f"\nScores:")
        for dim, score in pipeline_result["scores"].items():
            bar = "#" * int(score * 20)
            print(f"  {dim:25s} [{bar:<20s}] {score:.3f}")
        print(f"\nSynthesis (grounded in tool data):")
        print(pipeline_result["synthesis"])

    # Method B: Raw LLM (no tools)
    if verbose:
        print(f"\n{'='*70}")
        print("METHOD B: Raw LLM Analysis (no tools)")
        print(f"{'='*70}")

    try:
        raw_prompt = RAW_LLM_PROMPT.format(text=text)
        raw_response = generate_synthesis(raw_prompt, temperature=0.7, max_tokens=500)
        results["raw_llm"] = raw_response

        if verbose:
            print(f"\n{raw_response}")
    except Exception as e:
        results["raw_llm"] = f"[Error: {e}]"
        if verbose:
            print(f"\nRaw LLM call failed: {e}")

    # Comparison notes
    if verbose:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print("The pipeline approach provides:")
        print("  - Quantified, comparable scores across dimensions")
        print("  - Specific evidence sentences for each claim")
        print("  - Visual representation (radar chart)")
        print("  - Grounded synthesis that cannot hallucinate beyond tool data")
        print("  - Transparent reasoning (which tools were selected and why)")
        print("\nThe raw LLM approach provides:")
        print("  - Narrative interpretation without quantified backing")
        print("  - No visual output")
        print("  - Risk of hallucination or vague impressions")
        print("  - No reproducibility across runs")

    return results
