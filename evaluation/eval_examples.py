"""Run the full pipeline on 3+ diverse text samples and display results.

Used for the qualitative evaluation section of the notebook.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from agent.orchestrator import analyze
from ui.charts import create_radar_chart


# ---------------------------------------------------------------------------
# Three diverse evaluation samples
# ---------------------------------------------------------------------------
EVAL_SAMPLES = {
    "emotional": {
        "label": "Sample 1: Emotional / Personal Reflection",
        "text": (
            "I keep going back to that conversation we had last month. I remember "
            "feeling completely blindsided when she told me she was leaving the team. "
            "We had built something really good together, and now I wonder if I could "
            "have done something differently. Maybe if I had been more attentive to "
            "the signs, I would have seen it coming. But at the same time, I recognize "
            "that people need to pursue their own paths. I hope she finds what she is "
            "looking for, even though it leaves a gap in our work that will be hard to "
            "fill. Today I am trying to focus on what we can build going forward, but "
            "my mind keeps drifting back. I think the hardest part is not knowing "
            "whether I failed as a leader or whether this was simply inevitable. "
            "Sometimes I catch myself replaying moments, wondering if a different "
            "word or gesture might have changed things."
        ),
        "expectations": (
            "Expect: moderate negative sentiment, past-oriented, high self-focus, "
            "moderate complexity (reflection involves some nuance), moderate coherence."
        ),
    },
    "intellectual": {
        "label": "Sample 2: Intellectual / Analytical",
        "text": (
            "The relationship between cognitive load and decision quality fascinates me. "
            "When we are under pressure, our reasoning narrows and we tend to fall back "
            "on heuristics rather than careful analysis. However, some research suggests "
            "that moderate stress actually enhances certain types of pattern recognition. "
            "This creates a paradox: the conditions that degrade analytical thinking may "
            "simultaneously improve intuitive judgment. I have been reading about dual-process "
            "theory and how System 1 and System 2 thinking interact under various conditions. "
            "The implications for organizational decision-making are significant. If we could "
            "design environments that harness the benefits of moderate arousal while protecting "
            "against the costs of cognitive overload, we might achieve consistently better "
            "outcomes. Furthermore, the individual differences in stress response complicate "
            "any one-size-fits-all approach. Nevertheless, the emerging picture is encouraging "
            "and suggests practical applications for leadership training and team design."
        ),
        "expectations": (
            "Expect: neutral/positive sentiment, present-oriented, high cognitive "
            "complexity, low self-focus, high coherence."
        ),
    },
    "fragmented": {
        "label": "Sample 3: Fragmented / Stream of Consciousness",
        "text": (
            "Woke up at 3am again and could not get back to sleep. Something about that "
            "email from yesterday keeps nagging at me. I should probably call my mom this "
            "weekend. The weather has been strange lately, warm then cold then warm again. "
            "I wonder if the project deadline will get pushed back. My back hurts from "
            "sitting too much. Need to remember to buy groceries. That podcast about "
            "consciousness was wild, they were talking about how the brain creates an "
            "illusion of continuity. Speaking of illusions, I still cannot believe the "
            "election results. Maybe I should start meditating again. The dog needs to "
            "go to the vet soon. I have been feeling scattered lately, like my attention "
            "is spread too thin across too many things. Sometimes I worry that I am "
            "losing the ability to focus deeply on anything."
        ),
        "expectations": (
            "Expect: mixed/negative sentiment, mixed temporal orientation, low "
            "complexity, high self-focus, low coherence (fragmented)."
        ),
    },
}


def run_evaluation(verbose: bool = True) -> dict[str, dict]:
    """Run all evaluation samples and return results.

    If verbose, prints formatted output for each sample.
    """
    results = {}

    for key, sample in EVAL_SAMPLES.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"{sample['label']}")
            print(f"{'='*70}")
            print(f"Expectations: {sample['expectations']}\n")

        result = analyze(sample["text"], prompt_style=2)
        results[key] = result

        if verbose and result.get("status") == "success":
            print(f"Content type: {result['triage']['content_type']}")
            print(f"Tools run: {list(result['tool_results'].keys())}")
            print(f"\nScores:")
            for dim, score in result["scores"].items():
                bar = "#" * int(score * 20)
                print(f"  {dim:25s} [{bar:<20s}] {score:.3f}")
            print(f"\nSynthesis: {result['synthesis'][:300]}...")
            print(f"\nElapsed: {result['elapsed_seconds']}s")
        elif verbose:
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message', 'N/A')}")

    return results


def results_to_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Convert evaluation results to a summary DataFrame."""
    rows = []
    for key, result in results.items():
        if result.get("status") != "success":
            continue
        scores = result["scores"]
        row = {
            "Sample": key.capitalize(),
            "Content Type": result["triage"].get("content_type", "N/A"),
            "Emotional Intensity": scores.get("Emotional Intensity", 0),
            "Temporal Breadth": scores.get("Temporal Breadth", 0),
            "Cognitive Complexity": scores.get("Cognitive Complexity", 0),
            "Social Orientation": scores.get("Social Orientation", 0),
            "Coherence": scores.get("Coherence", 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)
