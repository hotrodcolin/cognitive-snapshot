"""Sentiment tool validation on Pennebaker essays.

Runs sentiment analysis on 50 random essays and reports distribution,
spread, and internal consistency to provide quantitative evidence
that the tool produces meaningful, varied outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from tools.sentiment import analyze_sentiment


def run_sentiment_validation(
    n_samples: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run sentiment analysis on n_samples Pennebaker essays.

    Returns a DataFrame with per-essay results and prints summary stats.
    """
    # Load and filter dataset
    dataset = load_dataset("jingjietan/essays-big5", split="train")
    texts = [" ".join(row["text"].split()) for row in dataset if len(row["text"].split()) >= 100]

    # Sample
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(texts), size=min(n_samples, len(texts)), replace=False)
    samples = [texts[i] for i in indices]

    # Run sentiment on each
    rows = []
    for i, text in enumerate(samples):
        result = analyze_sentiment(text)
        rows.append({
            "essay_idx": int(indices[i]),
            "word_count": len(text.split()),
            "overall_sentiment": result["overall_sentiment"],
            "overall_score": result["overall_score"],
            "arc_direction": result["arc_direction"],
            "n_sentences": len(result["sentence_scores"]),
            "score_std": float(np.std(result["sentence_scores"])) if result["sentence_scores"] else 0.0,
        })

    df = pd.DataFrame(rows)

    if verbose:
        print(f"Sentiment Validation: {len(df)} essays analyzed")
        print(f"{'='*50}")

        # Distribution across categories
        dist = df["overall_sentiment"].value_counts()
        print(f"\nCategory distribution:")
        for cat in ["negative", "neutral", "positive"]:
            count = dist.get(cat, 0)
            pct = count / len(df) * 100
            print(f"  {cat:10s}: {count:3d} ({pct:.1f}%)")

        # Score statistics
        print(f"\nScore statistics:")
        print(f"  Mean:   {df['overall_score'].mean():+.3f}")
        print(f"  Std:    {df['overall_score'].std():.3f}")
        print(f"  Min:    {df['overall_score'].min():+.3f}")
        print(f"  Max:    {df['overall_score'].max():+.3f}")
        print(f"  Spread: {df['overall_score'].max() - df['overall_score'].min():.3f}")

        # Arc direction distribution
        arc_dist = df["arc_direction"].value_counts()
        print(f"\nArc direction distribution:")
        for arc in ["stable", "improving", "declining", "volatile"]:
            count = arc_dist.get(arc, 0)
            print(f"  {arc:12s}: {count:3d}")

        # Internal consistency: correlation between word count and score variance
        corr = df["word_count"].corr(df["score_std"])
        print(f"\nWord count vs. score std correlation: {corr:.3f}")
        print("(Positive = longer essays show more emotional variation)")

    return df
