"""Radar chart and visualization helpers for the cognitive snapshot."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


def create_radar_chart(
    scores: dict[str, float],
    title: str = "Cognitive Snapshot",
) -> go.Figure:
    """Create a 5-axis radar chart from normalized scores (0-1).

    Returns a Plotly figure that Gradio can render directly.
    """
    if not scores:
        scores = {
            "Emotional Intensity": 0,
            "Temporal Breadth": 0,
            "Cognitive Complexity": 0,
            "Social Orientation": 0,
            "Coherence": 0,
        }

    categories = list(scores.keys())
    values = list(scores.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.25)",
        line=dict(color="rgba(99, 110, 250, 0.9)", width=2.5),
        marker=dict(size=8, color="rgba(99, 110, 250, 1)"),
        name="Snapshot",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.2", "0.4", "0.6", "0.8", "1.0"],
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.15)",
                tickfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.15)",
                tickfont=dict(size=12, color="rgba(255,255,255,0.85)"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=dict(
            text=title,
            font=dict(size=16, color="white"),
            x=0.5,
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=60, b=40),
        height=400,
    )

    return fig


def create_sentiment_arc_chart(sentence_scores: list[float]) -> go.Figure:
    """Create a line chart showing the emotional arc across sentences."""
    if not sentence_scores:
        return go.Figure()

    fig = go.Figure()
    x = list(range(1, len(sentence_scores) + 1))

    fig.add_trace(go.Scatter(
        x=x,
        y=sentence_scores,
        mode="lines+markers",
        line=dict(color="rgba(239, 85, 59, 0.9)", width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(239, 85, 59, 0.15)",
        name="Sentiment",
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        xaxis_title="Sentence",
        yaxis_title="Sentiment (-1 to 1)",
        yaxis=dict(range=[-1.1, 1.1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        margin=dict(l=50, r=20, t=30, b=40),
        height=250,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis_gridcolor="rgba(255,255,255,0.1)",
        showlegend=False,
    )
    return fig


def create_temporal_bar(distribution: dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart for temporal distribution."""
    labels = ["Past", "Present", "Future"]
    values = [
        distribution.get("past", 0),
        distribution.get("present", 0),
        distribution.get("future", 0),
    ]
    colors = ["#636EFA", "#EF553B", "#AB63FA"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="auto",
        textfont=dict(color="white"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        margin=dict(l=60, r=20, t=10, b=20),
        height=150,
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        showlegend=False,
    )
    return fig
