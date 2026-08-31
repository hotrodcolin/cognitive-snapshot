"""Gradio interface for the Cognitive Snapshot Agent.

Uses gr.Blocks() for full layout control. Dark theme, professional presentation.
Two-step output: concise snapshot then drill-down card on request.
Progressive reveal of analysis steps via generator function.
"""

from __future__ import annotations

import json
import time
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from agent.triage import triage
from agent.synthesis import synthesize, generate_drilldown
from config import DEFAULT_TOOLS, MIN_TEXT_LENGTH, generate_synthesis
from tools import TOOL_REGISTRY
from agent.orchestrator import _extract_scores
from ui.charts import create_radar_chart
from ui.text_annotator import annotate_text

# ---------------------------------------------------------------------------
# Sample texts with labels
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "Reflective": (
        "I keep going back to that conversation we had last month. I remember feeling "
        "completely blindsided when she told me she was leaving the team. We had built "
        "something really good together, and now I wonder if I could have done something "
        "differently. Maybe if I had been more attentive to the signs, I would have seen "
        "it coming. But at the same time, I recognize that people need to pursue their "
        "own paths. I hope she finds what she is looking for, even though it leaves a gap "
        "in our work that will be hard to fill. Today I am trying to focus on what we can "
        "build going forward, but my mind keeps drifting back. I think the hardest part "
        "is not knowing whether I failed as a leader or whether this was simply inevitable. "
        "Sometimes I catch myself replaying moments, wondering if a different word or "
        "gesture might have changed things."
    ),
    "Stressed": (
        "I cannot keep up with everything right now. The project deadline moved up by two "
        "weeks and nobody told me until yesterday. I have three reports due, two meetings "
        "that could have been emails, and I still have not finished the analysis from last "
        "sprint. My inbox is at 200 unread. I keep starting things and not finishing them "
        "because something more urgent pops up. I skipped lunch again today. I know this "
        "pace is not sustainable but I do not see how to slow down without dropping balls. "
        "Every time I think I am getting ahead, another request comes in. I feel like I am "
        "running on a treadmill that keeps speeding up. The worst part is I used to be good "
        "at managing my time. I do not know what changed. Maybe the workload doubled and I "
        "did not notice because it happened gradually."
    ),
    "Energized": (
        "Something clicked in today's brainstorm session and I cannot stop thinking about "
        "it. We were going back and forth about the user onboarding flow and then Marcus "
        "said something about treating it like a conversation instead of a form and it just "
        "unlocked everything. I spent the rest of the afternoon sketching out wireframes and "
        "the ideas kept coming. I think we might be onto something genuinely new here, not "
        "just a reskin of what everyone else does. The team energy was electric. Even Jamie, "
        "who is usually the skeptic, was leaning in and building on ideas. I want to prototype "
        "this before the momentum fades. I already pinged the engineering lead to see if we "
        "can get a spike on the calendar for next week. I have not felt this excited about "
        "work in months. This is exactly the kind of creative problem-solving I got into "
        "this field for. Tomorrow I am going to write up the concept and share it with the "
        "broader team. I think when people see it they will get it immediately. This could "
        "be the thing that differentiates our product. I keep smiling about it, which is "
        "a good sign. When the work makes you smile, you know you are on the right track. "
        "I am already thinking about edge cases and that is usually a sign my brain believes "
        "in the idea enough to stress-test it."
    ),
}

# Store the latest analysis result for annotation switching and drill-down
_latest_state = [None]

# ---------------------------------------------------------------------------
# Sample pairs for Compare tab (cycles on repeated clicks)
# ---------------------------------------------------------------------------
COMPARE_PAIRS = [
    # Pair 1: Career setback → three months later, new direction
    (
        "I got the call this morning and I still feel like the ground shifted under me. "
        "They eliminated my position. Seven years and they eliminated my position. I keep "
        "replaying the conversation with HR trying to figure out if I missed warning signs. "
        "Was it the reorg last quarter? Should I have seen this coming? I sat in the parking "
        "lot for twenty minutes before I could drive home. I keep picking up my phone to "
        "check work email and then remembering there is no work email anymore. My stomach "
        "hurts. I should probably eat something but I cannot imagine eating right now. I "
        "keep thinking about what I am going to tell people. I feel embarrassed which is "
        "stupid because it was not my fault but I feel it anyway. The house is too quiet. "
        "I do not know what I am supposed to do tomorrow morning.",

        "Had coffee with Dana this morning and told her about the workshop idea and she "
        "immediately got it. She wants to co-facilitate the first one. I have been sketching "
        "out the curriculum all week and it is coming together faster than I expected. Turns "
        "out all those years managing cross-functional teams gave me a framework I did not "
        "even realize I had. I am meeting with the community center on Thursday about space. "
        "I want to start small, maybe eight people, really interactive. I have been waking "
        "up early without an alarm which has not happened in years. There is something about "
        "building your own thing that feels completely different from working inside someone "
        "else's structure. I still have anxious moments about money but the consulting gigs "
        "are covering the basics and this feels like it could turn into something real. I am "
        "going to send the first draft of the website to Marcus tonight.",
    ),
    # Pair 2: High-energy collaboration → solo reflective evening
    (
        "That sprint planning session was incredible. Priya came in with the API redesign "
        "and then Marcus built on it and before we knew it we had completely rethought the "
        "onboarding flow. Everyone was talking over each other in the best way. I love when "
        "a team hits that rhythm where ideas just bounce and get better with each pass. We "
        "filled three whiteboards. Jin was sketching wireframes in real time while we talked. "
        "I think we cracked the retention problem we have been stuck on for months. The energy "
        "was electric. Even the PM was grinning which never happens in sprint planning. We "
        "are going to prototype the top three ideas this week. I already pinged the design "
        "lead to block time tomorrow. This is what good work feels like. This is why I got "
        "into this field.",

        "The apartment is quiet tonight. Everyone else went out but I wanted to just sit with "
        "my thoughts for a while. I have been running so fast lately I am not sure what I "
        "actually think about any of it. The project is going well I think but sometimes I "
        "wonder if I am just performing enthusiasm because the team expects it. Am I actually "
        "excited or am I excited because everyone else is excited? I used to know the answer "
        "to that question immediately. My mom called earlier and I let it go to voicemail "
        "which I feel guilty about. I should call her this weekend. I have been reading this "
        "book about attention and the author says we confuse busyness with meaning. That "
        "landed hard. The tea is getting cold. I should probably go to bed but my mind is "
        "doing that thing where it jumps between topics. The rain sounds nice though.",
    ),
    # Pair 3: Pre-presentation anxiety → post-presentation relief
    (
        "The presentation is tomorrow and I am not ready. I keep going over my slides and "
        "changing things and then changing them back. What if they ask about the methodology "
        "and I blank? I practiced in front of the mirror and I sounded robotic. I hate that "
        "I care this much about what people think but I do. My hands are already getting "
        "clammy just thinking about it. I rewrote the opening three times today. Sarah said "
        "it was fine two hours ago but fine is not good. I need it to be good. I keep "
        "imagining the worst version where I lose my train of thought midsentence and just "
        "stand there. I should probably stop looking at it and get some sleep but every time "
        "I close my laptop I think of something I should fix. Maybe I will run through it "
        "one more time. Just one more time.",

        "I cannot believe how well that went. The whole room was nodding during the "
        "methodology section which was the part I was most worried about. Dr. Chen asked a "
        "tough question about sample size and I actually had a good answer because I had "
        "thought about it so much beforehand. The anxiety was worth something after all. "
        "A couple of people came up afterward and said it was one of the clearest "
        "presentations they had seen on this topic. I am trying not to let it go to my head "
        "but honestly I am just relieved. I feel lighter than I have in weeks. I am sitting "
        "outside with a coffee and the sun is warm and I do not have to think about slides "
        "anymore. I want to remember this feeling the next time I am spiraling the night "
        "before something. The preparation mattered but the worry did not help as much as I "
        "thought it would. I think I might actually be getting better at this.",
    ),
]

_compare_pair_index = [0]


def _load_sample_pair():
    """Return the next sample pair, cycling through the list."""
    idx = _compare_pair_index[0]
    pair = COMPARE_PAIRS[idx]
    _compare_pair_index[0] = (idx + 1) % len(COMPARE_PAIRS)
    return pair[0], pair[1]

# ---------------------------------------------------------------------------
# Dimension card config
# ---------------------------------------------------------------------------
CARD_CONFIG = {
    "sentiment": {
        "label": "Emotional Intensity",
        "score_key": "Emotional Intensity",
        "detail": lambda r: f"{r.get('overall_sentiment', 'N/A')} (arc: {r.get('arc_direction', 'N/A')})",
        "evidence_key": "evidence",
        "color": "#EF553B",
    },
    "temporal": {
        "label": "Temporal Orientation",
        "score_key": "Temporal Breadth",
        "detail": lambda r: f"Dominant: {r.get('dominant_orientation', 'N/A')}",
        "evidence_key": None,
        "color": "#AB63FA",
    },
    "complexity": {
        "label": "Cognitive Complexity",
        "score_key": "Cognitive Complexity",
        "detail": lambda r: f"Nuance ratio: {r.get('nuance_ratio', 'N/A')}, avg sentence: {r.get('avg_sentence_length', 'N/A')} words",
        "evidence_key": "evidence",
        "color": "#FECB52",
    },
    "social": {
        "label": "Social Orientation",
        "score_key": "Social Orientation",
        "detail": lambda r: f"Self: {r['pronoun_distribution']['self_focus']:.0%}, Group: {r['pronoun_distribution']['group_identity']:.0%}, Other: {r['pronoun_distribution']['social_reference']:.0%} | {r['motivation_framing']['dominant']} framing",
        "evidence_key": None,
        "color": "#00CC96",
    },
    "coherence": {
        "label": "Coherence",
        "score_key": "Coherence",
        "detail": lambda r: f"Pattern: {r.get('pattern', 'N/A')} (consistency: {r.get('consistency', 'N/A')})",
        "evidence_key": None,
        "color": "#636EFA",
    },
}

TOOL_DISPLAY_NAMES = {
    "sentiment": "Sentiment & Emotional Tone",
    "temporal": "Temporal Orientation",
    "complexity": "Cognitive Complexity",
    "social": "Self vs. Social Reference",
    "coherence": "Coherence & Fragmentation",
}


def _build_single_card(tool_name: str, tool_result: dict, score: float) -> str:
    """Build HTML for a single dimension card."""
    config = CARD_CONFIG[tool_name]
    bar_pct = int(score * 100)
    detail = ""
    try:
        detail = config["detail"](tool_result)
    except Exception:
        pass

    evidence = ""
    if config["evidence_key"] and config["evidence_key"] in tool_result:
        evs = tool_result[config["evidence_key"]][:1]
        if evs:
            evidence = f'<div style="font-size: 11px; color: #aaa; margin-top: 4px; font-style: italic;">"{evs[0][:100]}..."</div>'

    return f'''<div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 14px; border-left: 3px solid {config["color"]};">
        <div style="font-weight: 600; font-size: 13px; color: #eee;">{config["label"]}</div>
        <div style="margin: 6px 0; background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="width: {bar_pct}%; height: 100%; background: {config["color"]}; border-radius: 4px;"></div>
        </div>
        <div style="font-size: 12px; color: #ccc;">{detail}</div>
        {evidence}
    </div>'''


def _format_dimension_cards(tool_results: dict, scores: dict) -> str:
    """Format all completed tool results as HTML dimension cards."""
    cards_html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">'
    for tool_name in DEFAULT_TOOLS:
        if tool_name in tool_results:
            score = scores.get(CARD_CONFIG[tool_name]["score_key"], 0)
            cards_html += _build_single_card(tool_name, tool_results[tool_name], score)
    cards_html += "</div>"
    return cards_html


def _format_agent_log(triage_result: dict, tools_run: list, elapsed: float, errors: dict) -> str:
    """Format the agent reasoning log."""
    lines = [
        f"**Content type:** {triage_result.get('content_type', 'N/A')}",
        f"**Triage tool selection:** {', '.join(triage_result.get('selected_tools', []))}",
        f"**Tools executed:** {', '.join(tools_run)} (all 5 always run for consistent comparison)",
        f"**Reasoning:** {triage_result.get('reasoning', 'N/A')}",
        f"**Elapsed:** {elapsed}s",
    ]
    if errors:
        lines.append(f"**Errors:** {json.dumps(errors)}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Progressive analysis generator (two-step: snapshot then drill-down offer)
# ---------------------------------------------------------------------------
DEMO_PACE = 0.5  # seconds between tool completions for watchable demo


def run_analysis_progressive(text: str):
    """Generator that yields progressive UI updates as each step completes.

    Outputs: (radar, cards, synthesis, drilldown_offer_visible, drilldown_card, annotated, agent_log)
    """
    empty_fig = create_radar_chart({})
    # 7-tuple: radar, cards, synthesis, drilldown_row visibility, drilldown_card, annotated, agent_log
    EMPTY = (empty_fig, "", "", gr.update(visible=False), "", "", "")

    if not text or not text.strip():
        yield (empty_fig, "", "Please enter some text.", gr.update(visible=False), "", "", "")
        return

    word_count = len(text.split())
    if word_count < MIN_TEXT_LENGTH:
        msg = f"Please provide at least {MIN_TEXT_LENGTH} words. You provided {word_count}."
        yield (empty_fig, "", msg, gr.update(visible=False), "", "", "")
        return

    start_time = time.time()

    # Step 1: Triage
    yield (empty_fig, "", "*Analyzing input... classifying content type...*", gr.update(visible=False), "", "", "")

    triage_result = triage(text, prompt_style=2)

    triage_msg = (
        f"**Content type:** {triage_result.get('content_type', 'N/A')}\n\n"
        f"**Reasoning:** {triage_result.get('reasoning', 'N/A')[:200]}\n\n"
        f"*Running analysis instruments...*"
    )
    yield (empty_fig, "", triage_msg, gr.update(visible=False), "", "", "")

    # Step 2: Run tools one by one with progressive radar updates
    tool_results = {}
    tool_errors = {}

    for tool_name in DEFAULT_TOOLS:
        if tool_name not in TOOL_REGISTRY:
            continue
        try:
            tool_results[tool_name] = TOOL_REGISTRY[tool_name](text)
        except Exception as e:
            tool_errors[tool_name] = str(e)

        # Partial radar and cards after each tool
        partial_scores = _extract_scores(tool_results)
        # Fill missing dimensions with 0 so radar has all 5 axes
        all_scores = {
            "Emotional Intensity": 0, "Temporal Breadth": 0,
            "Cognitive Complexity": 0, "Social Orientation": 0, "Coherence": 0,
        }
        all_scores.update(partial_scores)
        partial_radar = create_radar_chart(all_scores)
        cards = _format_dimension_cards(tool_results, partial_scores)
        tool_display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        progress = f"*{tool_display} complete ({len(tool_results)}/5)...*"
        yield (partial_radar, cards, progress, gr.update(visible=False), "", "", "")

        time.sleep(DEMO_PACE)

    # Final radar chart with all scores
    scores = _extract_scores(tool_results)
    radar = create_radar_chart(scores)
    cards = _format_dimension_cards(tool_results, scores)
    yield (radar, cards, "*Generating snapshot...*", gr.update(visible=False), "", "", "")

    # Step 3: Synthesis (Step 1 output — concise snapshot)
    synthesis_narrative = synthesize(text, tool_results, triage_result=triage_result)

    # Annotated text
    annotated = annotate_text(text, tool_results, "sentiment")

    elapsed = round(time.time() - start_time, 2)
    agent_log = _format_agent_log(triage_result, list(tool_results.keys()), elapsed, tool_errors)

    # Store state for drill-down and annotation switching
    _latest_state[0] = {
        "status": "success",
        "text": text,
        "tool_results": tool_results,
        "scores": scores,
        "triage": triage_result,
        "synthesis": synthesis_narrative,
    }

    # Show synthesis. If API failed (empty synthesis), don't show drill-down offer.
    if synthesis_narrative:
        drilldown_offer = gr.update(visible=True)
        synthesis_display = synthesis_narrative
    else:
        drilldown_offer = gr.update(visible=False)
        synthesis_display = "*Synthesis unavailable — tool results and scores are shown above.*"

    yield (radar, cards, synthesis_display, drilldown_offer, "", annotated, agent_log)


def run_drilldown():
    """Generate the Step 2 drill-down card from stored state."""
    state = _latest_state[0]
    if state is None or state.get("status") != "success":
        return "*No analysis available. Run a snapshot first.*"

    if not state.get("synthesis"):
        return "*Drill-down unavailable — synthesis step did not complete.*"

    drilldown = generate_drilldown(
        text=state["text"],
        tool_results=state["tool_results"],
        step1_synthesis=state["synthesis"],
    )
    if drilldown:
        return drilldown
    return "*Drill-down unavailable — API call did not complete.*"


def update_annotation(text: str, dimension: str) -> str:
    """Update text annotation when dimension changes."""
    state = _latest_state[0]
    if state is None or state.get("status") != "success":
        return text
    return annotate_text(text, state["tool_results"], dimension)


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------
def run_comparison(text_a: str, text_b: str):
    """Run both texts through the pipeline and produce comparison output."""
    from agent.orchestrator import analyze

    empty = create_radar_chart({})

    if not text_a.strip() or not text_b.strip():
        yield (empty, "", "Please provide text in both fields.")
        return

    yield (empty, "", "*Analyzing Snapshot A...*")

    result_a = analyze(text_a, prompt_style=2)
    yield (empty, "", "*Analyzing Snapshot B...*")

    result_b = analyze(text_b, prompt_style=2)

    if result_a.get("status") != "success" or result_b.get("status") != "success":
        msg_a = result_a.get("message", "")
        msg_b = result_b.get("message", "")
        yield (empty, "", f"Analysis failed. A: {msg_a} B: {msg_b}")
        return

    # Overlaid radar chart
    scores_a = result_a["scores"]
    scores_b = result_b["scores"]

    cats = list(scores_a.keys())
    vals_a = [scores_a.get(c, 0) for c in cats]
    vals_b = [scores_b.get(c, 0) for c in cats]
    cats_closed = cats + [cats[0]]
    vals_a_closed = vals_a + [vals_a[0]]
    vals_b_closed = vals_b + [vals_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)", line=dict(color="rgba(99, 110, 250, 0.9)", width=2.5),
        marker=dict(size=7), name="Snapshot A",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(239, 85, 59, 0.2)", line=dict(color="rgba(239, 85, 59, 0.9)", width=2.5),
        marker=dict(size=7), name="Snapshot B",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.15)",
                           tickfont=dict(size=10, color="rgba(255,255,255,0.6)")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.15)",
                            tickfont=dict(size=12, color="rgba(255,255,255,0.85)")),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), showlegend=True,
        legend=dict(font=dict(color="white")),
        margin=dict(l=60, r=60, t=40, b=40), height=420,
    )

    # Score difference table
    diff_html = '<table style="width:100%; color: #ccc; font-size: 13px; border-collapse: collapse;">'
    diff_html += '<tr style="border-bottom: 1px solid #333;"><th style="text-align:left; padding:6px;">Dimension</th><th>A</th><th>B</th><th>Delta</th></tr>'
    for dim in cats:
        a_val = scores_a.get(dim, 0)
        b_val = scores_b.get(dim, 0)
        delta = b_val - a_val
        arrow = "+" if delta > 0 else ""
        color = "#2ecc71" if abs(delta) < 0.1 else ("#EF553B" if delta < 0 else "#636EFA")
        diff_html += f'<tr><td style="padding:4px 6px;">{dim}</td><td style="text-align:center;">{a_val:.2f}</td><td style="text-align:center;">{b_val:.2f}</td><td style="text-align:center; color:{color};">{arrow}{delta:.2f}</td></tr>'
    diff_html += '</table>'

    yield (fig, diff_html, "*Generating differential analysis...*")

    # Differential analysis from Claude
    diff_prompt = (
        "You are a cognitive psychologist comparing two cognitive snapshots from the same person "
        "taken at different times.\n\n"
        f"Snapshot A scores: {json.dumps(scores_a)}\n"
        f"Snapshot A synthesis: {result_a['synthesis']}\n\n"
        f"Snapshot B scores: {json.dumps(scores_b)}\n"
        f"Snapshot B synthesis: {result_b['synthesis']}\n\n"
        "Compare these two cognitive states. What changed? What does the change suggest about "
        "the person's mental state evolution? Be specific about which dimensions shifted and "
        "what those shifts typically indicate. Keep it to 4-6 sentences."
    )
    try:
        diff_narrative = generate_synthesis(diff_prompt, temperature=0.7, max_tokens=400)
    except Exception as e:
        diff_narrative = f"[Differential analysis unavailable: {e}]"

    yield (fig, diff_html, diff_narrative)


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks app."""

    theme = gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#111318",
        body_background_fill_dark="#111318",
        block_background_fill="#1a1d24",
        block_background_fill_dark="#1a1d24",
        block_border_color="#2a2d34",
        block_border_color_dark="#2a2d34",
        input_background_fill="#1a1d24",
        input_background_fill_dark="#1a1d24",
        button_primary_background_fill="#636EFA",
        button_primary_background_fill_dark="#636EFA",
    )

    with gr.Blocks(theme=theme, title="Cognitive Snapshot Agent", css="""
        .gradio-container { max-width: 1100px !important; }
    """) as app:

        gr.Markdown(
            "# Cognitive Snapshot Agent\n"
            "*An instrument for mapping cognitive state through language analysis*",
        )

        with gr.Tabs():
            # ---- TAB 1: Single Snapshot ----
            with gr.TabItem("Snapshot"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Paste your journal entry, voice memo transcript, or stream-of-consciousness text here...",
                            lines=8,
                            max_lines=20,
                        )
                        with gr.Row():
                            analyze_btn = gr.Button("Take Snapshot", variant="primary", size="lg")
                            sample_reflective = gr.Button("Sample: Reflective", variant="secondary")
                            sample_stressed = gr.Button("Sample: Stressed", variant="secondary")
                            sample_energized = gr.Button("Sample: Energized", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        radar_chart = gr.Plot(label="Cognitive Snapshot")
                    with gr.Column(scale=1):
                        dimension_cards = gr.HTML(label="Dimension Scores")

                synthesis_output = gr.Markdown(label="Synthesis Narrative")

                # Drill-down offer (hidden until Step 1 completes successfully)
                with gr.Row(visible=False) as drilldown_row:
                    drilldown_btn = gr.Button(
                        "One pattern stands out. Want to look closer?",
                        variant="secondary",
                        size="lg",
                    )

                drilldown_card = gr.Markdown(label="Drill-Down Analysis")

                with gr.Row():
                    dimension_select = gr.Dropdown(
                        choices=["sentiment", "temporal", "complexity", "social", "coherence"],
                        value="sentiment",
                        label="Highlight Dimension",
                    )

                annotated_text = gr.HTML(label="Annotated Text")

                with gr.Accordion("Agent Reasoning Log", open=True):
                    agent_log = gr.Markdown()

                # Wire snapshot events — 7 outputs matching generator tuple
                analyze_btn.click(
                    fn=run_analysis_progressive,
                    inputs=[text_input],
                    outputs=[radar_chart, dimension_cards, synthesis_output,
                             drilldown_row, drilldown_card, annotated_text, agent_log],
                )

                # Drill-down button
                drilldown_btn.click(
                    fn=run_drilldown,
                    inputs=[],
                    outputs=[drilldown_card],
                )

                sample_reflective.click(fn=lambda: SAMPLE_TEXTS["Reflective"], inputs=[], outputs=[text_input])
                sample_stressed.click(fn=lambda: SAMPLE_TEXTS["Stressed"], inputs=[], outputs=[text_input])
                sample_energized.click(fn=lambda: SAMPLE_TEXTS["Energized"], inputs=[], outputs=[text_input])

                dimension_select.change(
                    fn=update_annotation,
                    inputs=[text_input, dimension_select],
                    outputs=[annotated_text],
                )

            # ---- TAB 2: Compare ----
            with gr.TabItem("Compare"):
                gr.Markdown("### Compare Two Snapshots\nPaste two text samples to see how their cognitive profiles differ.")
                with gr.Row():
                    compare_text_a = gr.Textbox(label="Snapshot A", lines=6, placeholder="First text sample...")
                    compare_text_b = gr.Textbox(label="Snapshot B", lines=6, placeholder="Second text sample...")

                with gr.Row():
                    compare_btn = gr.Button("Compare Snapshots", variant="primary", size="lg")
                    load_pair_btn = gr.Button("Load Sample Pair", variant="secondary")

                compare_radar = gr.Plot(label="Overlaid Radar Charts")
                compare_table = gr.HTML(label="Score Differences")
                compare_narrative = gr.Markdown(label="Differential Analysis")

                compare_btn.click(
                    fn=run_comparison,
                    inputs=[compare_text_a, compare_text_b],
                    outputs=[compare_radar, compare_table, compare_narrative],
                )

                load_pair_btn.click(
                    fn=_load_sample_pair,
                    inputs=[],
                    outputs=[compare_text_a, compare_text_b],
                )

    return app


def launch(**kwargs):
    """Convenience function to create and launch the app."""
    app = create_app()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch(share=False)
