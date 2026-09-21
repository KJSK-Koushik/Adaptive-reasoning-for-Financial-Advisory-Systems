"""Phase 8 - the advisory dashboard.

    streamlit run src/adaptive_reasoning/app/dashboard.py

One screen, one story: the model was asked a question, the policy stopped it early,
here is what that cost or saved. Everything else is behind an expander, because a
reviewer looking at this for the first time should not have to decide what matters.

The reasoning is replayed from the Phase 3 recordings, so it runs instantly with no
GPU, and it goes through the Phase 7 controller - the same code that would drive a live
model. It calls the controller directly rather than over HTTP: a demonstration that
depends on a second process starting is one that can fail in the room.

The look follows a cream-canvas, saturated-card system: warm off-white page, near-black
type, and single-colour cards (teal, lavender, pink, ochre, peach, cream) that never
repeat consecutively. Display headings are weight 500 with negative tracking, never
bolder. Depth comes from colour contrast, not shadows.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script, so the package needs to be importable.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import html  # noqa: E402
import re  # noqa: E402

import streamlit as st  # noqa: E402

from adaptive_reasoning.app.api import DemoStore  # noqa: E402
from adaptive_reasoning.config import load_config  # noqa: E402
from adaptive_reasoning.serve.controller import (  # noqa: E402
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
)

TOKENS_PER_SECOND = 91.5     # measured by the Phase 3 pilot on a Kaggle T4

POLICY_NAMES = {"dqn": "RL agent — Double DQN", "bc": "Behaviour cloning"}

#: FinQA and ConvFinQA question ids carry the filing year, e.g.
#: finqa::C/2009/page_141.pdf-3. The other five sources have no year in the id and are
#: never excluded by this filter.
_YEAR = re.compile(r"/((?:19|20)\d{2})/")

#: Filings older than this are hidden from the picker. A presentation choice only: the
#: reported accuracy in Phase 6 is measured over the whole test split, not this subset.
MIN_YEAR = 2015

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #
CANVAS, SOFT, CARD, STRONG = "#fffaf0", "#faf5e8", "#f5f0e0", "#ebe6d6"
INK, BODY, MUTED, HAIRLINE = "#0a0a0a", "#3a3a3a", "#6a6a6a", "#e5e5e5"
PINK, TEAL, LAVENDER, PEACH, OCHRE = "#ff4d8b", "#1a3a3a", "#b8a4ed", "#ffb084", "#e8b94a"
MINT, CORAL, WHITE = "#a4d4c5", "#ff6b5a", "#ffffff"

#: Card surface -> text colour. Pink and teal are deep enough for white type; the
#: lighter saturations take ink.
ON = {PINK: WHITE, TEAL: WHITE, LAVENDER: INK, PEACH: INK, OCHRE: INK, CARD: INK,
      SOFT: INK, CANVAS: INK}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{ font-family: {FONT}; }}
.stApp {{ background: {CANVAS}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1280px; padding-top: 2.5rem; padding-bottom: 0; }}

section[data-testid="stSidebar"] {{ background: {SOFT}; border-right: 1px solid {HAIRLINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
section[data-testid="stSidebar"] label p {{ font-size: 14px; font-weight: 500; color: {INK}; }}
div[data-baseweb="select"] > div {{ background: {CANVAS}; border: 1px solid {HAIRLINE};
    border-radius: 12px; min-height: 44px; }}
div[role="radiogroup"] label {{ background: {CANVAS}; border: 1px solid {HAIRLINE};
    border-radius: 9999px; padding: 6px 14px; margin-bottom: 6px; }}

div[data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {HAIRLINE};
    border-radius: 16px; margin-bottom: 12px; }}
div[data-testid="stExpander"] details {{ border: none; }}
div[data-testid="stExpander"] summary p {{ font-size: 16px; font-weight: 600; color: {INK}; }}
div[data-testid="stExpander"] pre, div[data-testid="stExpander"] code {{
    background: {CANVAS}; border: 1px solid {HAIRLINE}; border-radius: 12px;
    color: {BODY}; font-size: 13px; }}

.ar-eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; opacity: .75; margin-bottom: 8px; }}
.ar-display-lg {{ font-size: 56px; font-weight: 500; letter-spacing: -2px; line-height: 1.05;
    margin: 0 0 12px; }}
.ar-display-sm {{ font-size: 32px; font-weight: 500; letter-spacing: -0.5px; line-height: 1.15;
    margin: 0; word-break: break-word; }}
.ar-display-md {{ font-size: 40px; font-weight: 500; letter-spacing: -1px; line-height: 1.1;
    margin: 0; }}
.ar-title-lg {{ font-size: 24px; font-weight: 600; letter-spacing: -0.3px; line-height: 1.3;
    margin: 0; }}
.ar-title-md {{ font-size: 18px; font-weight: 600; line-height: 1.4; margin: 0; }}
.ar-lead {{ font-size: 18px; font-weight: 400; line-height: 1.5; color: {BODY}; margin: 0; }}
.ar-body {{ font-size: 16px; line-height: 1.55; margin: 8px 0 0; }}
.ar-caption {{ font-size: 13px; font-weight: 500; line-height: 1.4; opacity: .8; margin-top: 10px; }}

.ar-card {{ border-radius: 24px; padding: 32px; margin-bottom: 20px; min-height: 100%; }}
.ar-card-lg {{ border-radius: 16px; padding: 24px; margin-bottom: 20px; }}
.ar-hairline {{ border: 1px solid {HAIRLINE}; }}
.ar-hero {{ display: grid; grid-template-columns: 7fr 5fr; gap: 32px; align-items: center;
    margin-bottom: 40px; }}
.ar-hero-art {{ background: {SOFT}; border-radius: 24px; padding: 8px; }}
.ar-hero-art svg {{ width: 100%; height: auto; display: block; }}
@media (max-width: 900px) {{ .ar-hero {{ grid-template-columns: 1fr; }}
    .ar-display-lg {{ font-size: 36px; letter-spacing: -1px; }} }}

.ar-pill {{ display: inline-block; border-radius: 9999px; padding: 4px 12px; font-size: 13px;
    font-weight: 500; line-height: 1.4; margin: 0 6px 6px 0; }}
.ar-pills {{ margin-top: 14px; }}

table.ar-table {{ width: 100%; border-collapse: collapse; font-size: 14px; min-width: 520px; }}
.ar-scroll {{ overflow-x: auto; }}
table.ar-table td, table.ar-table th {{ border-left: none; border-right: none; border-top: none; }}
table.ar-table th {{ text-align: left; font-size: 12px; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: {MUTED}; padding: 8px 12px; border-bottom: 1px solid {HAIRLINE}; }}
table.ar-table td {{ padding: 10px 12px; border-bottom: 1px solid {HAIRLINE}; color: {BODY};
    vertical-align: middle; }}
table.ar-table tr:last-child td {{ border-bottom: none; }}
table.ar-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.ar-table td.conf {{ white-space: nowrap; }}
table.ar-table tr.stop td {{ background: {PINK}; color: {WHITE}; }}
table.ar-table tr.stop td:first-child {{ border-radius: 12px 0 0 12px; }}
table.ar-table tr.stop td:last-child {{ border-radius: 0 12px 12px 0; }}
table.ar-table tr.ours td {{ background: {TEAL}; color: {WHITE}; font-weight: 600; }}
table.ar-table tr.ours td:first-child {{ border-radius: 12px 0 0 12px; }}
table.ar-table tr.ours td:last-child {{ border-radius: 0 12px 12px 0; }}
.ar-bar {{ display: inline-block; height: 8px; border-radius: 9999px; background: {STRONG};
    width: 96px; vertical-align: middle; margin-right: 8px; overflow: hidden; }}
.ar-bar > span {{ display: block; height: 100%; background: {TEAL}; border-radius: 9999px; }}
tr.stop .ar-bar {{ background: rgba(255,255,255,.35); }}
tr.stop .ar-bar > span {{ background: {WHITE}; }}

.ar-footer {{ background: {SOFT}; border-radius: 24px 24px 0 0; padding: 40px 32px 32px;
    margin-top: 40px; color: {BODY}; font-size: 14px; line-height: 1.55; }}
.ar-footer .ar-eyebrow {{ color: {MUTED}; }}
</style>
"""

#: A horizon of rounded hills in the card palette - the hero artifact. Decorative only.
HERO_ART = f"""
<svg viewBox="0 0 520 300" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Rounded hills in lavender, peach and ochre under a mint sun">
  <rect width="520" height="300" rx="20" fill="{SOFT}"/>
  <circle cx="400" cy="86" r="46" fill="{MINT}"/>
  <path d="M0 250 C 90 130, 170 130, 260 220 S 430 150, 520 240 V300 H0 Z" fill="{LAVENDER}"/>
  <path d="M0 300 V262 C 80 190, 160 200, 240 262 S 380 210, 520 270 V300 Z" fill="{PEACH}"/>
  <path d="M0 300 V284 C 120 240, 220 250, 320 288 S 450 260, 520 292 V300 Z" fill="{OCHRE}"/>
  <circle cx="118" cy="214" r="10" fill="{PINK}"/>
  <circle cx="300" cy="238" r="7" fill="{PINK}"/>
  <circle cx="440" cy="252" r="12" fill="{TEAL}"/>
</svg>
"""


def _recent_enough(question_id: str) -> bool:
    match = _YEAR.search(str(question_id))
    return match is None or int(match.group(1)) >= MIN_YEAR


def _artifact_stamp() -> tuple:
    """Modification times of everything the store loads.

    Passing this as a cache key means retraining a policy or re-grading the traces
    invalidates the cache by itself. Without it Streamlit happily serves whatever it
    loaded at startup - which twice meant the dashboard showing results that no longer
    matched the code, the worst possible failure for something used to demonstrate.
    """
    from adaptive_reasoning import paths

    watched = (paths.UNIFIED_DATASET, paths.TRACE_DATASET, paths.RL_TRANSITIONS,
               paths.DQN_POLICY, paths.MODELS / "behaviour_cloning.joblib")
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in watched)


@st.cache_resource(show_spinner="Loading traces and policies ...")
def _load_store(experiment: str, stamp: tuple) -> DemoStore:
    return DemoStore(load_config(experiment))


def _store(experiment: str = "reported") -> DemoStore:
    return _load_store(experiment, _artifact_stamp())


def _run(store: DemoStore, question_id: str, policy_name: str):
    decide, floor = store.policies[policy_name]
    vector = store.difficulty_vectors.get(question_id)
    source = ReplaySource.from_frame(store.traces, question_id)

    adaptive = AdaptiveController(store.cfg, decide, difficulty_vector=vector,
                                  budget=store.budget, min_steps=floor).run(source)
    full = AdaptiveController(store.cfg, always_continue_policy(),
                              difficulty_vector=vector, budget=store.budget).run(source)
    return adaptive, full


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _h(text: object) -> str:
    return html.escape(str(text))


def _card(surface: str, *inner: str, rounded: str = "card", hairline: bool = False) -> str:
    """One coloured card. Everything inside is already-escaped HTML."""
    klass = "ar-card" if rounded == "card" else "ar-card-lg"
    if hairline:
        klass += " ar-hairline"
    return (f'<div class="{klass}" style="background:{surface};color:{ON[surface]}">'
            + "".join(inner) + "</div>")


def _pill(text: str, surface: str, colour: str | None = None) -> str:
    return (f'<span class="ar-pill" style="background:{surface};'
            f'color:{colour or ON.get(surface, INK)}">{_h(text)}</span>')


def _eyebrow(text: str) -> str:
    return f'<div class="ar-eyebrow">{_h(text)}</div>'


def _verdict_pill(correct: bool | None) -> str:
    if correct:
        return _pill("Correct", MINT, INK)
    return _pill("Incorrect", CORAL, WHITE)


def _answer_card(surface: str, eyebrow: str, answer: str, correct: bool | None) -> str:
    return _card(
        surface, _eyebrow(eyebrow),
        f'<div class="ar-display-sm">{_h(answer or "(no answer)")}</div>',
        f'<div class="ar-pills">{_verdict_pill(correct)}</div>',
    )


def _metric_card(surface: str, eyebrow: str, value: str, note: str) -> str:
    return _card(surface, _eyebrow(eyebrow),
                 f'<div class="ar-display-md">{_h(value)}</div>',
                 f'<div class="ar-caption">{_h(note)}</div>')


def _decision_table(decisions) -> str:
    rows = []
    for d in decisions:
        conf = max(0.0, min(1.0, float(d.confidence)))
        bar = f'<span class="ar-bar"><span style="width:{conf * 100:.0f}%"></span></span>'
        decision = "STOP" if d.stopped else "keep thinking"
        rows.append(
            f'<tr class="{"stop" if d.stopped else ""}">'
            f'<td class="num">{d.step_index + 1}</td>'
            f'<td class="num">{d.tokens_so_far}</td>'
            f'<td class="conf">{bar}{conf:.0%}</td>'
            f'<td>{_h(d.answer[:55])}</td>'
            f'<td>{decision}</td></tr>'
        )
    return ('<div class="ar-scroll"><table class="ar-table"><thead><tr><th>Step</th>'
            '<th>Tokens</th><th>Confidence</th><th>Answer at this point</th>'
            '<th>Decision</th></tr></thead><tbody>' + "".join(rows)
            + '</tbody></table></div>')


def _results_table(data: dict) -> tuple[str, str]:
    """The Phase 6 table and a one-line caption, both from the results file."""
    labels = {
        "full_reasoning": "Full reasoning — no early stop",
        "fixed_step_matched": "Fixed rule — same budget",
        "behaviour_cloning": "Behaviour cloning",
        "dqn": "RL agent — Double DQN",
        "oracle": "Best possible — uses hindsight",
    }
    rows = []
    for name, label in labels.items():
        if name not in data:
            continue
        r = data[name]
        saved = "—" if r["token_reduction_pct"] < 0.05 else f'{r["token_reduction_pct"]:.0f}%'
        rows.append(
            f'<tr class="{"ours" if name == "dqn" else ""}"><td>{_h(label)}</td>'
            f'<td class="num">{r["accuracy"]:.1%}</td>'
            f'<td class="num">{r["mean_tokens"]:.0f}</td>'
            f'<td class="num">{saved}</td></tr>'
        )
    table = ('<div class="ar-scroll"><table class="ar-table"><thead><tr><th>Method</th>'
             '<th>Accuracy</th><th>Tokens</th><th>Saved</th></tr></thead><tbody>'
             + "".join(rows) + '</tbody></table></div>')
    caption = ""
    if "dqn" in data and "fixed_step_matched" in data:
        margin = (data["dqn"]["accuracy"] - data["fixed_step_matched"]["accuracy"]) * 100
        caption = (f"At the same token budget the RL agent beats the fixed rule by "
                   f"{margin:.1f} points. The last row is an upper bound, not a method.")
    return table, caption


def main() -> None:
    cfg = load_config("reported")
    st.set_page_config(page_title=cfg.app.title, layout="wide", page_icon="🧭")
    st.markdown(CSS, unsafe_allow_html=True)
    store = _store()

    # -- pick a question ----------------------------------------------------- #
    with st.sidebar:
        st.markdown(_eyebrow("Choose a question"), unsafe_allow_html=True)
        frame = store.questions[store.questions.index.isin(store.test_ids)]
        frame = frame[[_recent_enough(q) for q in frame.index]]

        domain = st.selectbox("Topic",
                              ["all"] + sorted(frame.domain.unique().tolist()))
        if domain != "all":
            frame = frame[frame.domain == domain]

        options = frame.head(200).index.tolist()
        if not options:
            st.error("No questions for that topic.")
            st.stop()
        question_id = st.selectbox(
            "Question", options,
            format_func=lambda q: str(frame.loc[q, "question"])[:60] + " ...",
        )
        st.caption(f"Showing filings from {MIN_YEAR} onwards.")

        st.markdown(_eyebrow("Stopping policy"), unsafe_allow_html=True)
        keys = sorted(store.policies)
        policy_name = st.radio(
            "policy", keys, index=keys.index("dqn") if "dqn" in keys else 0,
            format_func=lambda k: POLICY_NAMES.get(k, k),
            label_visibility="collapsed",
        )

    row = store.questions.loc[question_id]
    adaptive, full = _run(store, question_id, policy_name)
    was_right = store.correct_at.get((question_id, adaptive.stop_step))
    full_right = store.correct_at.get((question_id, full.stop_step))
    saved = full.tokens_used - adaptive.tokens_used
    pct = 100.0 * saved / max(full.tokens_used, 1)

    # -- hero ---------------------------------------------------------------- #
    st.markdown(
        '<div class="ar-hero"><div>'
        + _eyebrow("Adaptive reasoning · Phase 8")
        + f'<h1 class="ar-display-lg">{_h(cfg.app.title)}</h1>'
        + '<p class="ar-lead">The system decides when the model has thought long '
          'enough — and stops it there.</p>'
        + '</div><div class="ar-hero-art">' + HERO_ART + '</div></div>',
        unsafe_allow_html=True,
    )

    # -- the question: cream ------------------------------------------------- #
    pills = [_pill(str(row.domain).replace("_", " "), STRONG),
             _pill(str(row.source), STRONG)]
    if str(row.get("difficulty") or "").strip():
        pills.append(_pill(f"{row.difficulty} difficulty", STRONG))
    st.markdown(
        _card(CARD, _eyebrow("The question"),
              f'<div class="ar-title-lg">{_h(row.question)}</div>',
              '<div class="ar-pills">' + "".join(pills) + '</div>'),
        unsafe_allow_html=True,
    )

    # -- the answers, side by side: teal then lavender ----------------------- #
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(_answer_card(TEAL, f"Stopped early · {adaptive.tokens_used} tokens",
                                 adaptive.answer, was_right), unsafe_allow_html=True)
    with right:
        st.markdown(_answer_card(LAVENDER, f"Full reasoning · {full.tokens_used} tokens",
                                 full.answer, full_right), unsafe_allow_html=True)

    # -- the verdict: pink when early stopping won, cream when it lost ------- #
    if was_right and not full_right:
        st.markdown(
            _card(PINK, _eyebrow("What happened"),
                  '<div class="ar-title-lg">The model had it, then talked itself out '
                  'of it.</div>',
                  '<p class="ar-body">Stopping early got the right answer; thinking '
                  'longer lost it.</p>'),
            unsafe_allow_html=True,
        )
    elif full_right and not was_right:
        st.markdown(
            _card(CANVAS, _eyebrow("What happened"),
                  '<div class="ar-title-lg">On this question, stopping early was too '
                  'soon.</div>',
                  '<p class="ar-body">Full reasoning reached the right answer; the '
                  'policy stopped before it got there.</p>', hairline=True),
            unsafe_allow_html=True,
        )

    # -- the saving: ochre, peach, cream ------------------------------------- #
    a, b, c = st.columns(3, gap="medium")
    with a:
        st.markdown(_metric_card(OCHRE, "Reasoning saved", f"{pct:.0f}%",
                                 "of the tokens full reasoning used"),
                    unsafe_allow_html=True)
    with b:
        st.markdown(_metric_card(PEACH, "Tokens", f"{adaptive.tokens_used}",
                                 f"{saved} fewer than full reasoning"),
                    unsafe_allow_html=True)
    with c:
        st.markdown(_metric_card(CARD, "Time", f"{adaptive.tokens_used / TOKENS_PER_SECOND:.1f}s",
                                 f"{saved / TOKENS_PER_SECOND:.1f}s faster on a T4 GPU"),
                    unsafe_allow_html=True)

    # -- how it decided ------------------------------------------------------ #
    st.markdown(
        _card(CANVAS, _eyebrow("How it decided"),
              '<div class="ar-title-md" style="margin-bottom:14px">One decision per '
              'reasoning step. The highlighted row is where the policy stopped.</div>',
              _decision_table(adaptive.decisions), rounded="lg", hairline=True),
        unsafe_allow_html=True,
    )

    # -- everything else, folded away ---------------------------------------- #
    if str(row.context or "").strip():
        with st.expander("Source document given to the model"):
            st.text(str(row.context)[:4000])

    with st.expander("The model's reasoning, step by step"):
        for d in adaptive.decisions:
            st.markdown(f"**Step {d.step_index + 1}**")
            st.text(d.step_text[:800] or "(no text recorded)")

    with st.expander("Overall results on the unseen test questions"):
        import json

        from adaptive_reasoning import paths

        summary = paths.RESULTS / "phase6_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))["results"]
            table, caption = _results_table(data)
            n = data.get("dqn", {}).get("n")
            st.markdown(table, unsafe_allow_html=True)
            if caption:
                st.caption(caption + (f" Measured on {n} questions." if n else ""))
        else:
            st.info("Run scripts/run_phase6.py to fill this in.")

    # -- footer: cream, never dark ------------------------------------------- #
    st.markdown(
        '<div class="ar-footer">' + _eyebrow("About this demonstration")
        + '<p>Replayed from the Phase 3 recordings through the Phase 7 controller — '
          'the same code that would drive a live model — so it runs with no GPU. '
          'Accuracy figures come from the Phase 6 evaluation on the held-out test '
          'split.</p>'
        + (f'<p>{_h(adaptive.disclaimer)}</p>' if adaptive.disclaimer else "")
        + '</div>',
        unsafe_allow_html=True,
    )


main()
