"""Phase 8 - the advisory dashboard.

    streamlit run src/adaptive_reasoning/app/dashboard.py

One screen, one story: the model was asked a question, the policy stopped it early,
here is what that cost or saved. Everything else is behind an expander, because a
reviewer looking at this for the first time should not have to decide what matters.

The reasoning is replayed from the Phase 3 recordings, so it runs instantly with no
GPU, and it goes through the Phase 7 controller - the same code that would drive a live
model. It calls the controller directly rather than over HTTP: a demonstration that
depends on a second process starting is one that can fail in the room.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script, so the package needs to be importable.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from adaptive_reasoning.app.api import DemoStore  # noqa: E402
from adaptive_reasoning.config import load_config  # noqa: E402
from adaptive_reasoning.serve.controller import (  # noqa: E402
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
)

TOKENS_PER_SECOND = 91.5     # measured by the Phase 3 pilot on a Kaggle T4

POLICY_NAMES = {"dqn": "Difficulty-aware DQN", "bc": "Behaviour cloning"}


@st.cache_resource(show_spinner="Loading traces and policies ...")
def _store(experiment: str = "reported") -> DemoStore:
    return DemoStore(load_config(experiment))


def _run(store: DemoStore, question_id: str, policy_name: str):
    decide, floor = store.policies[policy_name]
    vector = store.difficulty_vectors.get(question_id)
    source = ReplaySource.from_frame(store.traces, question_id)

    adaptive = AdaptiveController(store.cfg, decide, difficulty_vector=vector,
                                  budget=store.budget, min_steps=floor).run(source)
    full = AdaptiveController(store.cfg, always_continue_policy(),
                              difficulty_vector=vector, budget=store.budget).run(source)
    return adaptive, full


def main() -> None:
    cfg = load_config("reported")
    st.set_page_config(page_title=cfg.app.title, layout="wide")
    store = _store()

    st.title("Adaptive Financial Advisory")
    st.caption("The system decides when the model has thought long enough.")

    # -- pick a question ----------------------------------------------------- #
    with st.sidebar:
        st.subheader("Choose a question")
        frame = store.questions[store.questions.index.isin(store.test_ids)]

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

        st.subheader("Stopping policy")
        policy_name = st.radio(
            "policy", sorted(store.policies),
            format_func=lambda k: POLICY_NAMES.get(k, k),
            label_visibility="collapsed",
        )

    row = store.questions.loc[question_id]
    adaptive, full = _run(store, question_id, policy_name)
    was_right = store.correct_at.get((question_id, adaptive.stop_step))
    full_right = store.correct_at.get((question_id, full.stop_step))
    saved = full.tokens_used - adaptive.tokens_used
    pct = 100.0 * saved / max(full.tokens_used, 1)

    # -- the question -------------------------------------------------------- #
    st.markdown(f"#### {row.question}")

    # -- the answer, side by side -------------------------------------------- #
    left, right = st.columns(2)
    with left:
        st.caption(f"STOPPED EARLY  ·  {adaptive.tokens_used} tokens")
        if was_right:
            st.success(f"{adaptive.answer}\n\n**Correct**")
        else:
            st.error(f"{adaptive.answer}\n\n**Incorrect**")
    with right:
        st.caption(f"FULL REASONING  ·  {full.tokens_used} tokens")
        if full_right:
            st.success(f"{full.answer}\n\n**Correct**")
        else:
            st.error(f"{full.answer}\n\n**Incorrect**")

    if was_right and not full_right:
        st.info("**The model had it, then talked itself out of it.** Stopping early "
                "got the right answer; thinking longer lost it.")
    elif full_right and not was_right:
        st.warning("On this question, stopping early was too soon.")

    # -- the saving ---------------------------------------------------------- #
    a, b, c = st.columns(3)
    a.metric("Reasoning saved", f"{pct:.0f}%")
    b.metric("Tokens", f"{adaptive.tokens_used}", delta=f"{-saved}",
             delta_color="inverse")
    c.metric("Time", f"{adaptive.tokens_used / TOKENS_PER_SECOND:.1f}s",
             delta=f"-{saved / TOKENS_PER_SECOND:.1f}s", delta_color="inverse")

    # -- how it decided ------------------------------------------------------ #
    st.markdown("##### How it decided")
    st.dataframe(
        [
            {
                "Step": d.step_index + 1,
                "Tokens": d.tokens_so_far,
                "Confidence": f"{d.confidence:.0%}",
                "Answer at this point": d.answer[:55],
                "Decision": "STOP  ←" if d.stopped else "keep thinking",
            }
            for d in adaptive.decisions
        ],
        width="stretch", hide_index=True,
    )

    # -- everything else, folded away ---------------------------------------- #
    if str(row.context or "").strip():
        with st.expander("Source document given to the model"):
            st.text(str(row.context)[:4000])

    with st.expander("The model's reasoning, step by step"):
        for d in adaptive.decisions:
            st.markdown(f"**Step {d.step_index + 1}**")
            st.text(d.step_text[:800] or "(no text recorded)")

    with st.expander("Overall results on 599 unseen questions"):
        import json

        from adaptive_reasoning import paths

        summary = paths.RESULTS / "phase6_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))["results"]
            labels = {
                "full_reasoning": "Full reasoning (no early stop)",
                "fixed_step_matched": "Fixed rule, same budget",
                "behaviour_cloning": "Behaviour cloning",
                "dqn": "Difficulty-aware DQN",
                "oracle": "Best possible (uses hindsight)",
            }
            st.dataframe(
                [
                    {"Method": labels[name],
                     "Accuracy": f"{data[name]['accuracy']:.1%}",
                     "Tokens": f"{data[name]['mean_tokens']:.0f}",
                     "Saved": f"{data[name]['token_reduction_pct']:.0f}%"}
                    for name in labels if name in data
                ],
                width="stretch", hide_index=True,
            )
            st.caption("At the same token budget the DQN beats the fixed rule by 7.2 "
                       "points. The last row is an upper bound, not a method.")
        else:
            st.info("Run scripts/run_phase6.py to fill this in.")

    st.caption(adaptive.disclaimer)


main()
