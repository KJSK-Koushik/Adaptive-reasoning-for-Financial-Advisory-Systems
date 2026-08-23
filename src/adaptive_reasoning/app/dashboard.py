"""Phase 8 - the advisory dashboard.

    streamlit run src/adaptive_reasoning/app/dashboard.py

Shows the thing the project is about: the model reasoning step by step, the policy
deciding when to stop, and what that decision cost or saved. The reasoning is replayed
from the Phase 3 recordings, so it runs instantly on a laptop with no GPU - and it goes
through the Phase 7 controller, the same code that would drive a live model.

It talks to the controller directly rather than over HTTP. A demonstration that depends
on a separate server process is a demonstration that can fail in the room.
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
    st.set_page_config(page_title=cfg.app.title, page_icon="||", layout="wide")
    store = _store()

    st.title(cfg.app.title)
    st.caption("Adaptive reasoning termination — the system decides when the model has "
               "thought long enough.")

    # -- sidebar: pick a question ------------------------------------------- #
    with st.sidebar:
        st.header("Question")
        policy_name = st.radio(
            "Stopping policy",
            sorted(store.policies),
            format_func=lambda k: {"dqn": "Difficulty-aware DQN",
                                   "bc": "Behaviour cloning"}.get(k, k),
            help="Both were trained in Phase 5 and evaluated in Phase 6.",
        )

        frame = store.questions[store.questions.index.isin(store.test_ids)]
        domains = ["all"] + sorted(frame.domain.unique().tolist())
        domain = st.selectbox("Domain", domains)
        if domain != "all":
            frame = frame[frame.domain == domain]

        options = frame.head(200).index.tolist()
        if not options:
            st.error("No traced questions for that domain.")
            st.stop()
        question_id = st.selectbox(
            "Question", options,
            format_func=lambda q: str(frame.loc[q, "question"])[:70] + " ...",
        )
        st.divider()
        st.caption(f"{len(store.traced_ids):,} traced questions · "
                   f"budget {store.budget} tokens")

    row = store.questions.loc[question_id]

    st.subheader("Question")
    st.write(str(row.question))
    if str(row.context or "").strip():
        with st.expander("Context provided to the model"):
            st.text(str(row.context)[:4000])

    adaptive, full = _run(store, question_id, policy_name)
    was_right = store.correct_at.get((question_id, adaptive.stop_step))
    full_right = store.correct_at.get((question_id, full.stop_step))
    saved = full.tokens_used - adaptive.tokens_used
    pct = 100.0 * saved / max(full.tokens_used, 1)

    # -- the answer ---------------------------------------------------------- #
    st.subheader("Answer")
    left, right = st.columns(2)
    with left:
        st.markdown("**With adaptive stopping**")
        # A bare conditional expression would leave a DeltaGenerator on the line,
        # which Streamlit's magic then renders as a block of repr text.
        if was_right:
            st.success(adaptive.answer)
        else:
            st.warning(adaptive.answer)
        st.caption(f"stopped at step {adaptive.stop_step} · {adaptive.tokens_used} tokens "
                   f"· {'correct' if was_right else 'incorrect'}")
    with right:
        st.markdown("**With full reasoning**")
        if full_right:
            st.success(full.answer)
        else:
            st.warning(full.answer)
        st.caption(f"ran to step {full.stop_step} · {full.tokens_used} tokens "
                   f"· {'correct' if full_right else 'incorrect'}")

    if was_right and not full_right:
        st.info("Stopping early produced the **correct** answer where full reasoning "
                "did not — the model had it, then talked itself out of it.")
    elif full_right and not was_right:
        st.warning("Stopping early cost the correct answer on this question.")

    # -- what it saved ------------------------------------------------------- #
    if cfg.app.show_savings_panel:
        st.subheader("What early stopping saved")
        a, b, c, d = st.columns(4)
        a.metric("Tokens used", f"{adaptive.tokens_used}",
                 delta=f"-{saved}", delta_color="inverse")
        b.metric("Reasoning saved", f"{pct:.0f}%")
        c.metric("Latency", f"{adaptive.tokens_used / TOKENS_PER_SECOND:.2f}s",
                 delta=f"-{saved / TOKENS_PER_SECOND:.2f}s", delta_color="inverse")
        d.metric("Stopped because", adaptive.stop_reason.replace("_", " "))

    # -- the reasoning, step by step ----------------------------------------- #
    if cfg.app.show_reasoning_trace:
        st.subheader("The decision at every step")
        st.caption("The policy sees confidence, entropy and answer stability at each "
                   "boundary, and chooses CONTINUE or STOP.")
        rows = [
            {
                "step": d.step_index,
                "tokens": d.tokens_so_far,
                "confidence": round(d.confidence, 3),
                "entropy": round(d.entropy, 3),
                "answer so far": d.answer[:60],
                "decision": "STOP" if d.stopped else "continue",
            }
            for d in adaptive.decisions
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

        with st.expander("Reasoning text the model produced"):
            for d in adaptive.decisions:
                st.markdown(f"**Step {d.step_index}** · {d.tokens_so_far} tokens")
                st.text(d.step_text[:800] or "(no text recorded)")

    # -- headline evaluation ------------------------------------------------- #
    with st.expander("How this performs overall (599 held-out test questions)"):
        import json

        from adaptive_reasoning import paths

        summary = paths.RESULTS / "phase6_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))["results"]
            table = [
                {"policy": name.replace("_", " "),
                 "accuracy": f"{data[name]['accuracy']:.1%}",
                 "mean tokens": f"{data[name]['mean_tokens']:.0f}",
                 "tokens saved": f"{data[name]['token_reduction_pct']:.1f}%"}
                for name in ("full_reasoning", "fixed_step_matched",
                             "behaviour_cloning", "dqn", "oracle")
                if name in data
            ]
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption("Fixed step is the standard approach in prior work, matched to "
                       "the same token budget. Oracle is an upper bound that uses "
                       "hindsight and is not implementable.")
        else:
            st.info("Run scripts/run_phase6.py to populate this table.")

    st.divider()
    st.caption(adaptive.disclaimer)


main()
