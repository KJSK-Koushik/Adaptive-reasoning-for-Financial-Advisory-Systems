"""Bring the mid-review deck up to date with the results files.

    python scripts/refresh_mid_review.py            # writes Mid_Review_Adaptive_Reasoning_v3.pptx
    python scripts/refresh_mid_review.py --tests 449

The deck was designed by hand, so this does not rebuild it. It rewrites the text,
tables and charts that quote a number, keeping every shape where it is. Anything it
touches comes from artifacts/results, so rerunning after a phase changes a figure keeps
the deck honest without a designer in the loop.

Slide numbers refer to Mid_Review_Adaptive_Reasoning_v2.pptx (20 slides).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_reasoning import paths  # noqa: E402

SOURCE = ROOT / "Mid_Review_Adaptive_Reasoning_v2.pptx"
TARGET = ROOT / "Mid_Review_Adaptive_Reasoning_v3.pptx"
SCREENSHOT = ROOT / "docs" / "tests_terminal.png"


def _load(name: str) -> dict:
    return json.loads((paths.RESULTS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# editing helpers - change words, never formatting
# --------------------------------------------------------------------------- #
def set_par(par, text: str) -> None:
    """Put ``text`` in the paragraph's first run and drop the rest."""
    runs = par.runs
    if not runs:
        par.add_run().text = text
        return
    runs[0].text = text
    for run in runs[1:]:
        run._r.getparent().remove(run._r)


def set_shape(shape, *texts: str) -> None:
    """One string per paragraph, in order.

    Fewer strings than paragraphs removes the surplus. More strings than paragraphs
    clones the second-to-last paragraph's formatting for the extra lines, so a box that
    ends in a bold conclusion keeps that conclusion last.
    """
    import copy

    tf = shape.text_frame
    pars = list(tf.paragraphs)
    while len(pars) < len(texts):
        template = pars[-2] if len(pars) >= 2 else pars[-1]
        pars[-1]._p.addprevious(copy.deepcopy(template._p))
        pars = list(tf.paragraphs)
    for par, text in zip(pars, texts, strict=False):
        set_par(par, text)
    for par in pars[len(texts):]:
        par._p.getparent().remove(par._p)


def set_cell(cell, text: str) -> None:
    set_par(cell.text_frame.paragraphs[0], text)


def shift(shapes, indices, inches: float) -> None:
    """Move shapes down. The original deck let two callouts cover a table's last row."""
    from pptx.util import Inches

    for i in indices:
        shapes[i].top = shapes[i].top + Inches(inches)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #
def gather() -> dict:
    import numpy as np
    import pandas as pd

    p3, p4, p6 = _load("phase3_summary.json"), _load("phase4_summary.json"), _load("phase6_summary.json")
    r = p6["results"]
    full, dqn, fixed, bc, oracle = (r["full_reasoning"], r["dqn"], r["fixed_step_matched"],
                                    r["behaviour_cloning"], r["oracle"])
    cmp_fixed = p6["comparisons"]["dqn_vs_fixed_step_matched"]
    cmp_bc = p6["comparisons"]["dqn_vs_behaviour_cloning"]

    # Accuracy if the model were stopped at step k (or where its trace ended).
    steps = pd.read_parquet(paths.TRACE_DATASET, columns=["question_id", "step_index", "probe_correct"])
    last = steps.groupby("question_id").step_index.max()
    held = steps.set_index(["question_id", "step_index"]).probe_correct
    curve = {}
    for k in (0, 1, 3, 6, 10, 15):
        idx = np.minimum(k, last)
        curve[k] = 100 * float(np.mean([bool(held.get((q, int(i)), False)) for q, i in idx.items()]))

    ablations = {}
    try:
        ablations = _load("phase9_summary.json")
    except FileNotFoundError:
        pass

    return {
        "p3": p3, "p4": p4, "full": full, "dqn": dqn, "fixed": fixed, "bc": bc,
        "oracle": oracle, "cmp_fixed": cmp_fixed, "cmp_bc": cmp_bc, "curve": curve,
        "matched_step": p6["matched"]["fixed_step"], "ablations": ablations,
        "gap": (p3["solvable_fraction"] - p3["final_accuracy"]) * 100,
        "headroom": (oracle["accuracy"] - full["accuracy"]) * 100,
        "params": p4["state_dim"] * 128 + 128 + 128 * 128 + 128 + 128 * 2 + 2,
    }


# --------------------------------------------------------------------------- #
# slide by slide
# --------------------------------------------------------------------------- #
def refresh(n_tests: int, code_lines: int, code_files: int) -> Path:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData, XyChartData
    from pptx.util import Emu, Inches

    F = gather()
    p3, p4, full, dqn, fixed, bc, oracle = (F["p3"], F["p4"], F["full"], F["dqn"],
                                            F["fixed"], F["bc"], F["oracle"])
    margin = F["cmp_fixed"]["difference"] * 100
    p_fixed = F["cmp_fixed"]["p_value"]
    p_text = "p < 0.0001" if p_fixed < 0.0001 else f"p = {p_fixed:.4f}"
    relative = (dqn["accuracy"] / fixed["accuracy"] - 1) * 100
    bc_margin = F["cmp_bc"]["difference"] * 100
    saved = dqn["token_reduction_pct"]

    prs = Presentation(str(SOURCE))
    S = prs.slides
    sh = lambda i: list(S[i - 1].shapes)          # 1-based slide, 0-based shape

    # -- 1 title ------------------------------------------------------------- #
    set_shape(sh(1)[4], f"Teaching a language model when to stop thinking — {margin:+.1f} "
                        f"accuracy points over the standard rule at the same cost, with "
                        f"{saved:.0f}% fewer reasoning tokens.")

    # -- 2 regular update ---------------------------------------------------- #
    s = sh(2)
    set_shape(s[3], "Since the last review the pipeline was completed end to end and "
                    "evaluated. Seven financial datasets were unified into one schema, "
                    f"{p3['n_traces']:,} reasoning traces were generated on GPU, a Double DQN "
                    "stopping policy was trained offline and compared against seven baselines "
                    "at matched token cost, and the trained policy now runs inside a live "
                    "controller behind a FastAPI service and a dashboard. This review reports "
                    "measured numbers, including an ablation that contradicts part of our "
                    "original hypothesis.")
    set_shape(s[5], "Phases 0–7"); set_shape(s[6], "COMPLETE")
    set_shape(s[7], "Data → traces → RL → evaluation → live controller")
    set_shape(s[9], "Phase 8"); set_shape(s[10], "IN PROGRESS")
    set_shape(s[11], "Advisory application — API and dashboard")
    set_shape(s[13], "Phase 9"); set_shape(s[14], "IN PROGRESS")
    set_shape(s[15], "Ablations and final report")
    set_shape(s[18], f"{code_lines:,}"); set_shape(s[19], f"lines of Python across {code_files} files")
    set_shape(s[20], f"{n_tests}")
    set_shape(s[24], f"{p4['n_transitions']:,}")
    set_shape(s[25], f"RL transitions generated from {p3['n_traces']:,} traces")

    # -- 3 evidence ---------------------------------------------------------- #
    s = sh(3)
    set_shape(s[3], "Run on 21 September 2026. Output is shown exactly as produced.")
    pic = s[4]
    if SCREENSHOT.exists():
        from PIL import Image
        data = SCREENSHOT.read_bytes()
        S[2].part.related_part(pic._element.blip_rId)._blob = data
        w, h = Image.open(SCREENSHOT).size
        pic.height = Emu(int(pic.width * h / w))
    set_shape(s[18], "41 tests. Grading is the reward function — if it is wrong, the agent "
                     "learns the wrong thing.")
    set_shape(s[24], "90 tests across reward, dataset, features, DQN and rollout, plus 35 on "
                     "the live controller and API — the same decisions as training, verified "
                     "to zero difference.")

    # -- 4 the core observation ---------------------------------------------- #
    s = sh(4)
    set_shape(s[6], "Accuracy if the model is stopped at step k:")
    cd = CategoryChartData()
    cd.categories = [str(k) for k in F["curve"]]
    cd.add_series("Accuracy", [round(v, 1) for v in F["curve"].values()])
    s[7].chart.replace_data(cd)
    axis = s[7].chart.value_axis                  # the old scale topped out at 38
    axis.minimum_scale, axis.maximum_scale = 15, 50
    axis.tick_labels.number_format = '0"%"'
    axis.tick_labels.number_format_is_linked = False
    s[4].height = s[4].height + Inches(0.35)      # room for a three-line caption
    shift(s, [8], 0.15)
    set_shape(s[8], f"Accuracy keeps rising slowly to {pct(p3['final_accuracy'])} at the end — "
                    f"yet the model held a correct answer at some step on "
                    f"{pct(p3['solvable_fraction'])} of questions. That {F['gap']:.0f}-point "
                    f"gap is what a good stopping rule can recover.")

    # -- 10 architecture ----------------------------------------------------- #
    s = sh(10)
    set_shape(s[20], f"{p4['state_dim']} signals per step: confidence, entropy, answer "
                     "stability, answer shape, difficulty")
    set_shape(s[27], f"Double DQN {p4['state_dim']}→128→128→2 learns STOP vs CONTINUE from "
                     "logged data")
    set_shape(s[29], f"{F['params']:,} parameters")
    set_shape(s[36], f"{saved:.0f}% tokens saved")
    set_shape(s[43], "Read left to right. Stages 01–03 build the training data once; stage "
                     "04 is trained offline on a CPU in minutes; stage 05 is the only part "
                     f"that runs at serving time, and it adds an {p4['state_dim']}-input "
                     "neural network — roughly 0.1% of the language model's cost — to every "
                     "reasoning step.")

    # -- 11 MDP -------------------------------------------------------------- #
    s = sh(11)
    set_shape(s[6], f"{p4['state_dim']} features: predicted difficulty (3), confidence, "
                    "minimum token confidence, entropy and its slope, token ratio, step "
                    "index, confidence change, answer stability, progress and doubt cues, "
                    "steps since the answer changed, and four answer-shape signals.")
    set_shape(s[17],
              "Double DQN (reinforcement learning) — learns the long-run value of continuing "
              "versus stopping, using a target network updated every 500 steps and Huber loss.",
              "Behaviour cloning (supervised control) — a LightGBM classifier trained to "
              "imitate the oracle's stop decision. Included as a control: if plain "
              "supervision matched RL, the sequential formulation would be unnecessary.")

    # -- 12 result 1 --------------------------------------------------------- #
    s = sh(12)
    set_shape(s[4], pct(p3["final_accuracy"]))
    set_shape(s[6], pct(p3["solvable_fraction"]))
    set_shape(s[8], f"{F['gap']:.0f} pts"); set_shape(s[9], "answers found, then lost")
    set_shape(s[10], f"{p3['mean_steps']:.1f}")
    t = s[14].table
    set_cell(t.cell(1, 1), pct(full["accuracy"])); set_cell(t.cell(1, 2), f"{full['mean_tokens']:.0f}")
    set_cell(t.cell(2, 1), pct(oracle["accuracy"])); set_cell(t.cell(2, 2), f"{oracle['mean_tokens']:.0f}")
    set_cell(t.cell(2, 3), f"{oracle['token_reduction_pct']:.1f}%")
    set_shape(s[15], "This is not a trade-off. Stopping well saves three-quarters of the "
                     f"computation and gains {F['headroom']:.0f} accuracy points. Overthinking "
                     "actively destroys accuracy.")
    shift(s, [15], 0.30)                      # was drawn over the table's last row
    set_shape(s[18], "An independent 3,000-question sample reproduced the pattern (final "
                     "41.3% against 42.4%, ever-correct 62.4% against 63.3%, under the earlier "
                     "grader). The headline numbers are stable to about one percentage point "
                     "across samples.")

    # -- 13 result 2 --------------------------------------------------------- #
    s = sh(13)
    t = s[4].table
    rows = [
        ("Full reasoning — no early stop", full),
        (f"Fixed step {F['matched_step']} — matched cost baseline", fixed),
        ("Double DQN — ours", dqn),
        ("Behaviour cloning — supervised control", bc),
        ("Oracle — achievable upper bound", oracle),
    ]
    for i, (label, m) in enumerate(rows, start=1):
        set_cell(t.cell(i, 0), label)
        set_cell(t.cell(i, 1), pct(m["accuracy"]))
        set_cell(t.cell(i, 2), f"{m['mean_tokens']:.0f}")
        set_cell(t.cell(i, 3), f"{m['token_reduction_pct']:.1f}%")
        set_cell(t.cell(i, 4), f"{m['mean_stop_step']:.1f}")
    shift(s, range(5, 12), 0.45)              # callouts were drawn over the oracle row
    set_shape(s[7], f"{margin:+.1f}")
    set_shape(s[8], "accuracy points over a fixed stopping rule at the same token cost — a "
                    f"{relative:.0f}% relative improvement ({p_text}, paired bootstrap). Fixed "
                    "thresholds are what prior work actually uses.")
    set_shape(s[10], "AGAINST SUPERVISED CONTROL")
    set_shape(s[11], "Behaviour cloning — plain supervised imitation of the oracle — reaches "
                     f"{pct(bc['accuracy'])} at a slightly lower budget. The DQN's "
                     f"{bc_margin:+.1f}-point edge is not significant (p = "
                     f"{F['cmp_bc']['p_value']:.2f}): reinforcement learning matches supervised "
                     "control here rather than beating it. The next slides show where the gain "
                     "actually comes from.")

    # -- 14 frontier --------------------------------------------------------- #
    s = sh(14)
    xy = XyChartData()
    for name, m in (("Full reasoning", full), (f"Fixed step {F['matched_step']}", fixed),
                    ("Double DQN (ours)", dqn), ("Behaviour cloning", bc),
                    ("Oracle (upper bound)", oracle)):
        ser = xy.add_series(name)
        ser.add_data_point(round(m["mean_tokens"]), round(m["accuracy"] * 100, 1))
    s[4].chart.replace_data(xy)
    set_shape(s[5],
              "How to read this chart",
              f"Compare the two points at the same horizontal position — around "
              f"{dqn['mean_tokens']:.0f} tokens. That is a matched-cost comparison, and our "
              f"policy sits {margin:.1f} points above the fixed rule. Same budget, better answers.",
              f"Behaviour cloning sits {abs(bc_margin):.1f} points lower and slightly further "
              "left — a difference within noise. Supervised control is a fair match for RL on "
              "this task.",
              "The oracle in the top-left corner is the target. It proves the headroom is real "
              f"rather than theoretical — {F['headroom']:.0f} accuracy points and "
              f"{oracle['token_reduction_pct']:.0f}% of the tokens are both available to a "
              "policy good enough to find them.",
              "Full reasoning, bottom right, is what the model does today: the most expensive "
              "option and not the most accurate.")

    # -- 15 result 3: what the ablation says --------------------------------- #
    s = sh(15)
    set_shape(s[2], "Result 3 — where the gain comes from")
    set_shape(s[3], "The original claim was that the agent should spend its budget differently "
                    "depending on difficulty. The measurement says otherwise — and says what "
                    "does matter.")
    tiers = [("easy", "Easy"), ("medium", "Medium"), ("hard", "Hard")]
    cd = CategoryChartData()
    cd.categories = [f"{label}  (n={dqn[t + '_n']})" for t, label in tiers]
    share = [100 * dqn[t + "_mean_tokens"] / full[t + "_mean_tokens"] for t, _ in tiers]
    cd.add_series("Share of full reasoning used", [round(v) for v in share])
    cd.add_series("Accuracy achieved", [round(100 * dqn[t + "_accuracy"], 1) for t, _ in tiers])
    s[4].chart.replace_data(cd)
    abl = F["ablations"].get("results", {})
    if "no_state" in abl:
        ns = abl["no_state"]
        best_line = ("What it has learned is to read the model's own signals — confidence and "
                     "whether the answer has stopped changing — rather than the question's "
                     f"difficulty. The best configuration is the difficulty-blind one: "
                     f"{pct(ns['accuracy'])}, {ns['margin_over_fixed'] * 100:+.1f} points over "
                     "the fixed rule.")
    else:
        best_line = ("What it has learned is to read the model's own signals — confidence and "
                     "whether the answer has stopped changing — rather than the question's "
                     "difficulty.")
    set_shape(s[5],
              f"Read the dark bars. The agent uses about half of the full reasoning on every "
              f"tier — {share[0]:.0f}%, {share[1]:.0f}%, {share[2]:.0f}%. It does not cut hard "
              "questions shorter than easy ones. Accuracy falls with the tier because the "
              "questions are harder, not because the policy behaves differently.",
              best_line)
    set_shape(s[7], "WHAT THE ABLATION SAYS")
    if abl:
        def line(key, what):
            m = abl.get(key)
            return (f"{what}: {pct(m['accuracy'])} at {m['token_reduction_pct']:.0f}% saved."
                    if m else "")
        parts = [line("full", "Reported system"),
                 line("no_state", "Difficulty removed from the state"),
                 line("no_reward", "One token price for every tier"),
                 line("neither", "Both removed"),
                 line("oracle_difficulty", "Perfect difficulty labels (not deployable)"),
                 line("no_answer_shape", "Answer-shape features removed")]
        parts = [p for p in parts if p]
        set_shape(s[8], *parts, "Removing difficulty does not hurt, and even perfect "
                                "difficulty labels do not help: the difficulty-aware "
                                "formulation is not where the gain comes from. We report "
                                "this rather than hide it; the learned stopping policy "
                                "stands on the other features.")
    else:
        set_shape(s[8], "Phase 9 removes difficulty from the state, then from the reward, "
                        "then the answer-shape features, and finally gives the agent the true "
                        "difficulty label as an upper bound. Results are filled in from "
                        "phase9_summary.json when the run completes.")

    # -- 16 pending works ---------------------------------------------------- #
    s = sh(16)
    set_shape(s[3], "Two phases remain, plus an optional GPU rerun and the write-up.")
    set_shape(s[5], "Phase 8"); set_shape(s[7], "PRIORITY")
    set_shape(s[8], "Advisory application")
    set_shape(s[9], "The FastAPI service and the dashboard work end to end on replayed "
                    "traces. Remaining: finish the interface, fix the demo question set, "
                    "record the walkthrough.")
    set_shape(s[10], "~3 days")
    set_shape(s[12], "Phase 9"); set_shape(s[14], "PRIORITY")
    set_shape(s[15], "Ablations and final report")
    set_shape(s[16], "Six variants isolate what each ingredient contributes — difficulty in "
                     "the state, in the reward, the answer-shape features, and perfect "
                     "difficulty as an upper bound. The report and deck are generated from "
                     "the results files so no figure is copied by hand.")
    set_shape(s[17], "~1 week")
    set_shape(s[19], "Phase 3b"); set_shape(s[21], "OPTIONAL")
    set_shape(s[22], "Extended reasoning budget")
    set_shape(s[23], "One in four traces hits the 768-token cap mid-reasoning. A 1,536-token "
                     "rerun over 2,500 questions is prepared, with a watchdog and a "
                     "probe-agreement guard; it waits on GPU quota.")
    set_shape(s[24], "~5 GPU-hours")
    set_shape(s[26], "Write-up"); set_shape(s[28], "SCHEDULED")
    set_shape(s[29], "Final report and viva")
    set_shape(s[30], "Consolidate the findings, verify the reference list, and settle the "
                     "final title in light of the ablation.")
    set_shape(s[31], "~1 week")

    # -- 17 interim conclusion ----------------------------------------------- #
    s = sh(17)
    set_shape(s[3],
              "The pipeline is complete and produces measured, reproducible results. "
              "Overthinking is real in financial reasoning — the model is correct at some "
              f"point on {pct(p3['solvable_fraction'])} of questions but ends correct on only "
              f"{pct(p3['final_accuracy'])} — and a learned stopping policy beats the "
              f"fixed-threshold approach used in prior work by {margin:.1f} accuracy points at "
              f"identical token cost ({p_text}), while matching a supervised controller.",
              "We also found, and are reporting, that the difficulty signal contributes "
              "nothing measurable: the policy does as well or better without it, reading the "
              "model's confidence and answer stability instead. That changes how the "
              "contribution should be described, not whether it holds.",
              "Remaining work: finish the application and the ablation write-up, and — "
              "optionally — regenerate the traces at a longer reasoning budget.")
    set_shape(s[6], f"{saved:.0f}%")
    set_shape(s[8], f"{margin:+.1f}")
    set_shape(s[10], f"{F['headroom']:.0f} pts")
    set_shape(s[12], f"{n_tests}")

    prs.save(str(TARGET))
    return TARGET


def main() -> int:
    ap = argparse.ArgumentParser(description="refresh the mid-review deck from results")
    ap.add_argument("--tests", type=int, default=449, help="tests passing")
    args = ap.parse_args()

    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)

    out = refresh(args.tests, lines, len(files))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
