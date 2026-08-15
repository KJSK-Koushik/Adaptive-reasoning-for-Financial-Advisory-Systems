"""Building the offline RL dataset from reasoning traces.

For every probe point we emit **both** transitions - what happens if the agent stops,
and what happens if it continues - because the trace records the entire future. That is
the whole reason Phase 5 needs no LLM calls: the environment is already fully known, so
the DQN can be trained, tuned and retrained in minutes on a CPU.

Terminal handling: at the last probe point there is nothing left to continue into, so
CONTINUE is emitted with the same outcome as STOP and marked terminal. Dropping it
instead would leave the network with no gradient for CONTINUE in exactly the states
where it most needs to have learned to stop.
"""

from __future__ import annotations

import json

import numpy as np

from .. import paths
from ..config import Config
from ..logging_utils import get_logger
from .features import build_states, difficulty_one_hot
from .reward import StepOutcome, continue_reward, oracle_return, stop_reward, token_cost

log = get_logger("rl.dataset")

ACTION_CONTINUE = 0
ACTION_STOP = 1


def _outcomes(steps: list[dict]) -> list[StepOutcome]:
    outcomes = []
    for i, step in enumerate(steps):
        next_tokens = (
            steps[i + 1]["tokens_so_far"] - step["tokens_so_far"]
            if i + 1 < len(steps) else 0
        )
        outcomes.append(
            StepOutcome(
                tokens_so_far=int(step["tokens_so_far"]),
                probe_correct=bool(step["probe_correct"]),
                answer_changed=bool(step["answer_changed"]),
                tokens_in_next_step=max(0, int(next_tokens)),
            )
        )
    return outcomes


def transitions_for_trace(
    question_id: str,
    steps: list[dict],
    difficulty: str | None,
    difficulty_vector: np.ndarray,
    cfg: Config,
    budget: int,
) -> list[dict]:
    """All (state, action, reward, next_state, done) tuples for one question."""
    if not steps:
        return []

    states = build_states(steps, difficulty_vector, cfg, budget)
    outcomes = _outcomes(steps)
    last = len(steps) - 1

    rows: list[dict] = []
    for i, outcome in enumerate(outcomes):
        state = states[i]
        r_stop = stop_reward(outcome, cfg, difficulty, budget)
        terminal_state = np.zeros_like(state).tolist()

        # One definition of the shared columns. Three near-identical literals here
        # previously drifted apart: two of them silently lost the fields
        # recompute_rewards depends on, which produced NaN rewards and a DQN that
        # trained on nothing without ever raising.
        base = {
            "question_id": question_id,
            "step_index": i,
            "difficulty": difficulty,
            "state": state.tolist(),
            "probe_correct": outcome.probe_correct,
            "tokens_so_far": outcome.tokens_so_far,
            # Kept so rewards can be recomputed for a different cost multiplier
            # without rebuilding features - that is what makes the Phase 5 sweep cheap.
            "tokens_in_next_step": outcome.tokens_in_next_step,
            "answer_changed": outcome.answer_changed,
        }

        rows.append(base | {
            "action": ACTION_STOP,
            "reward": r_stop,
            "done": True,
            "next_state": terminal_state,
        })

        if i < last:
            rows.append(base | {
                "action": ACTION_CONTINUE,
                "reward": continue_reward(outcome, cfg, difficulty, budget),
                "done": False,
                "next_state": states[i + 1].tolist(),
            })
        else:
            # Nothing left to continue into: the model has finished generating.
            rows.append(base | {
                "action": ACTION_CONTINUE,
                "reward": r_stop,
                "done": True,
                "next_state": terminal_state,
            })

    return rows


def recompute_rewards(frame, cfg: Config, budget: int):
    """Recompute the reward column for the current config, in place on a copy.

    The state features do not depend on the reward, so sweeping ``cost_multiplier``
    only needs the rewards redone. Rebuilding the whole table instead would mean
    re-running the sentence encoder for every point on the frontier.
    """
    import numpy as np

    frame = frame.copy()
    r = cfg.rl.reward

    costs = frame["difficulty"].map(
        lambda d: token_cost(cfg, d if isinstance(d, str) else None)
    ).to_numpy(dtype=float)

    correct = frame["probe_correct"].to_numpy().astype(bool)
    tokens = frame["tokens_so_far"].to_numpy(dtype=float)
    next_tokens = frame["tokens_in_next_step"].to_numpy(dtype=float)
    changed = frame["answer_changed"].to_numpy().astype(bool)

    stop = np.where(correct, r.correct_bonus, r.incorrect_penalty)
    stop = stop - costs * tokens / max(budget, 1)
    stop = stop + np.where(changed, 0.0, r.stability_bonus)

    cont = -costs * next_tokens / max(budget, 1)

    is_stop = frame["action"].to_numpy() == ACTION_STOP
    is_terminal_continue = (~is_stop) & frame["done"].to_numpy().astype(bool)

    rewards = np.where(is_stop | is_terminal_continue, stop, cont)

    if not np.isfinite(rewards).all():
        n_bad = int((~np.isfinite(rewards)).sum())
        raise ValueError(
            f"{n_bad} of {len(rewards)} recomputed rewards are not finite. This means "
            f"a column recompute_rewards depends on is missing or null - check "
            f"tokens_in_next_step and answer_changed, and rebuild with "
            f"scripts/run_phase4.py."
        )

    frame["reward"] = rewards
    return frame


def build(cfg: Config, write: bool = True) -> dict:
    """Build the full transition table from ``traces.parquet``."""
    import pandas as pd

    from ..difficulty.from_traces import observed_budget
    from ..traces.runner import TRACE_SUMMARY

    if not paths.TRACE_DATASET.exists():
        raise FileNotFoundError(
            f"{paths.TRACE_DATASET} not found - run scripts/run_phase3.py first"
        )

    steps_frame = pd.read_parquet(paths.TRACE_DATASET)
    summary = pd.read_parquet(TRACE_SUMMARY)
    budget = observed_budget(summary, cfg)

    unified = pd.read_parquet(paths.UNIFIED_DATASET)[["id", "difficulty", "split"]]
    meta = unified.set_index("id")

    vectors = _difficulty_vectors(cfg, meta, steps_frame)

    rows: list[dict] = []
    oracle_values: list[float] = []
    n_traces = 0

    for question_id, group in steps_frame.groupby("question_id", sort=False):
        steps = group.sort_values("step_index").to_dict("records")
        difficulty = meta.at[question_id, "difficulty"] if question_id in meta.index else None
        if difficulty is not None and difficulty != difficulty:      # NaN
            difficulty = None

        rows.extend(
            transitions_for_trace(
                str(question_id), steps, difficulty,
                vectors.get(str(question_id), difficulty_one_hot(difficulty)),
                cfg, budget,
            )
        )
        value, _ = oracle_return(_outcomes(steps), cfg, difficulty, budget)
        oracle_values.append(value)
        n_traces += 1

    frame = pd.DataFrame(rows)
    split_map = unified.set_index("id")["split"]
    frame["split"] = frame["question_id"].map(split_map).fillna("train")

    summary_stats = {
        "n_traces": n_traces,
        "n_transitions": int(len(frame)),
        "state_dim": cfg.rl.state_dim,
        "token_budget": budget,
        "mean_oracle_return": round(float(np.mean(oracle_values)), 4),
        "stop_reward_mean": round(float(frame[frame.action == ACTION_STOP].reward.mean()), 4),
        "continue_reward_mean": round(
            float(frame[(frame.action == ACTION_CONTINUE) & (~frame.done)].reward.mean()), 4
        ),
        "by_split": frame.groupby("split").size().to_dict(),
        "by_difficulty": frame.groupby("difficulty").size().to_dict(),
    }

    if write:
        paths.TRACES.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(paths.RL_TRANSITIONS, index=False)
        log.info("wrote %s (%d rows)", paths.RL_TRANSITIONS, len(frame))

        path = paths.RESULTS / "phase4_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary_stats, indent=2, default=str), encoding="utf-8")

    return summary_stats


def _difficulty_vectors(cfg, meta, steps_frame) -> dict[str, np.ndarray]:
    """Difficulty features for each question, per ``rl.difficulty_source``.

    ``predicted`` uses the classifier's probabilities, which is what the live system
    will see - training on ground-truth one-hots and deploying on a 60%-accurate
    classifier would be a train/serve mismatch that flatters the offline numbers.
    ``true`` uses measured labels and exists for the ablation.
    """
    source = cfg.rl.difficulty_source
    question_ids = [str(q) for q in steps_frame["question_id"].unique()]

    if source == "none":
        neutral = np.full(3, 1 / 3, dtype=np.float32)
        return dict.fromkeys(question_ids, neutral)

    if source == "true":
        return {
            qid: difficulty_one_hot(
                meta.at[qid, "difficulty"] if qid in meta.index else None
            )
            for qid in question_ids
        }

    import pandas as pd

    from ..difficulty.classifier import DifficultyClassifier

    if not paths.DIFFICULTY_MODEL.exists():
        log.warning(
            "no trained classifier at %s; falling back to measured labels. Run "
            "`python scripts/run_phase2.py --stage train` for a faithful state.",
            paths.DIFFICULTY_MODEL,
        )
        return {
            qid: difficulty_one_hot(
                meta.at[qid, "difficulty"] if qid in meta.index else None
            )
            for qid in question_ids
        }

    unified = pd.read_parquet(paths.UNIFIED_DATASET)[["id", "question", "context"]]
    subset = unified[unified.id.isin(question_ids)]
    model = DifficultyClassifier.load(cfg)
    probabilities = model.predict_proba(
        subset.question.tolist(), subset.context.fillna("").tolist()
    )
    log.info("using classifier probabilities for the difficulty features")
    return {
        str(qid): probabilities[i].astype(np.float32)
        for i, qid in enumerate(subset.id.tolist())
    }
