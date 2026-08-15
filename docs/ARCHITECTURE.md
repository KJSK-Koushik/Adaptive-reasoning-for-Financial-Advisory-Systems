# System architecture

## The problem

Reasoning LLMs generate a long chain of thought before answering. They often reach the
correct answer early and then keep reasoning — re-checking, second-guessing, exploring
alternatives — with no accuracy gain. Those extra tokens are pure cost: latency, GPU
memory, energy, money.

Existing fixes use **fixed rules**: stop at N tokens, or stop when confidence exceeds τ.
A fixed rule cannot be right for both "is this sentence positive or negative?" and a
multi-step cash-flow calculation. It also cannot learn.

## The contribution

A **Deep Q-Network that learns a stopping policy**, conditioned on predicted question
difficulty, with a **difficulty-aware reward**. Two things distinguish it from prior work:

1. Difficulty enters the **state**, so the policy can behave differently per tier.
2. Difficulty enters the **reward**, via a per-token cost `β(difficulty)` where
   `β_easy > β_medium > β_hard`. Wasting tokens on an easy question is punished hard;
   on a hard question the agent is given room to think.

## Offline RL — the decision that makes this feasible

Training a stopping policy online means calling the LLM every time the agent picks
CONTINUE. That is millions of LLM calls and is not achievable on student hardware.

Instead, Phase 3 records **early-exit probes** at every step of a full-length reasoning
trace. At each step boundary we append `"Therefore, the final answer is"`, decode ~24
tokens from the cached state, record the answer and its confidence/entropy, then discard
the probe and continue reasoning from the same KV cache.

The result is, for each question, a complete table of *what would have happened had we
stopped at step t*. Because the entire future is known, the reward for **both** actions is
computable in closed form, so DQN training needs **zero LLM calls**. Trace generation is
paid once; the policy can then be retrained fifty times in minutes while tuning rewards.

```
Phase 3 (GPU, once)              Phases 4-5 (CPU, minutes, repeatable)
┌──────────────────────┐         ┌──────────────────────────────────┐
│ LLM reasons to full  │         │ Offline env replays traces       │
│ length; probe at     │  ────►  │ DQN learns Q(state, stop/continue)│
│ every step boundary  │         │ Reward tuning is now cheap        │
└──────────────────────┘         └──────────────────────────────────┘
```

## Pipeline

```
Datasets ──► unified.parquet ──► difficulty labels ──► classifier
                                        │
                                        ▼
                          reasoning traces + step probes
                                        │
                                        ▼
                          transitions (state, action, reward)
                                        │
                                        ▼
                              stopping_policy.pt
```

## Runtime path

```
query ──► difficulty classifier ──► LLM begins streaming reasoning
                                          │
                            every step:  build 14-dim state
                                          │
                                    DQN ──┴──► CONTINUE (loop)
                                          └──► STOP ──► force answer
                                                        + savings report
```

Safety nets in `serve`: a `hard_token_cap` the policy cannot exceed, and
`min_steps_before_stop` so it can never answer with zero reasoning.

## State vector (14 features)

| Group | Features |
|---|---|
| Difficulty | `difficulty_easy`, `difficulty_medium`, `difficulty_hard` (one-hot) |
| Certainty | `confidence`, `min_token_confidence`, `entropy`, `entropy_slope`, `delta_confidence` |
| Budget | `token_ratio`, `step_index_norm` |
| Convergence | `answer_stability`, `steps_since_answer_change` |
| Text cues | `progress_cue` ("therefore", "so"), `doubt_cue` ("wait", "actually") |

## Reward

```
STOP      R = +1 if correct else -1   -   β(difficulty) · tokens_used / budget
                                      +   stability_bonus if answer was stable
CONTINUE  R = -β(difficulty) · tokens_in_next_step / budget
```

All coefficients live in `configs/default.yaml` under `rl.reward`.

## Evaluation

Eight policies are compared on identical traces: full reasoning, fixed budget, confidence
threshold, entropy threshold, answer stability, random, **oracle** (earliest correct stop —
the achievable ceiling), and ours. Reported as an accuracy-vs-cost Pareto curve plus a
table of accuracy, mean tokens, token reduction %, p50/p95 latency, peak memory and
energy (Joules, sampled via NVML).

Ablations that must appear in the report: remove difficulty from the state; remove
difficulty from the reward; remove entropy; DQN vs the best fixed threshold.

## Honest expectations

Early stopping trades a little accuracy for a lot of tokens. The target is a strong
trade — roughly 40–55% fewer reasoning tokens for ≤1–2% accuracy loss — not a free lunch.
The evaluation is built to surface the failure cases (stopped too early, wrong answer)
rather than hide them.
