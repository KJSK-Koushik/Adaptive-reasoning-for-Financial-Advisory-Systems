# Findings

Empirical results, including the ones that contradicted the original hypothesis. All
figures are on the held-out test split (599 questions, 4,000 traced in total) using
`DeepSeek-R1-Distill-Qwen-1.5B` at a 768-token budget.

All numbers were regenerated after the grading fix of 2026-08-23: the parser was
multiplying an explicit scale word ("407 million") by its scale while the gold answer
states the unit in the question and stores a bare `407`, so correct answers were being
marked wrong. Re-grading turned 858 probe answers from wrong to right and none the
other way. See `scripts/regrade_traces.py`.

**Reproduce every number below with:**

```
python scripts/run_phase5.py --experiment reported
```

That config differs from the default in one setting: the best DQN checkpoint is
chosen on validation by `weighted_score` rather than `accuracy_at_budget`, which
fixes the operating point at ~65% of tokens saved. The default lands at ~45%
saved and 37.1% accuracy - a legitimate result, but a different point on the same
curve. Figures quoted here are all at the 65% point unless stated otherwise.

---

## 1. Overthinking is real, and it costs accuracy

| | |
|---|---|
| Correct at the **end** of reasoning | **44.3%** |
| Correct at **some point** during reasoning | **65.5%** |

On roughly **one question in five, the model reaches the right answer and then talks
itself out of it.** This is the premise of the project, measured rather than assumed.

Supporting evidence — the probability that the current answer is correct, by step:

| Step | 0 | 1 | 3 | 6 | 10 | 15 |
|---|---|---|---|---|---|---|
| Correct | 22.6% | 22.8% | 28.3% | 34.7% | 33.7% | 32.9% |

**Accuracy plateaus at about step 6 and never improves again.** The model reasons for a
mean of 11.5 steps, so roughly half of all reasoning is spent after the point where it
stops helping.

## 2. A perfect stopping policy would be both cheaper *and* more accurate

| Policy | Accuracy | Mean tokens |
|---|---|---|
| Full reasoning | 44.7% | 538 |
| Oracle (stops at the earliest correct step) | **65.3%** | **137** |

Both rows are the 599 test questions. (Across all 4,000 traces full reasoning is
44.3% at 533 tokens.)

This is not a trade-off. Stopping well would save ~75% of tokens *and* gain ~21
accuracy points. That reframes the contribution: overthinking actively destroys
accuracy, and good stopping recovers it.

## 3. Learned stopping clearly beats fixed-rule stopping

At matched cost (~45% of tokens saved):

| Policy | Accuracy |
|---|---|
| Stop at a fixed step | 32.1% |
| Stop at a fixed token budget | 32.1% |
| Confidence threshold | 30.2% |
| Entropy threshold | 26.5% |
| **Difficulty-aware DQN** | **39.2%** |

**+7.2 accuracy points over a fixed step at the same cost** (95% CI +3.3 to +10.9,
p = 0.0004). Confidence and entropy thresholds are what the early-exit literature
actually uses, and the DQN beats both by more (+9.0 and +12.7). This is the comparison
the contribution rests on.

### The operating point is a choice, and it has to be stated

How the best checkpoint is selected on validation decides where on the frontier the
policy sits, and the two objectives land somewhere quite different:

| Selection | Tokens saved | Accuracy | vs fixed step | vs behaviour cloning |
|---|---|---|---|---|
| `accuracy_at_budget` (reported) | 45% | **39.2%** | +7.2, p=0.0004 | +0.2, p=1.0 |
| `weighted_score` (aggressive) | 65% | 33.7% | +4.5, p=0.027 | -5.3, p=0.006 |

The reported point states a budget. The aggressive one uses `accuracy + 0.30 x
token_reduction`, where the 0.30 silently prices a token against accuracy: at that
weight the policy stops at **step 0 on 54% of questions**, where accuracy is 21%. Both
are honest points on one frontier, but a stated budget is defensible in a way an
arbitrary weight is not, so the reported figures use the former.

## 4. Reinforcement learning matches supervised imitation, but does not beat it

| Policy | Accuracy | Tokens saved |
|---|---|---|
| Behaviour cloning (supervised) | 39.1% | **53.5%** |
| DQN (reinforcement learning) | **39.2%** | 44.7% |

On accuracy this is a dead heat (+0.2 points, p = 1.0). But behaviour cloning reaches
it **8.8 points of budget cheaper**, so it still edges the DQN on the trade-off rather
than losing to it. Reporting the accuracy tie alone would flatter the DQN, which is
why Phase 6 labels any comparison where the two policies spend differently.

Behaviour cloning imitates the oracle and therefore has exactly one operating point;
the DQN can be moved along the frontier. That is the honest statement of what each
method offers.

An earlier version of this document reported the DQN 5.3 points *behind*. That figure
was real but taken at the aggressive operating point above - a reminder that a
single-number comparison between two policies at different costs says as much about
the operating points as about the methods.

### Why - and it is not the reward

The first hypothesis was that the reward's flat −1 for a wrong answer made quitting
attractive on hard questions. That is probably not it: the reward already handles this
through bootstrapping, since on a trace that becomes correct at step 8 continuing is
worth about +0.8 against −1 for stopping, and TD learning propagates that backwards.

The more likely explanation is **the state cannot distinguish a recoverable question
from a hopeless one.** Measured as AUC for "would stopping here be correct":

| Feature | AUC |
|---|---|
| difficulty (predicted) | 0.725 |
| confidence | 0.581 |
| min token confidence | 0.575 |
| answer stability | 0.558 |
| entropy | 0.556 |
| progress / doubt cues | 0.51-0.53 |
| **all features combined** | **0.730** |

Almost all the signal is *question-level* (how hard is this question) rather than
*step-level* (is the answer right yet). At step 2 a recoverable question and a hopeless
one look nearly identical, 63% of hard questions are never correct, and the network
averages over them and concludes "stop".

### The structural argument

RL earns its keep when actions change what happens next. Here they do not. The traces
are fixed recordings: choosing CONTINUE does not alter the reasoning, it only reveals
the step that was always going to come. The agent influences *when it exits*, nothing
else.

That makes this much closer to a **prediction** problem than a **control** problem, and
a well-calibrated one-step predictor with a threshold - which is essentially what
behaviour cloning is - is correspondingly hard to beat.

### What the DQN still offers

Behaviour cloning imitates the oracle and yields exactly one behaviour. The DQN's
reward exposes a **tunable cost-accuracy dial**: sweeping `reward.cost_multiplier`
produces a family of policies along the frontier, so an operating point can be chosen
for the deployment rather than accepted from the method.

**Measured before the 2026-08-23 grading fix and not yet regenerated** - the shape of
the frontier holds but every accuracy here is understated by roughly 2 points. Rerun
with `python scripts/run_phase5.py --experiment reported --sweep` to refresh.


| cost_multiplier | Accuracy | Tokens saved |
|---|---|---|
| 0.25 | 35.1% | 46.2% |
| 0.5 | 35.1% | 48.9% |
| 1.0 | 32.6% | 57.9% |
| 2.0 | 29.5% | 70.7% |

## 5. Difficulty-awareness works - in the opposite direction to the hypothesis

The original hypothesis was *easy questions → stop early; hard questions → be patient*.
The learned policy does the reverse:

| Tier | Unaided reasoning | Stopped at | Share used | Accuracy |
|---|---|---|---|---|
| Easy | 427 tokens | 258 | **60%** | 83.7% |
| Medium | 540 tokens | 264 | **49%** | 60.8% |
| Hard | 585 tokens | 327 | **56%** | 12.3% |

At the reported operating point the allocation is close to flat - the budget is loose
enough that the policy does not have to choose. At the aggressive point it becomes
strongly difficulty-ordered:

| Tier | Full reasoning | DQN tokens | Share used | Accuracy |
|---|---|---|---|---|
| Easy | 427 tokens | 222 | **52%** | 77.3% |
| Medium | 540 tokens | 224 | **41%** | 45.6% |
| Hard | 585 tokens | 166 | **28%** | 10.8% |

So difficulty-awareness shows up **when the budget binds**, which is the situation it
was designed for. Claiming it as an unconditional property of the policy would
overstate it.

Note that absolute token counts are misleading: hard questions get *longer* traces to
begin with, so the policy is in fact cutting them hardest (73% saved on hard vs 46% on
easy).

And it is right to. On hard questions the model is correct only 9% of the time however
long it thinks, because 63% of them are never correct at any point. Extra thinking pays
off only when the answer is *reachable*.

**The useful signal is not "is this question hard?" but "is this answer reachable?"**

---

## Honest summary

* The premise is validated: overthinking is real and costly.
* Learned stopping substantially beats the fixed-threshold methods used in prior work.
* Among learned policies, supervised imitation currently edges out reinforcement
  learning at matched cost; the RL formulation's advantage is tunability, not accuracy.
* The difficulty-aware mechanism works, but the mechanism it discovered is about
  answer reachability rather than question difficulty.
