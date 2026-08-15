# Findings

Empirical results, including the ones that contradicted the original hypothesis. All
figures are on the held-out test split (599 questions, 4,000 traced in total) using
`DeepSeek-R1-Distill-Qwen-1.5B` at a 768-token budget.

---

## 1. Overthinking is real, and it costs accuracy

| | |
|---|---|
| Correct at the **end** of reasoning | **42.4%** |
| Correct at **some point** during reasoning | **63.3%** |

On roughly **one question in five, the model reaches the right answer and then talks
itself out of it.** This is the premise of the project, measured rather than assumed.

Supporting evidence — the probability that the current answer is correct, by step:

| Step | 0 | 1 | 3 | 6 | 10 | 15 |
|---|---|---|---|---|---|---|
| Correct | 21.5% | 21.1% | 25.9% | 32.5% | 32.2% | 31.3% |

**Accuracy plateaus at about step 6 and never improves again.** The model reasons for a
mean of 11.5 steps, so roughly half of all reasoning is spent after the point where it
stops helping.

## 2. A perfect stopping policy would be both cheaper *and* more accurate

| Policy | Accuracy | Mean tokens |
|---|---|---|
| Full reasoning | 42.4% | 533 |
| Oracle (stops at the earliest correct step) | **62.9%** | **135** |

This is not a trade-off. Stopping well would save ~75% of tokens *and* gain ~20
accuracy points. That reframes the contribution: overthinking actively destroys
accuracy, and good stopping recovers it.

## 3. Learned stopping clearly beats fixed-rule stopping

At matched cost (~65% of tokens saved):

| Policy | Accuracy |
|---|---|
| Stop at a fixed step | 26.9% |
| **Difficulty-aware DQN** | **32.4%** |

**+5.5 accuracy points at the same cost.** Fixed thresholds are what prior work
actually uses, so this is the comparison the contribution rests on.

A conservative operating point is also available: **24% fewer tokens for a 1-point
accuracy cost** (39.6% vs 40.6%).

## 4. But reinforcement learning does *not* beat supervised imitation

Compared at matched cost (~52% of tokens saved), selecting each on validation by
"maximise accuracy subject to the cost budget":

| Policy | Accuracy |
|---|---|
| Behaviour cloning (supervised) | **37.2%** |
| DQN (reinforcement learning) | 34.6% |

The DQN is ~2.7 points behind, and this held across four reward cost-multipliers
(0.05, 0.1, 0.2, 0.35), so it is not a tuning artefact.

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
| Easy | 427 tokens | 233 | **54%** | 76.6% |
| Medium | 540 tokens | 216 | **39%** | 44.8% |
| Hard | 585 tokens | 159 | **30%** | 9.0% |

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
