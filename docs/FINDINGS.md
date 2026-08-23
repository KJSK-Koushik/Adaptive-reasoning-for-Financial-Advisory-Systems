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

### A known under-count in the grading

Probe answers are capped at 16 tokens (`traces.probe_max_tokens`), so a longer answer
is cut off mid-sentence and the trailing fragment is often a partial number:

    gold 4751   <- "a decrease of $4,751 in operating income from 20"
    gold 150    <- "150 units. Wait, let me double-check that. 3"

`extract_numbers` reads the *last* number in a string, which is the right rule for a
finished answer ("...so the growth is 12.4%") and the wrong one for a truncated one.
Roughly **295 probe rows, 0.6% of the total, are marked wrong this way** - accuracy
here is therefore a slight under-count.

It has not been fixed. Preferring the first number instead rescues those 295 rows but
introduces the opposite error, and the data contains clear examples:

    gold 16     <- "16.0 divided by 3, which equals approximately 5"

where the answer is 5.33 and accepting the leading 16 would score a wrong answer as
right. A grader that marks wrong answers right inflates results; one that marks some
right answers wrong understates them. The second is the safer failure, so the
under-count stands and is reported rather than removed.

The root cause is the 16-token probe limit, and fixing it properly means regenerating
traces on GPU.

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

## 7. Ablations: difficulty-awareness does not help, and removing it helps

Every result above compares the whole system against baselines, which cannot separate
"difficulty-awareness helps" from "any learned policy helps". Phase 9 removes each
ingredient and retrains from scratch. All five variants are measured against the same
fixed-step baseline (32.1%) at nearly the same operating point, so the margin column is
directly comparable.

| Variant | Accuracy | Tokens saved | Margin over fixed step | vs the full system |
|---|---|---|---|---|
| full - the reported system | 40.1% | 47.8% | +8.0 | - |
| **no difficulty in the state** | **44.1%** | 46.6% | **+12.0** | **+4.0, p = 0.018** |
| **difficulty-blind entirely** | **43.9%** | 44.4% | **+11.9** | **+3.8, p = 0.024** |
| uniform token price | 41.4% | 46.2% | +9.3 | +1.3, p = 0.44 |
| without answer-shape features | 39.2% | 44.7% | +7.2 | -0.8, p = 0.67 |

**Removing difficulty from the state makes the policy significantly better**, by 4
accuracy points. Removing it from the reward as well changes nothing further. The
difficulty-aware formulation - the project's stated novelty - is not contributing; it
is costing about four points.

### Why

The policy does not see *measured* difficulty. It sees the Phase 2 classifier's
prediction, because that is all a live system could have. That classifier scores 59.9%
against a 55.6% majority baseline, with 16.7% recall on the medium class - it is barely
better than guessing "hard" every time. Three of the eighteen state dimensions are
therefore close to noise, and the network pays for them.

The reward ablation is the cleaner test of the idea itself, because the reward uses the
*measured* label rather than the prediction. Making the token price uniform changed
accuracy by 1.3 points, which is not significant (p = 0.44). So even with a perfect
difficulty signal in the reward, difficulty-aware pricing did not earn its place here.

### Could a better difficulty classifier rescue it? No.

The obvious objection to the above is that the policy only sees a 59.9% classifier, so
perhaps the idea is sound and the signal is the problem. That is testable: put the
*measured* difficulty label in the state. It is not deployable - a live system cannot
know it, and approximating it is precisely the classifier's job - but it is an upper
bound on what any difficulty signal could ever buy.

| Variant | Accuracy | Tokens saved | Margin over fixed step |
|---|---|---|---|
| difficulty-blind | 44.1% | 46.6% | **+12.0** |
| **perfect difficulty in the state** | 43.6% | 52.7% | **+11.8** |
| predicted difficulty (the reported system) | 40.1% | 47.8% | +8.0 |

**A perfect difficulty signal performs the same as no difficulty signal at all**
(+11.8 against +12.0). Handed the exact label it was trying to predict, the policy gains
nothing in accuracy. So the noisy classifier is not the bottleneck and improving it
would not change the conclusion: on this data, conditioning the stopping decision on
question difficulty does not help.

One honest nuance: the oracle variant reaches its margin at a *cheaper* operating point
(52.7% of tokens saved against 46.6%), so perfect difficulty does buy something - about
six points of budget at equal accuracy advantage. That is a real but small effect, and
it is not the claim the project was making.

### What survives

The central result is unaffected, and in fact stronger: **a learned stopping policy
beats fixed and threshold rules at matched cost.** The best configuration found is the
difficulty-blind one, at **+12.0 points over a fixed step** rather than +8.0.

### A correction

An earlier version of this document called the answer-shape features "the largest
single improvement in the project". They lift stop-correctness AUC from 0.700 to 0.767,
which is real, but the ablation puts their effect on final accuracy at 0.8 points and
not significant (p = 0.67). The AUC gain did not translate.


## 3. Learned stopping clearly beats fixed-rule stopping

At matched cost (~48% of tokens saved):

| Policy | Accuracy |
|---|---|
| Stop at a fixed step | 32.1% |
| Stop at a fixed token budget | 32.1% |
| Confidence threshold | 30.2% |
| Entropy threshold | 26.5% |
| **Difficulty-aware DQN** | **40.1%** |

**+8.0 accuracy points over a fixed step at the same cost** (95% CI +4.2 to +11.8,
p = 0.0001). Confidence and entropy thresholds are what the early-exit literature
actually uses, and the DQN beats both by more (+9.9 and +13.5). This is the comparison
the contribution rests on.

### The operating point is a choice, and it has to be stated

How the best checkpoint is selected on validation decides where on the frontier the
policy sits, and the two objectives land somewhere quite different:

| Selection | Tokens saved | Accuracy | vs fixed step | vs behaviour cloning |
|---|---|---|---|---|
| `accuracy_at_budget` (reported) | 48% | **40.1%** | +8.0, p=0.0001 | +1.3, p=0.47 |
| `weighted_score` (aggressive) | 65% | 33.7% | +4.5, p=0.027 | -5.3, p=0.006 |

The reported point states a budget. The aggressive one uses `accuracy + 0.30 x
token_reduction`, where the 0.30 silently prices a token against accuracy: at that
weight the policy stops at **step 0 on 54% of questions**, where accuracy is 21%. Both
are honest points on one frontier, but a stated budget is defensible in a way an
arbitrary weight is not, so the reported figures use the former.

## 4. Reinforcement learning matches supervised imitation, but does not beat it

| Policy | Accuracy | Tokens saved |
|---|---|---|
| Behaviour cloning (supervised) | 38.7% | **52.8%** |
| DQN (reinforcement learning) | **40.1%** | 47.8% |

The DQN is now 1.3 points ahead on accuracy, but the gap is not significant
(p = 0.47) and behaviour cloning still spends **4.9 points of budget less**. Calling
this a win for reinforcement learning would overstate it: the fair summary is that the
two are level, with behaviour cloning slightly cheaper. Reporting the accuracy tie alone would flatter the DQN, which is
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
| answer length, word count, is-a-value, repeat count | **added later - see below** |

### The fix: what the answer *looks like*

Acting on that, four features describing the **shape of the current answer** were
added - its length, word count, whether it is a bare value rather than a sentence, and
how many earlier steps held the same string. A model part-way through emits "the net
change in repurchase reserves between 2008 and..." and a finished one emits "407
million dollars", so the form of the string says more about whether it is done than
confidence or entropy do.

| State | AUC for "would stopping here be correct" |
|---|---|
| difficulty only (3 features) | 0.654 |
| the original 14 features | 0.700 |
| **plus answer shape (18 features)** | **0.767** |

That is the largest single improvement found in the project, and it lifted the DQN
from 39.2% to 40.1% while widening its margin over a fixed rule from +7.2 to +8.0
points. All four are computable online from what the model has already produced.

The original diagnosis still stands for the features available *before* that change:
almost all the signal was *question-level* (how hard is this question) rather than
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
| Easy | 427 tokens | 234 | **55%** | 84.4% |
| Medium | 540 tokens | 274 | **51%** | 57.6% |
| Hard | 585 tokens | 303 | **52%** | 14.7% |

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
