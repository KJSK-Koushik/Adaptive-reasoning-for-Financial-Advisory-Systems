# Phase 2 — Difficulty labelling and classification

## Why difficulty has to be measured, not assumed

No financial QA dataset ships with difficulty labels, and hand-labelling 30,000
questions is not feasible. More importantly, hand labels would be the *wrong* labels.

The stopping policy's job is to predict **how long this model needs on this question**.
A question a finance professional finds hard but the model answers instantly should be
labelled easy, because the correct action is to stop early. So we measure
**model-perceived difficulty** by sampling the model `k` times and observing how it
copes.

## Two notions of difficulty, kept apart

| Field | Meaning | Set by |
|---|---|---|
| `difficulty_prior` | *Intrinsic* difficulty — how many arithmetic steps the question needs | Phase 1, synthetic generator only |
| `difficulty` | *Model-perceived* difficulty — measured from `k` sampled attempts | Phase 2 |

These are deliberately separate columns. Conflating them would train the classifier on
the wrong notion. Keeping both means Phase 9 can report how far apart they are, which
is itself a result: if they diverged sharply, that is direct evidence that
difficulty-aware stopping needs *measured* difficulty rather than a hand-written rule.

## How difficulty is actually measured — changed after the pilot

The original design sampled the model `k` times per question purely to measure how hard
each one was. **The Kaggle pilot killed that plan on cost**: measured at 129 tokens/sec
on a T4, that job alone was ~20M of 29M total generated tokens — **71% of the GPU
budget** — which would have put the run at ~60 hours against a 30-hour weekly quota.

It was also redundant. The Phase 3 traces already probe the model at every step
boundary and record whether it was right there and whether its answer moved. **That is
the same evidence, already paid for.**

So difficulty is now derived from traces
(`src/adaptive_reasoning/difficulty/from_traces.py`) on CPU, at zero GPU cost. The
primary signal is `correct_ratio`: the fraction of probe points at which the answer was
already correct.

| Trace shape | correct_ratio | Tier |
|---|---|---|
| Right from step 2 of 16, held | ~0.88 | easy |
| Right from step 8 of 16 | ~0.50 | medium |
| Right only at the final step | ~0.06 | hard |
| Never right | 0.00 | hard |

This is arguably a **better** measure than a k-sample pass rate for this purpose,
because it answers the question the stopping policy actually faces — *how early, and
how consistently, does the model know this?* — rather than the different question of
how often the model gets it right across independent attempts.

The k-sampling route still exists and is still tested (`--source samples`); it is
simply not the default. Worth reporting as a methodological choice with its measured
justification.

## The labelling rules

Both routes use the same three signals, so the methodology story stays consistent. For
k-sampling (`src/adaptive_reasoning/difficulty/labeling.py`):

**1. Pass rate** over `k` samples — the primary signal.
- `>= easy_min_pass_rate` (0.8) → **easy**
- `<= hard_max_pass_rate` (0.2) → **hard**
- otherwise → **medium**

**2. Reasoning-length bump.** A question answered correctly every time but only after
600 tokens of deliberation is not easy. If median reasoning length exceeds
`long_reasoning_token_threshold`, the tier is bumped one harder.

Length is measured on the **correct attempts only**. A wrong answer that rambled for
5,000 tokens says nothing about how long the question genuinely needs, and including it
would systematically mislabel questions the model fails at.

**3. Instability demotion.** If every one of the `k` samples produced a different
answer, the model is guessing rather than reasoning, and a medium label is demoted to
hard even if one sample happened to land correctly.

The module has no LLM dependency at all — it consumes sample *results*. That means the
rules can be re-tuned and re-tested in milliseconds without touching a GPU, which
matters because these thresholds will need adjusting once real sampling data exists.

## The classifier

At training time we can afford `k` samples per question. At inference time we cannot —
that would cost more than the reasoning we are trying to save. So a small classifier
learns to predict the measured label from text alone.

**Architecture:** frozen MiniLM sentence embeddings (384-d) + 10 interpretable surface
features → LightGBM. Roughly 1 ms per query on CPU, negligible against the reasoning it
gates.

**It deliberately does not use `source` or `domain` as features.** Both exist in the
training data and both would inflate the accuracy number. But a live query typed into
the advisory app has neither, and a model leaning on *"this came from PhraseBank,
therefore easy"* would collapse on contact with a real user. Every feature is
derivable from the query itself.

**The majority baseline is always reported alongside accuracy.** A classifier that
cannot beat "always predict the commonest tier" contributes nothing, and at three
imbalanced classes that is an easy trap to miss.

## Running it

The default route needs no GPU at all — run it after Phase 3 traces are downloaded:

```bash
# Derive labels from the traces (CPU, seconds)
python scripts/run_phase2.py --stage label

# Train the classifier (CPU, seconds)
python scripts/run_phase2.py --stage train
```

`--stage label` picks the source automatically: traces if they exist, otherwise
k-samples. Force it with `--source traces` or `--source samples`.

The GPU-expensive route remains available if you ever want the independent-sample
comparison:

```bash
python scripts/run_phase2.py --stage sample          # GPU, hours
python scripts/run_phase2.py --stage label --source samples
```

Because labelling is now a seconds-long CPU job, the thresholds in
`difficulty.from_traces` can be re-tuned and the labels regenerated freely. The label
stage prints a warning if more than 85% of questions land in one tier, which is the
signal that they need adjusting.

Sampling checkpoints every `traces.checkpoint_every` questions into
`data/processed/_difficulty_shards/` and skips already-completed question ids on
restart, so a killed Kaggle session costs minutes rather than hours.

## Local smoke test

```bash
python scripts/run_phase2.py --stage sample --experiment smoke --limit 12
```

Runs the identical code path on `Qwen/Qwen2.5-0.5B-Instruct` on CPU. It proves the
pipeline — batching, extraction, grading, checkpointing, consolidation — but **not** the
science: a 0.5B instruction model does not reason. See below.

## Finding: the smoke model does not reason

The first smoke run produced a median of **3 reasoning tokens** per generation. The
model ignored the requested format entirely and emitted a bare number. Every numeric
question was wrong; PaySim scored 6/6 purely because yes/no is guessable.

Strengthening the system prompt made no difference (5 tokens vs 6). This is a capability
limit of the 0.5B model, not a prompting fault.

**Why this matters:** if the model does not reason, there is nothing to stop early and
the project has no signal. The smoke configuration is therefore only ever a *plumbing*
test.

This finding is now enforced in code. `scripts/run_pilot.py` measures reasoning length
and refuses to let Phase 3 proceed below a threshold — see `docs/PHASE3_REMOTE.md`. Run
it on the GPU before anything else.

The reasoning behaviour of `DeepSeek-R1-Distill-Qwen-1.5B` (which emits `<think>` blocks
natively) has **not** yet been confirmed empirically; the local attempt was abandoned
when the 3.5 GB download stalled on a 481 KB/s link. The pilot gate exists so that this
gets checked automatically rather than assumed.
