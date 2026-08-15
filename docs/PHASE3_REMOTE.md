# Phase 3 on a remote GPU

## Why this document exists

The development machine is an Intel i7-1360P with **Intel Iris Xe integrated graphics**
(no CUDA), 15.7 GB RAM. Phases 0–2 and 4–8 run comfortably here. Phase 3 — generating
reasoning traces from a 1.5B reasoning model — does not.

Rough CPU estimate: a 1.5B model on this laptop generates on the order of 5–12 tokens/sec.
Phase 3 needs roughly 10 million generated tokens. That is **weeks of wall-clock time**,
which is not a viable plan. Phase 3 therefore runs on a free remote GPU.

## Recommended: Kaggle Notebooks

| | Kaggle | Colab (free) |
|---|---|---|
| GPU | P100 16GB or T4 ×2 | T4 16GB |
| Quota | **30 GPU-hours/week, guaranteed** | Variable, throttled with use |
| Session length | 12 h | ~4–12 h, disconnects more often |
| Persistent storage | Datasets + notebook output | Google Drive mount |
| Account needed | Yes — **you already need one for PaySim** | Google account |

Kaggle wins on the guaranteed quota, and you're signing up anyway. Colab is the fallback.

## The cost-saving design decisions

**1. Reuse the KV cache for probes.** A naive implementation re-runs the prompt plus all
reasoning so far for every probe — quadratic cost. Instead we keep the KV cache, append
the probe prompt, decode ~24 tokens, then *discard the probe tokens and continue the main
reasoning from the cached state*. Probing becomes nearly free. This is what makes 32 probes
per question affordable.

**2. Fold difficulty labelling into the same GPU job.** Phase 2 needs `k` samples per
question to derive a difficulty label; Phase 3 needs traces. Both are LLM generation.
Running them as two separate remote jobs doubles the GPU bill for no reason, so
`scripts/run_phase3.py` will do both in one pass and emit both artifacts.

**3. Drop `k` from 5 to 3.** Five samples per question costs ~14 GPU-hours on its own.
Three gives a pass rate of 0/3, 1/3, 2/3, 3/3 — still enough to separate the three tiers,
at 40% less compute. This is a config change (`difficulty.k_samples`), and the trade-off
should be noted in the report's limitations section.

**4. Checkpoint every 100 questions.** Sessions get killed. Partial parquet shards in
`artifacts/traces/` mean a dropped session costs minutes, not hours.

## Estimated budget

These are back-of-envelope figures at ~600 tokens/sec aggregate throughput (1.5B, fp16,
batch 8, T4). **The first thing Phase 3 does is a 50-question pilot to replace these
guesses with measured numbers.**

| Job | Tokens | Estimate |
|---|---|---|
| Difficulty sampling, k=3, 512 tok cap, 8k questions | ~12M | ~5.5 h |
| Trace generation, 6k questions, ~1.8k tok each | ~11M | ~5 h |
| Pilot + retries + overhead | — | ~2 h |
| **Total** | | **~12–13 h** |

That fits inside a single week of Kaggle's free quota, with room to redo it once.

## Run the pilot gate FIRST

```bash
python scripts/run_pilot.py
```

Two minutes on a GPU. It exits non-zero and tells Phase 3 not to run if either
load-bearing assumption fails:

1. **The model must actually reason at length.** If it answers in three tokens there is
   nothing to stop early, no confidence trajectory to observe, and no signal for the
   DQN. This is not hypothetical — the Phase 2 smoke run on `Qwen2.5-0.5B-Instruct`
   produced a **median of 3 reasoning tokens**, and strengthening the system prompt
   changed nothing (5 tokens vs 6). Running Phase 3 on a model like that would have
   burned twelve GPU-hours producing traces with nothing to learn from.
2. **The throughput estimates must be real.** Every figure in the table below is a
   guess until the pilot measures it. The pilot recomputes the projected GPU-hours from
   observed tokens/second and prints the number.

Verified failing behaviour, run locally against the 0.5B model:

```
  median reasoning tokens  3
  <think> tag rate         0%
  GATE FAILED - do not run Phase 3
```

> **Status: the reasoning behaviour of `DeepSeek-R1-Distill-Qwen-1.5B` has not yet been
> confirmed empirically.** An attempt to verify it on the development laptop was
> abandoned: the download stalled twice at ~1.4 GB of 3.5 GB on a link measured at
> 481 KB/s. This is the first thing to run on Kaggle, where the model downloads in
> under a minute. Do not skip it.

## Workflow

0. **Run `scripts/run_pilot.py` and confirm the gate passes.**
1. Phase 1 and 2 (data prep, no GPU) run locally and produce `unified.parquet`.
2. Upload `unified.parquet` to Kaggle as a private Dataset (~30 MB).
3. Run `notebooks/phase3_kaggle.ipynb` — it pip-installs `requirements-gpu.txt`, clones the
   repo, and calls the same `src/adaptive_reasoning/traces/` code that runs locally.
   **No forked logic**: the notebook is a thin wrapper so results stay reproducible.
4. Download `traces.parquet` (~200–400 MB) back into `artifacts/traces/`.
5. Phases 4–9 run locally on CPU. DQN training on 14 features is a matter of minutes.

## Local fallback

`scripts/run_phase3.py --smoke --n 20` runs the full pipeline on CPU with a tiny model, so
the code can be developed and debugged locally before burning remote GPU time. This is how
Phase 3 gets built: correctness locally, scale remotely.

## Other options

- **Rented GPU** (vast.ai, RunPod): an RTX 3090 runs ~$0.20–0.35/hour, so the whole of
  Phase 3 costs a few dollars and finishes in one uninterrupted session. Worth it if the
  free tiers become a hassle.
- **Intel OpenVINO on the Iris Xe**: viable for the *live demo* in Phases 7–8 (int4, one
  query at a time), not for bulk generation. Revisit at Phase 7.
- **Shrink the study**: 1,500–2,000 questions instead of 6,000 makes CPU generation
  possible over a few nights. Weaker statistics, but the project still stands.
