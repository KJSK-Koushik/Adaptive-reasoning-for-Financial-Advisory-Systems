# Difficulty-Aware Adaptive Reasoning for Financial Advisory Systems

[![CI](https://github.com/KJSK-Koushik/Adaptive-reasoning-for-Financial-Advisory-Systems/actions/workflows/ci.yml/badge.svg)](https://github.com/KJSK-Koushik/Adaptive-reasoning-for-Financial-Advisory-Systems/actions/workflows/ci.yml)

A Deep Q-Network (DQN) that learns **when an LLM should stop reasoning**, instead of using
fixed stopping rules. Applied to financial question answering (investment analysis, fraud
detection, credit risk).

## The idea in one paragraph

Reasoning LLMs think step by step before answering. They frequently reach the correct answer
early and then keep going — burning tokens, latency, memory and energy for no accuracy gain.
This project trains a small RL agent that watches the reasoning stream (confidence, entropy,
token count, answer stability, and the predicted difficulty of the question) and decides at
each step: **CONTINUE** or **STOP AND ANSWER**. The reward is difficulty-aware — wasting
tokens on an easy question is penalised far more heavily than on a hard one.

## Repository layout

```
configs/                  YAML configuration (single source of truth for every phase)
data/
  raw/                    Downloaded datasets, untouched
  interim/                Per-source cleaned output
  processed/              Unified schema, splits, difficulty labels
artifacts/
  traces/                 Generated reasoning traces + step features
  models/                 Trained difficulty classifier and DQN policy
  results/                Evaluation tables, plots, ablations
  logs/                   Run logs
scripts/                  Entry points, one per phase
src/adaptive_reasoning/
  config.py               Typed config loader
  paths.py                Canonical project paths
  schema.py               Unified data record + trace step schemas
  device.py               CPU / CUDA / XPU / MPS detection
  data/                   Phase 1 - dataset adapters, templating, synthetic generation
  difficulty/             Phase 2 - difficulty labelling + classifier
  traces/                 Phase 3 - reasoning trace generation and early-exit probing
  rl/                     Phases 4-5 - offline environment, replay buffer, DQN
  eval/                   Phase 6 - baselines, metrics, energy measurement
  serve/                  Phase 7 - real-time adaptive reasoning controller
  app/                    Phase 8 - FastAPI service and Streamlit dashboard
tests/                    pytest suite
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffolding, config, environment | **complete** |
| 1 | Data pipeline and unified schema | **complete** — 30,660 rows, all 7 sources |
| 2a | Difficulty labelling and classifier (CPU) | **complete** — all 3 stages verified |
| 2b | k-sample run on GPU | **complete** - bundled with Phase 3 |
| 3a | Trace generator + probing (code) | **complete** — validated on CPU |
| 3b | Trace generation run (GPU) | **complete** - 4,000 traces at a 768-token budget |
| 4 | Offline RL environment | **complete** — 91,706 transitions |
| 5 | DQN training | **complete** — see docs/FINDINGS.md |
| 6 | Baselines and evaluation | **complete** - see artifacts/results/phase6_summary.json |
| 7 | Real-time inference controller | **complete** - verified identical to the offline evaluation |
| 8 | Financial advisory application | **complete** - FastAPI service and Streamlit dashboard |
| 9 | Ablations and final results | **complete** - see docs/FINDINGS.md section 7 |

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Verify the environment and see what hardware is available:

```powershell
python scripts/check_env.py
```

Create the data and artifact directory tree:

```powershell
python scripts/bootstrap.py
```

Build the unified dataset (downloads Tier 1 sources automatically, ~110 MB):

```powershell
python scripts/run_phase1.py
```

## Results

Measured on 599 held-out test questions, every policy replayed over the same recorded
traces (`python scripts/run_phase6.py --experiment reported` regenerates the table).

| Policy | Accuracy | Mean tokens | Tokens saved |
|---|---|---|---|
| Full reasoning — no early stop | 44.7% | 538 | — |
| Fixed step — the standard approach | 32.0% | 284 | 47% |
| Confidence threshold — prior work | 30.2% | 272 | 50% |
| Entropy threshold — prior work | 26.5% | 301 | 44% |
| Behaviour cloning — supervised control | 38.7% | 254 | 53% |
| **Double DQN — ours** | **40.1%** | **281** | **48%** |
| Oracle — upper bound, uses hindsight | 65.3% | 137 | 75% |

**+8.0 accuracy points over a fixed stopping rule at the same token cost** (paired
bootstrap, p < 0.0001), and a match for supervised control (+1.3, not significant).

The ablation (Phase 9) contradicts part of the original hypothesis: removing difficulty
from the state *improves* the policy to 44.1% (+12.0 over the fixed rule), and giving it
the true difficulty label does not help either. The gain comes from reading the model's
own confidence and answer stability, not from question difficulty. Full discussion in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Running the pipeline

Phases 0-2 and 4-9 run on a CPU. Phase 3 needs a GPU; see `docs/PHASE3_REMOTE.md` for
the Kaggle notebook cell, and `docs/KAGGLE_SETUP.md` for the account setup.

```powershell
python scripts/run_phase1.py                       # unified dataset
python scripts/run_phase2.py                       # difficulty labels and classifier
python scripts/run_pilot.py                        # gate: is the model reasoning at length? (GPU)
python scripts/run_phase3.py                       # reasoning traces with step-wise probing (GPU)
python scripts/run_phase4.py --experiment reported # offline RL transitions
python scripts/run_phase5.py --experiment reported # Double DQN + behaviour-cloning control
python scripts/run_phase6.py --experiment reported # baselines at matched cost, significance
python scripts/run_phase7.py --experiment reported # live controller, checked against Phase 6
python scripts/run_phase8.py --check               # API and dashboard smoke test
python scripts/run_phase9.py --experiment reported # ablations
python scripts/build_deliverables.py               # report and deck from the results files
```

`configs/experiment/reported.yaml` pins the checkpoint-selection objective and the
operating point every reported figure uses; running without `--experiment reported`
uses the defaults in `configs/default.yaml`.

Tests and lint, as run in CI on Python 3.11 and 3.12:

```powershell
pytest
ruff check .
```

The dashboard replays recorded traces through the Phase 7 controller, so it needs no GPU:

```powershell
streamlit run src/adaptive_reasoning/app/dashboard.py
```

## Datasets

See `docs/DATASETS.md` for sources, licences and download instructions.
Tier 1 sources download automatically in Phase 1. PaySim and ConvFinQA are manual.

## Scope note

The system produces *analysis* — computations, risk scores, fraud likelihood, and factual
question answering over financial documents. It does not produce personalised investment
recommendations, and every response carries a disclaimer. This is also a research
requirement: answers must be automatically gradeable for the RL reward to exist at all.
