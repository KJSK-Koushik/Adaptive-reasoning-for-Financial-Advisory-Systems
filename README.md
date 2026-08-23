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

## Datasets

See `docs/DATASETS.md` for sources, licences and download instructions.
Tier 1 sources download automatically in Phase 1. PaySim and ConvFinQA are manual.

## Scope note

The system produces *analysis* — computations, risk scores, fraud likelihood, and factual
question answering over financial documents. It does not produce personalised investment
recommendations, and every response carries a disclaimer. This is also a research
requirement: answers must be automatically gradeable for the RL reward to exist at all.
