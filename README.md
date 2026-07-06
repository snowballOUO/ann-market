# ANN Marketplace — Adaptive Vector Retrieval & Pricing System

Per-query adaptive retrieval configuration and pricing for vector database services. Jointly optimizes nprobe (search depth) and price under heterogeneous buyer constraints, with offline safe learning via Doubly Robust estimation and online contextual bandits.

---

## Architecture Overview

```
Query arrives
    |
    v
DifficultyEstimator (MLP ONNX, p99 <0.5ms)
    |
    v
ContextCache (100-query sliding window)
    |
    v
PolicyAgent ──┬── LinUCB (6-dim, online bandit)
              ├── Q-Net (7-dim, offline distilled)
              ├── HeteroLinUCB (10-dim, persona-encoded)
              └── FeasibleFixedPolicy (constrained baseline)
    |
    v
Feasible Action Filter (cost <= budget, price >= cost)
    |
    v
ExecutionAgent (FAISS IVF-PQ, 5 nprobe levels)
    |
    v
ShadowSampler (2% async exact recall)
    |
    v
BuyerSimulator (6 switchable versions)
    |
    v
Reward = satisfaction x (price - cost)
```

---

## Two Experiment Pipelines

### Pipeline A: Original (Homogeneous Market)

- **Entry**: `scripts/run_main_experiment.py --dataset <name>`
- **Buyer**: `original` — GT real recall + 3-type Utility + Market Sentiment
- **Strategies**: FixedPolicy, SLA Heuristic, Cost-Based, LinUCB, Naive DQN (no U_t), Naive DR (IPS), Q-Net (ours)
- **Features**: GT recall computation, Shadow quality adjustment, DR offline validation, Q-Net distillation
- **Goal**: Prove adaptive > fixed in standard random-workload settings

### Pipeline B: V2 Heterogeneous (Persona-Encoded Market)

- **Entry**: `scripts/run_hetero_experiment.py` / `scripts/run_scenario_experiment.py`
- **Buyer**: `hetero` (persona-locked + continuous satisfaction) / `must_serve` (never reject)
- **Strategies**: FeasibleFixedPolicy, HeteroLinUCB (10-dim with persona one-hot)
- **Workloads**: PersonaWorkload (Enterprise/Budget/Quality), ScenarioWorkload (S1-S6: mindset x difficulty)
- **Goal**: Prove adaptive >> fixed when buyer heterogeneity is explicitly modeled

---

## Quick Start

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Data (SIFT1M, ~600MB)
bash scripts/download_sift1m.sh data/sift1m
python scripts/build_index.py --config configs/sift1m.yaml
python scripts/pretrain_difficulty.py --config configs/sift1m.yaml

# Run Pipeline A (7 strategies, 5 seeds)
python scripts/run_main_experiment.py --dataset sift1m

# Run Pipeline B (HeteroLinUCB vs Fixed)
python scripts/switch_buyer_version.py hetero
python scripts/run_hetero_experiment.py --config configs/sift1m_hetero.yaml \
    --n-queries 10000 --seeds 42,123,456

# Scenario experiment (S1-S6, must-serve, multi-process)
python scripts/switch_buyer_version.py must_serve
python scripts/run_scenario_experiment.py --config configs/sift1m_scenario.yaml \
    --scenarios all --seeds 42,123,456 --n-queries 5000
```

---

## Buyer Versions

Switch via `python scripts/switch_buyer_version.py <name>` or `BUYER_VERSION=<name>` env var:

| Version | Behavior | Accept | Use Case |
|---------|----------|:------:|------|
| `baseline` | Original bug: ignores query.sla/budget | ~98% | Reproduce legacy results |
| `fixed` | Hard reject if price > budget or latency > sla | Varies | Constraint validation |
| `soft` | budget/sla as utility anchors (recommended) | Varies | General experiments |
| `hetero` | Persona-locked + continuous satisfaction | Varies | Heterogeneous market |
| `must_serve` | Never reject (SLA-guaranteed cloud service) | 100% | Scenario S1-S6 |
| `original` | GT real recall + 3-type buyer + sentiment | Varies | Pipeline A default |

---

## Key Experiment Results (SIFT1M)

### Pipeline A: 7-strategy comparison (5 seeds)

| Method | Revenue | vs Fixed | Accept |
|--------|---------|:---:|------|
| SLA Heuristic | $45.99 | -3.2% | 99.3% |
| Cost-Based | $47.45 | -0.2% | 99.4% |
| FixedPolicy | $47.53 | — | 97.7% |
| LinUCB (online) | $49.38 | +3.9% | 83.7% |
| Naive DR (IPS) | $51.95 | +9.3% | 84.2% |
| Naive DQN (no U_t) | $52.04 | +9.5% | 84.0% |
| **Q-Net (ours)** | **$52.69** | **+10.9%** | 83.8% |

### Pipeline B: HeteroLinUCB vs Fixed (3 seeds)

| Experiment | Fixed | HeteroLinUCB | Gain |
|-----------|-------|-------------|:---:|
| Hetero market | $26.70 | $29.89 | +12.0% |
| S1-S6 scenarios (avg) | $24.30 | $36.10 | +49% |

---

## Key Files

```
src/
  agents/         — Policy agents (LinUCB, Q-Net, HeteroLinUCB, FeasibleFixed, Naive)
  causal/         — DR estimator, LightGBM reward model
  data/           — Dataset loaders, buyer versions, workload generators
  models/         — Q-Net architecture + distillation
  pricing/        — Feasible actions, state features, reward shaping
  system/         — Orchestrator, types, context cache, log writer
scripts/
  run_main_experiment.py    — Pipeline A: 7-strategy comparison
  run_hetero_experiment.py  — Pipeline B: hetero market
  run_scenario_experiment.py— Pipeline B: S1-S6 scenarios (multi-process)
  compare_bandit.py         — Standalone policy comparison
  train_qnet.py             — Q-Net offline training + distillation
  validate_dr.py            — DR offline validation
  build_index.py            — FAISS IVF-PQ index construction
  pretrain_difficulty.py    — MLP difficulty estimator training
  switch_buyer_version.py   — Buyer version switching
configs/
  sift1m.yaml               — Pipeline A config (fvecs format)
  sift1m_hetero.yaml        — Pipeline B hetero config
  sift1m_scenario.yaml      — Pipeline B scenario config
  deep1m.yaml / ag_news.yaml / gist1m.yaml
```

---

## Environment

```
Python 3.11+
faiss-cpu, torch, numpy, pandas, pyarrow, scikit-learn, lightgbm
onnx, onnxruntime, matplotlib, tqdm, h5py, pytest
```

## Datasets

Four standard ANN-benchmarks datasets (download separately):
- **SIFT1M** (128-dim, 1M vectors, image descriptors)
- **DEEP1M** (96-dim, 1M vectors, angular distance)
- **GIST1M** (960-dim, 1M vectors, scene descriptors)
- **AG_NEWS** (384-dim, 120K vectors, text embeddings)

---

*Branch: v2-integrated | Builds on clean-final with V2 heterogeneous marketplace extensions*
