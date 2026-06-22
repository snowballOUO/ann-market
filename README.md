# ANN Marketplace — Week 1 prototype

Causal Bellman ANN serving marketplace (SIGMOD prototype, Week 1 skeleton).

This is the working code for the **Week 1 milestone**: a runnable end-to-end
pipeline with all five agents wired up, using a deterministic fixed policy
and a stub buyer. No learning yet — that comes in Week 3-5. The goal of
Week 1 is to prove the plumbing works and the trajectory log has all the
fields downstream learning will need.

## What's in this repo

```
ann-marketplace/
├── configs/base.yaml           # all knobs in one place
├── src/
│   ├── agents/
│   │   ├── difficulty_estimator.py  # heuristic U_t
│   │   ├── policy_agent.py          # FixedPolicy with eps-exploration
│   │   ├── execution_agent.py       # FAISS wrapper
│   │   ├── shadow_sampler.py        # async exact-recall sampler
│   │   └── learner_agent.py         # stub for now (Week 5 work)
│   ├── data/
│   │   ├── datasets.py              # SIFT1M .fvecs loader
│   │   └── buyer_stub.py            # Week 1 placeholder buyer
│   └── system/
│       ├── types.py                 # Query, Action, Outcome, Trajectory
│       ├── context_cache.py
│       ├── log_writer.py            # parquet append-only log
│       └── orchestrator.py          # ties everything together
├── scripts/
│   ├── download_sift1m.sh
│   ├── build_index.py               # one-time IVF-PQ build
│   └── run_experiment.py            # main entry point
└── tests/
    ├── test_propensity.py           # critical invariant
    ├── test_shadow_unbiased.py      # critical invariant
    ├── test_latency.py              # 2ms budget check
    └── test_end_to_end.py           # full pipeline smoke
```

---

## 1. Environment setup

### 1.1 OS support

Tested on Linux and macOS. On **Windows**, use WSL2 — `faiss-cpu` wheels
for native Windows are unreliable. Inside WSL2 Ubuntu, everything below
just works.

### 1.2 System requirements

* Python 3.11 or newer
* ~2 GB free disk for SIFT1M + index
* 4+ GB RAM (SIFT1M in float32 = ~500 MB, index another ~200 MB)
* No GPU needed for Week 1

### 1.3 Create virtualenv and install

```bash
cd ann-marketplace

# Use whatever Python venv tool you prefer. Plain venv:
python3.11 -m venv .venv
source .venv/bin/activate

# Install in editable mode so 'import src.xxx' works
pip install --upgrade pip
pip install -e ".[dev]"
```

Sanity check:

```bash
python -c "import faiss, torch, numpy, pyarrow; print('ok')"
```

If `faiss-cpu` install fails on macOS, install via conda instead:

```bash
conda install -c pytorch faiss-cpu
```

---

## 2. Get the data

```bash
chmod +x scripts/download_sift1m.sh
bash scripts/download_sift1m.sh data/sift1m
```

This downloads ~500 MB and extracts four files into `data/sift1m/`:

```
sift_base.fvecs        (1M base vectors)
sift_query.fvecs       (10K query vectors)
sift_learn.fvecs       (100K training vectors)
sift_groundtruth.ivecs (10K * 100 ground-truth nearest neighbours)
```

If the FTP mirror is slow, the script falls back to the HTTP mirror. If
both fail, manually download from
http://corpus-texmex.irisa.fr/ and place `sift.tar.gz` inside
`data/sift1m/`, then re-run the script.

Sanity check the load:

```bash
python -c "from src.data.datasets import load_sift1m; \
xb,xq,xt,gt = load_sift1m('data/sift1m'); \
print(xb.shape, xq.shape, xt.shape, gt.shape)"
# expected:
# (1000000, 128) (10000, 128) (100000, 128) (10000, 100)
```

---

## 3. Build the index (one-time, ~1 minute)

```bash
python scripts/build_index.py --config configs/base.yaml
```

Output (abridged):

```
Building IVF-PQ index: nlist=4096, m=16, nbits=8
  training on 100000 vectors...
  training took 25.3s
  adding 1000000 vectors...
  add took 18.1s
Wrote index to data/sift1m/index_ivfpq.faiss (15.8 MB)

Sanity check with nprobe=16, 10 queries:
  recall@10: 0.90
```

If recall@10 is below 0.5, something is wrong with the index — usually a
dim mismatch. Re-check `configs/base.yaml` against the loaded data shapes.

---

## 4. Run the Week 1 experiment

### 4.1 Smoke test (10 queries, ~2 seconds)

```bash
python scripts/run_experiment.py --config configs/base.yaml --n-queries 10
```

Expected tail:

```
Done in 0.X s (XXX qps)
  accepts: 4/10 (40.0%)
  revenue: $0.0XYZ

============================================================
Learner summary
============================================================
  n_trajectories: 10
  n_shadow_sampled: 0 or 1
  mean_recall_when_sampled: 0.X or None
  mean_latency_ms: X.X
  p99_latency_ms: X.X
  accept_rate: 0.4
  total_revenue: 0.0XYZ
  mean_propensity: 0.91
  min_propensity: 0.0X
  policy_versions: ['fixed-v1-eps0.1']
```

### 4.2 Full Week 1 run (1000 queries)

```bash
python scripts/run_experiment.py --config configs/base.yaml --n-queries 1000
```

Expected:
* ~5-15 seconds wall-clock
* ~20 queries shadow-sampled (2% of 1000)
* mean recall when shadow-sampled around 0.85-0.95
* `min_propensity` should be > 0 (this is the killer invariant)

### 4.3 Inspect the trajectory log

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('logs/run_*/run_*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files[-5:]])
print(df.head())
print()
print(df.describe())
"
```

You should see columns `query_id, U_t, z_nprobe, p_t, propensity, L_t, C_t, Q_t, A_t, S_t, R_t`.

---

## 5. Run the tests

```bash
pytest tests/ -v
```

Critical tests:

| Test                              | What it guards against                                  |
|-----------------------------------|---------------------------------------------------------|
| `test_propensity.py`              | Zero or invalid propensity (would break off-policy RL) |
| `test_shadow_unbiased.py`         | Shadow recall computation correctness                  |
| `test_latency.py`                 | Decision-path latency budget regression                |
| `test_end_to_end.py`              | Full pipeline integration                              |

To see latency numbers printed:

```bash
pytest tests/test_latency.py -s
```

To run just one test:

```bash
pytest tests/test_propensity.py::test_fixed_policy_propensity_always_positive -v
```

---

## 6. Debugging cookbook

### 6.1 "No trajectories logged yet"

The `LogWriter` flushes every `flush_every_n` records (default 100). For
small runs (< 100 queries), records sit in the buffer until `close()` is
called. The experiment runner calls `close()` at the end, so this should
not happen during normal use. If you see it after a full run, check that
your run actually finished — Python's KeyboardInterrupt does not call
`close()`.

### 6.2 Recall is suspiciously low

* Check `nprobe` in `configs/base.yaml` — try increasing the default
  `search_param_configs` entries.
* Verify `nlist` is sane for your data size (~4*sqrt(N) is a good rule
  of thumb).
* Run `scripts/build_index.py` and check the sanity-check recall — if
  it's below 0.7, the index itself is bad.

### 6.3 Latency tests fail on slow laptop

The hard latency budgets are for production hardware. On a 2018 MacBook
the policy agent might exceed 2ms p99 for the first few iterations due
to JIT warm-up. The test already does warm-up; if it still fails, bump
the budget temporarily and file a TODO. Don't silently weaken the
production target.

### 6.4 Profile a slow run

```bash
pip install py-spy
py-spy record -o profile.svg -- python scripts/run_experiment.py --n-queries 500
# Open profile.svg in a browser
```

For finer-grained Python-level profiling:

```bash
pip install scalene
scalene scripts/run_experiment.py --cli-only
```

### 6.5 Shadow sampler not catching up

If a long run finishes but `shadow.drain()` times out, increase the timeout
in `run_experiment.py` or bump `max_workers` in `ShadowSampler.__init__`.
The default of 2 workers is conservative to avoid contending with serving.

### 6.6 "FileNotFoundError: index_ivfpq.faiss"

Run `python scripts/build_index.py` first.

### 6.7 Verify propensity invariant manually

```bash
python -c "
import pandas as pd, glob
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('logs/run_*/run_*.parquet'))])
print('zero propensities:', (df.propensity == 0).sum())
print('min propensity:', df.propensity.min())
print('max propensity:', df.propensity.max())
"
```

If `zero propensities > 0`, stop immediately and audit the policy. This
silently corrupts all downstream off-policy learning.

---

## 7. What changes in Week 2 and after

This skeleton is intentionally feature-thin. Upcoming work:

* **Week 2** — replace `buyer_stub.py` with a calibrated `buyer_simulator.py`
  (the most important deliverable, since all experiments depend on buyer
  behaviour fidelity).
* **Week 3** — replace `FixedPolicy` with a `ContextualBandit` policy.
* **Week 5** — add `LearnerAgent.causal_dr_bellman_loss()` and a real
  `QNet` in `src/models/`. Begin off-policy learning loop.
* **Week 7** — full main experiment with baselines.

The interfaces (Query, Action, Trajectory, propensity field) will not
change. Code written against them now is forward-compatible.

---

## 8. Quick reference: full setup from zero

```bash
git clone <repo>  # or just unzip
cd ann-marketplace

python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

bash scripts/download_sift1m.sh data/sift1m
python scripts/build_index.py --config configs/base.yaml

pytest tests/ -v
python scripts/run_experiment.py --config configs/base.yaml --n-queries 1000
```

Total time on a modern laptop: ~10 minutes including download.
