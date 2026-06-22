"""
Smoke test for the per-query latency contributions of non-FAISS agents.

DifficultyEstimator + PolicyAgent together must stay well under 2ms p99 even
on a developer laptop. If this regresses, the whole "low-latency agent" story
breaks.

Run with: pytest tests/test_latency.py -s
"""
import time
import numpy as np
from src.system.types import Query
from src.agents.difficulty_estimator import DifficultyEstimator
from src.agents.policy_agent import FixedPolicy


def make_query(qid=0):
    return Query(
        id=f"q_{qid}",
        v_t=np.random.randn(128).astype(np.float32),
        k_t=10,
        filter_t={},
        sla_t=0.050,
        budget_t=0.010,
    )


def test_difficulty_estimator_fast():
    sample = np.random.randn(5000, 128).astype(np.float32)
    est = DifficultyEstimator(sample_vectors=sample)
    # Warm up
    for _ in range(5):
        est.estimate(make_query())

    times = []
    for i in range(200):
        q = make_query(i)
        t0 = time.perf_counter()
        est.estimate(q)
        times.append(time.perf_counter() - t0)

    p99 = np.percentile(times, 99) * 1000  # ms
    mean = np.mean(times) * 1000
    print(f"\nDifficultyEstimator: mean={mean:.3f}ms p99={p99:.3f}ms")
    assert p99 < 5.0, f"DifficultyEstimator p99={p99:.3f}ms exceeds budget"


def test_policy_agent_fast():
    configs = [
        {"nprobe": 8, "rerank_k": 50, "early_stop": False},
        {"nprobe": 32, "rerank_k": 200, "early_stop": False},
        {"nprobe": 128, "rerank_k": 800, "early_stop": False},
    ]
    prices = [0.001, 0.005, 0.01, 0.02]
    pol = FixedPolicy(configs, prices, default_z_index=1, default_p_index=1, epsilon=0.1, seed=0)

    times = []
    for i in range(1000):
        q = make_query(i)
        t0 = time.perf_counter()
        pol.decide(q, U_t=0.5, h_t={})
        times.append(time.perf_counter() - t0)

    p99 = np.percentile(times, 99) * 1000
    mean = np.mean(times) * 1000
    print(f"\nFixedPolicy: mean={mean:.3f}ms p99={p99:.3f}ms")
    assert p99 < 2.0, f"FixedPolicy p99={p99:.3f}ms exceeds budget"
