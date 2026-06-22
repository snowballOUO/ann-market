"""
Shadow recall must be an unbiased estimate of approximate-search recall.

We construct a tiny synthetic dataset where we know ground-truth exactly,
run a "fake approximate" search that returns a random subset, and check
that ShadowSampler's recall matches what we compute manually.
"""
import numpy as np
import pytest
from src.agents.shadow_sampler import ShadowSampler
from src.system.types import Query


def test_shadow_recall_matches_manual():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 32)).astype(np.float32)
    recalls = {}

    def cb(qid, r):
        recalls[qid] = r

    ss = ShadowSampler(base, sample_rate=1.0, max_workers=1, on_recall_computed=cb, seed=0)

    # Query vector
    v = rng.standard_normal(32).astype(np.float32)
    k = 10

    # Manual exact KNN
    diffs = base - v
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    exact_top10 = set(np.argpartition(d2, k)[:k].tolist())

    # Fake "approximate" results: take 7 from exact, 3 random non-exact
    exact_list = list(exact_top10)
    non_exact = [i for i in range(500) if i not in exact_top10][:50]
    approx = [(i, 0.0) for i in exact_list[:7]] + [(i, 0.0) for i in non_exact[:3]]
    # Expected recall = 7 / 10
    expected = 0.7

    q = Query(id="q_test", v_t=v, k_t=k, filter_t={}, sla_t=0.05, budget_t=0.01)
    sampled = ss.maybe_sample(q, approx)
    assert sampled, "sample_rate=1.0 should always sample"

    ss.drain(timeout=10)
    ss.shutdown()

    assert "q_test" in recalls
    assert abs(recalls["q_test"] - expected) < 1e-6, (
        f"computed recall {recalls['q_test']} != expected {expected}"
    )


def test_shadow_sample_rate_respected():
    """With sample_rate=0.1, roughly 10% of queries should be sampled."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 16)).astype(np.float32)
    ss = ShadowSampler(base, sample_rate=0.1, max_workers=1, on_recall_computed=None, seed=42)

    sampled_count = 0
    for i in range(1000):
        v = rng.standard_normal(16).astype(np.float32)
        q = Query(id=f"q_{i}", v_t=v, k_t=5, filter_t={}, sla_t=0.05, budget_t=0.01)
        approx = [(j, 0.0) for j in range(5)]
        if ss.maybe_sample(q, approx):
            sampled_count += 1

    # Expect ~100, allow generous slack
    assert 70 <= sampled_count <= 130, f"got {sampled_count} samples, expected ~100"
    ss.drain(timeout=30)
    ss.shutdown()
