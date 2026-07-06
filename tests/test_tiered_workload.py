"""Tests for tiered workload generation."""
import numpy as np
import pytest

from src.data.tiered_workload import (
    TieredWorkloadBuilder,
    WorkloadTier,
    estimate_fair_price,
    summarize_workload_with_queries,
)


@pytest.fixture
def builder():
    cost_model = {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}
    tiers = [0.001, 0.002, 0.005, 0.01, 0.02]
    return TieredWorkloadBuilder(cost_model, tiers, margin=0.5)


def test_tier1_budget_very_low_independent_of_fair(builder):
    rng = np.random.default_rng(0)
    v = np.random.randn(128).astype(np.float32)
    budgets = []
    fairs = []
    for i in range(200):
        q, meta = builder.build_query(WorkloadTier.TIER1_RANDOM_LOW, i, v, U_t=0.9, rng=rng)
        budgets.append(q.budget_t)
        fairs.append(meta.fair_price)
    assert max(budgets) <= 0.002
    assert np.mean(fairs) > np.mean(budgets) * 2


def test_tier2_budget_tracks_fair_price(builder):
    rng = np.random.default_rng(1)
    v = np.random.randn(128).astype(np.float32)
    ratios = []
    for i in range(300):
        U_t = rng.uniform(0, 1)
        q, meta = builder.build_query(WorkloadTier.TIER2_COST_MICRO, i, v, U_t, rng=rng)
        ratios.append(q.budget_t / meta.fair_price)
    ratios = np.array(ratios)
    assert ratios.mean() > 0.85
    assert ratios.mean() < 1.15
    assert ratios.std() < 0.2


def test_tier3_polarized_easy_hard(builder):
    rng = np.random.default_rng(2)
    v = np.random.randn(128).astype(np.float32)
    easy_budgets = []
    hard_budgets = []
    for i in range(400):
        U_t = 0.1 if i % 2 == 0 else 0.9
        q, meta = builder.build_query(WorkloadTier.TIER3_POLARIZED, i, v, U_t, rng=rng)
        if meta.bucket == "easy":
            easy_budgets.append(q.budget_t)
        else:
            hard_budgets.append(q.budget_t)
    assert np.mean(easy_budgets) < 0.005
    assert np.mean(hard_budgets) > 0.005
    assert np.std(easy_budgets + hard_budgets) > 0.003


def test_generate_sequence_reproducible(builder):
    xq = np.random.randn(50, 128).astype(np.float32)

    def diff_fn(v, k):
        return float(np.linalg.norm(v) % 1.0)

    q1, m1 = builder.generate_sequence(1, xq, 100, seed=42, difficulty_fn=diff_fn)
    q2, m2 = builder.generate_sequence(1, xq, 100, seed=42, difficulty_fn=diff_fn)
    assert [q.budget_t for q in q1] == [q.budget_t for q in q2]
    assert [m.fair_price for m in m1] == [m.fair_price for m in m2]


def test_summarize_workload_stats(builder):
    rng = np.random.default_rng(3)
    v = np.zeros(128, dtype=np.float32)
    queries, metas = [], []
    for i in range(50):
        q, m = builder.build_query(WorkloadTier.TIER2_COST_MICRO, i, v, 0.5, rng)
        queries.append(q)
        metas.append(m)
    stats = summarize_workload_with_queries(queries, metas)
    assert stats["n"] == 50
    assert 0.8 < stats["budget_fair_ratio_mean"] < 1.2


def test_estimate_fair_price_monotone_in_u():
    cost_model = {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}
    _, p_easy = estimate_fair_price(0.1, 10, cost_model)
    _, p_hard = estimate_fair_price(0.9, 100, cost_model)
    assert p_hard > p_easy
