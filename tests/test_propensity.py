"""
Critical invariant: every logged action must carry a non-zero propensity.

If propensity = 0 for an action that was actually taken, off-policy importance
weighting blows up (division by zero) and the trajectory is unusable.

This is one of the easiest mistakes to make and one of the hardest to catch
after the fact, so we test it explicitly.
"""
import numpy as np
import pytest
from src.system.types import Query
from src.agents.policy_agent import FixedPolicy


def make_query(qid=0):
    return Query(
        id=f"q_{qid}",
        v_t=np.zeros(128, dtype=np.float32),
        k_t=10,
        filter_t={},
        sla_t=0.050,
        budget_t=0.010,
    )


def test_fixed_policy_propensity_always_positive():
    configs = [
        {"nprobe": 8, "rerank_k": 50, "early_stop": False},
        {"nprobe": 32, "rerank_k": 200, "early_stop": False},
        {"nprobe": 128, "rerank_k": 800, "early_stop": False},
    ]
    prices = [0.001, 0.005, 0.01]
    pol = FixedPolicy(configs, prices, default_z_index=1, default_p_index=1, epsilon=0.1, seed=0)

    for i in range(500):
        action, propensity, version = pol.decide(make_query(i), U_t=0.5, h_t={})
        assert propensity > 0.0, f"zero propensity at iter {i}, action={action}"
        assert propensity <= 1.0, f"propensity > 1 at iter {i}: {propensity}"


def test_propensity_normalised():
    """Across many calls with same state, propensities should be consistent with mixture."""
    configs = [{"nprobe": 8, "rerank_k": 50, "early_stop": False}]
    prices = [0.001, 0.005]
    pol = FixedPolicy(configs, prices, default_z_index=0, default_p_index=0, epsilon=0.2, seed=0)

    # n_actions = 1 * 2 = 2
    # default: z=0, p=0 → propensity = (1-0.2) + 0.2 * 0.5 = 0.9
    # other:   z=0, p=1 → propensity = 0.2 * 0.5 = 0.1
    counts = {0: 0, 1: 0}
    propensities = {0: None, 1: None}
    for _ in range(10000):
        action, propensity, _ = pol.decide(make_query(0), U_t=0.0, h_t={})
        p_idx = prices.index(action.p_t)
        counts[p_idx] += 1
        propensities[p_idx] = propensity

    # Empirical frequencies should match logged propensities (loose check)
    emp_default = counts[0] / 10000
    assert abs(emp_default - propensities[0]) < 0.05, (
        f"empirical {emp_default} vs logged {propensities[0]}"
    )
