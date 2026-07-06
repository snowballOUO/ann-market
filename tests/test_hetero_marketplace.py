"""Tests for heterogeneous marketplace redesign."""
import numpy as np
import pytest

from src.data.persona_workload import PersonaWorkloadBuilder, summarize_persona_workload
from src.pricing.reward import compute_reward
from src.pricing.state_features import build_hetero_state, PERSONA_BUDGET, PERSONA_ENTERPRISE
from src.system.types import Query


@pytest.fixture
def cost_model():
    return {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}


def test_persona_workload_spread(cost_model):
    builder = PersonaWorkloadBuilder(cost_model, [0.001, 0.002, 0.005, 0.01, 0.02])
    rng = np.random.default_rng(0)
    xq = np.random.randn(100, 128).astype(np.float32)

    def diff_fn(v, k):
        return float(np.linalg.norm(v) % 1.0)

    builder.calibrate_u_thresholds(np.array([diff_fn(xq[i], 32) for i in range(100)]))
    queries, metas = [], []
    for i in range(200):
        q, m = builder.build_query(i, xq[i % 100], diff_fn(xq[i % 100], 32), rng)
        queries.append(q)
        metas.append(m)

    stats = summarize_persona_workload(queries, metas)
    assert stats["budget_std"] > 0.002
    assert stats["enterprise_frac"] > 0.1
    assert stats["budget_frac"] > 0.3
    ent_budgets = [q.budget_t for q in queries if q.persona_t == PERSONA_ENTERPRISE]
    bud_budgets = [q.budget_t for q in queries if q.persona_t == PERSONA_BUDGET]
    assert np.mean(ent_budgets) > np.mean(bud_budgets) * 2


def test_hetero_state_has_cost_and_persona(cost_model):
    q = Query(
        id="t", v_t=np.zeros(128), k_t=50, filter_t={},
        sla_t=0.01, budget_t=0.01,
        persona_t=PERSONA_ENTERPRISE, difficulty_bucket="hard",
    )
    s = build_hetero_state(q, 0.8, {"recent_accept_rate": 0.5, "recent_mean_latency": 0.003}, cost_model)
    assert s.shape == (10,)
    assert s[8] == 1.0
    assert s[7] > 0


def test_satisfaction_reward_softer_than_hard():
    r_hard_reject = compute_reward(0.01, 0.001, False, 0.2, mode="hard")
    r_soft_reject = compute_reward(0.01, 0.001, False, 0.2, mode="satisfaction")
    assert r_soft_reject > r_hard_reject


def test_hetero_buyer_uses_query_persona():
    import os
    os.environ["BUYER_VERSION"] = "hetero"
    from importlib import reload
    import src.data.buyer_simulator as bs
    reload(bs)
    buyer = bs.BuyerSimulator(seed=0)
    q_ent = Query(id="e", v_t=np.zeros(128), k_t=10, filter_t={},
                  sla_t=0.01, budget_t=0.02, persona_t="enterprise")
    q_bud = Query(id="b", v_t=np.zeros(128), k_t=10, filter_t={},
                  sla_t=0.1, budget_t=0.002, persona_t="budget")
    p_ent = buyer.profile_for_query(q_ent)
    p_bud = buyer.profile_for_query(q_bud)
    assert p_ent.name == "Enterprise"
    assert p_bud.name == "Budget"
    prob_ent = bs.BuyerSimulator.compute_accept_probability(
        p_ent, q_ent, price=0.015, latency=0.003, perceived_recall=0.9)
    prob_bud = bs.BuyerSimulator.compute_accept_probability(
        p_bud, q_bud, price=0.015, latency=0.003, perceived_recall=0.9)
    assert prob_ent > prob_bud
