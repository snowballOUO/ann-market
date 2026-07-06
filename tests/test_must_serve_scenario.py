"""Tests for must-serve buyer, feasible actions, scenario workloads."""
import numpy as np
import pytest

from src.data.scenario_workload import SCENARIO_SPEC, ScenarioWorkloadBuilder
from src.pricing.feasible_actions import is_action_feasible, list_feasible_actions
from src.system.types import Query


@pytest.fixture
def cost_model():
    return {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}


@pytest.fixture
def configs_prices():
    configs = [
        {"nprobe": 8}, {"nprobe": 16}, {"nprobe": 32},
        {"nprobe": 64}, {"nprobe": 128},
    ]
    prices = [0.001, 0.002, 0.005, 0.01, 0.02]
    return configs, prices


def test_must_serve_always_accepts():
    import os
    os.environ["BUYER_VERSION"] = "must_serve"
    from importlib import reload
    import src.data.buyer_simulator as bs
    reload(bs)
    buyer = bs.BuyerSimulator(seed=0)
    q = Query(
        id="t", v_t=np.zeros(128), k_t=10, filter_t={},
        sla_t=0.05, budget_t=0.001, persona_t="budget",
        scenario_t="bargain_easy",
    )
    a, s = buyer.respond(q, [(1, 100.0)], price=0.02, latency=0.1, nprobe=8)
    assert a is True
    assert 0.0 <= s <= 1.0


def test_feasible_low_budget_prefers_low_nprobe(cost_model, configs_prices):
    configs, prices = configs_prices
    q = Query(
        id="t", v_t=np.zeros(128), k_t=10, filter_t={},
        sla_t=0.1, budget_t=0.00019,
    )
    feas = list_feasible_actions(q, configs, prices, cost_model)
    nprobes = [configs[z]["nprobe"] for z, _ in feas]
    assert max(nprobes) <= 16
    assert all(is_action_feasible(q, z, p, configs, prices, cost_model) for z, p in feas)


def test_scenario_s1_s6_budget_spread(cost_model):
    builder = ScenarioWorkloadBuilder(cost_model, [0.001, 0.002, 0.005, 0.01, 0.02])
    rng = np.random.default_rng(0)
    xq = np.random.randn(50, 128).astype(np.float32)
    builder.calibrate_u_thresholds(np.linspace(0, 1, 50))

    budgets = {}
    for sid in SCENARIO_SPEC:
        q, _ = builder.build_query(0, xq[0], 0.5, sid, rng)
        budgets[sid] = q.budget_t
    assert budgets["S5"] > budgets["S1"] * 3
    assert budgets["S1"] <= budgets["S3"]


def test_scenario_sequence_reproducible(cost_model):
    xq = np.random.randn(20, 128).astype(np.float32)
    builder = ScenarioWorkloadBuilder(cost_model, [0.001, 0.002, 0.005, 0.01, 0.02])
    fn = lambda v, k: float(np.linalg.norm(v) % 1.0)
    q1, _ = builder.generate_sequence("S3", xq, 50, 42, fn)
    q2, _ = builder.generate_sequence("S3", xq, 50, 42, fn)
    assert [q.budget_t for q in q1] == [q.budget_t for q in q2]
