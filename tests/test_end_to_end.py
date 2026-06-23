"""
Smoke test: the full pipeline runs end-to-end on a tiny synthetic dataset
without crashing, and produces a non-empty parquet log with valid schema.
"""
import os
import tempfile
import numpy as np
import faiss
import pandas as pd

from src.system.types import Query
from src.system.orchestrator import Orchestrator
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.agents.difficulty_estimator import DifficultyEstimator
from src.agents.policy_agent import FixedPolicy
from src.agents.execution_agent import ExecutionAgent
from src.agents.shadow_sampler import ShadowSampler
from src.agents.learner_agent import LearnerAgent
from src.data.buyer_stub import StubBuyer


def test_end_to_end_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    dim = 32
    n_base = 2000
    n_query = 100
    xb = rng.standard_normal((n_base, dim)).astype(np.float32)
    xq = rng.standard_normal((n_query, dim)).astype(np.float32)
    xt = rng.standard_normal((1000, dim)).astype(np.float32)

    # Tiny IVF-PQ
    quantizer = faiss.IndexFlatL2(dim)
    nlist = 32
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
    index.train(xt)
    index.add(xb)
    index.nprobe = 8

    configs = [
        {"nprobe": 4,  "rerank_k": 20, "early_stop": False},
        {"nprobe": 16, "rerank_k": 60, "early_stop": False},
    ]
    prices = [0.001, 0.005, 0.01]
    cost_model = {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}

    log_dir = str(tmp_path / "logs")
    log_writer = LogWriter(log_dir, flush_every_n=20)

    diff = DifficultyEstimator(sample_vectors=xt[:500])
    policy = FixedPolicy(configs, prices, default_z_index=1, default_p_index=1, epsilon=0.2, seed=0)
    ex = ExecutionAgent(index, cost_model)
    shadow = ShadowSampler(
        base_vectors=xb,
        sample_rate=0.2,
        max_workers=1,
        on_recall_computed=log_writer.record_recall,
        seed=0,
    )
    ctx = ContextCache(window_size=20)
    buyer = StubBuyer(seed=0)
    orch = Orchestrator(diff, policy, ex, shadow, log_writer, ctx)

    n_queries = 50
    for i in range(n_queries):
        q = Query(
            id=f"q_{i}",
            v_t=xq[i % n_query],
            k_t=5,
            filter_t={},
            sla_t=0.100,
            budget_t=0.010,
        )
        outcome = orch.handle_query(q, buyer)
        assert outcome.L_t > 0
        assert outcome.C_t > 0

    shadow.drain(timeout=10)
    shadow.shutdown()
    log_writer.close()

    # Check log file exists and is sensible
    learner = LearnerAgent(log_dir)
    df = learner.read_trajectories()
    assert len(df) == n_queries, f"expected {n_queries} rows, got {len(df)}"
    # Critical schema checks
    for col in ["query_id", "propensity", "U_t", "z_nprobe", "p_t", "L_t", "R_t", "A_t"]:
        assert col in df.columns, f"missing column {col}"
    # Propensity must be > 0 everywhere
    assert (df["propensity"] > 0).all(), "found zero propensity"
    # At least some queries should be shadow-sampled
    shadow_count = df["Q_t"].notna().sum()
    assert shadow_count > 0, "no shadow recalls recorded"
    print(f"\nEnd-to-end OK: {n_queries} queries, {shadow_count} shadow-sampled")


def test_end_to_end_with_qnet(tmp_path):
    """Full pipeline with QLearningPolicy — ensures propensity > 0 and no crash."""
    import torch
    from src.models.q_net import LargeQNet
    from src.agents.q_learning_policy import QLearningPolicy

    rng = np.random.default_rng(0)
    dim = 32
    n_base = 2000
    xb = rng.standard_normal((n_base, dim)).astype(np.float32)
    xq = rng.standard_normal((100, dim)).astype(np.float32)
    xt = rng.standard_normal((1000, dim)).astype(np.float32)

    quantizer = faiss.IndexFlatL2(dim)
    nlist = 32
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
    index.train(xt)
    index.add(xb)
    index.nprobe = 8

    configs = [
        {"nprobe": 4,  "rerank_k": 20, "early_stop": False},
        {"nprobe": 16, "rerank_k": 60, "early_stop": False},
    ]
    prices = [0.001, 0.005, 0.01]
    cost_model = {"base_per_ms": 0.00005, "fixed_overhead": 0.0001}

    # ── Create a synthetic Q-net model (random weights) for testing ──
    state_dim = 6
    n_actions = len(configs) * len(prices)  # 6
    synthetic_qnet = LargeQNet(state_dim, n_actions)
    model_path = str(tmp_path / "test_qnet.pt")
    dummy = torch.randn(1, state_dim)
    traced = torch.jit.trace(synthetic_qnet, dummy)
    traced.save(model_path)

    log_dir = str(tmp_path / "logs")
    log_writer = LogWriter(log_dir, flush_every_n=20)

    diff = DifficultyEstimator(sample_vectors=xt[:500])
    policy = QLearningPolicy(configs, prices, model_path=model_path, temperature=0.1)
    ex = ExecutionAgent(index, cost_model)
    shadow = ShadowSampler(
        base_vectors=xb, sample_rate=0.2, max_workers=1,
        on_recall_computed=log_writer.record_recall, seed=0,
    )
    ctx = ContextCache(window_size=20)
    buyer = StubBuyer(seed=0)
    orch = Orchestrator(diff, policy, ex, shadow, log_writer, ctx)

    n_queries = 50
    for i in range(n_queries):
        q = Query(
            id=f"q_{i}", v_t=xq[i % 100], k_t=5,
            filter_t={}, sla_t=0.100, budget_t=0.010,
        )
        outcome = orch.handle_query(q, buyer)
        assert outcome.L_t > 0

    shadow.drain(timeout=10)
    shadow.shutdown()
    log_writer.close()

    learner = LearnerAgent(log_dir)
    df = learner.read_trajectories()
    assert len(df) == n_queries
    assert (df["propensity"] > 0).all(), "Q-net produced zero propensity"
    shadow_count = df["Q_t"].notna().sum()
    assert shadow_count > 0
    print(f"\nEnd-to-end Q-net OK: {n_queries} queries, {shadow_count} shadow-sampled")
