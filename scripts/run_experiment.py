"""
Main Week 1 experiment runner: streams N queries through the orchestrator.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml --n-queries 1000
"""
import argparse
import os
import time
import uuid
import yaml
import numpy as np
import faiss
from tqdm import tqdm

from src.data.datasets import load_sift1m, load_hdf5
from src.data.buyer_simulator import BuyerSimulator
from src.agents.difficulty_estimator import MLPDifficultyEstimator
from src.agents.bandit_policy import LinUCBPolicy
from src.agents.execution_agent import ExecutionAgent
from src.agents.shadow_sampler import ShadowSampler
from src.agents.learner_agent import LearnerAgent
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.system.orchestrator import Orchestrator
from src.system.types import Query
from src.agents.q_learning_policy import QLearningPolicy

def build_query(qid: int, v: np.ndarray, rng: np.random.Generator) -> Query:
    """Wrap a raw vector into a Query with synthetic SLA/budget/k."""
    # Vary k, SLA and budget so the system sees a realistic mix
    k = int(rng.choice([10, 20, 50, 100]))
    sla = float(rng.choice([0.020, 0.050, 0.100]))   # 20, 50, 100 ms
    budget = float(rng.choice([0.005, 0.010, 0.020]))
    # No filter in SIFT1M (no metadata); pass empty dict
    return Query(
        id=f"q_{qid:06d}",
        v_t=v,
        k_t=k,
        filter_t={},
        sla_t=sla,
        budget_t=budget,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--index-path", default=None)
    # add policy arg
    ap.add_argument("--policy", type=str, choices=["linucb", "qnet"], default="qnet", help="Which policy to run")
    # 
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_queries = args.n_queries or cfg["experiment"]["n_queries"]
    seed = cfg["experiment"]["seed"]

    # --- Load data ---
    data_dir = cfg["dataset"]["path"]
    name = cfg["dataset"]["name"]
    print(f"Loading {name} from {data_dir}...")
    if name == "ag_news":
        filepath = os.path.join(data_dir, cfg["dataset"]["file"])
        xb, xq, xt, gt = load_hdf5(filepath)
    else:
        xb, xq, xt, gt = load_sift1m(data_dir)

    index_path = args.index_path or os.path.join(data_dir, "index_ivfpq.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run scripts/build_index.py first."
        )
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)

    # --- Build agents ---
    print("Building agents...")
    # Use a subsample of xt for the difficulty estimator's density features
    diff_est = MLPDifficultyEstimator(
        onnx_path="models/difficulty_v1.onnx",
        sample_vectors=xt[:5000],
    )
    # add policy arg
    if args.policy == "linucb":
        policy = LinUCBPolicy(
            search_param_configs=cfg["execution"]["search_param_configs"],
            price_tiers=cfg["pricing"]["tiers"],
            alpha=1.0,
            temperature=0.5,
            seed=seed,
        )
    # 终极形态的在线策略！
    else:  # args.policy == "qnet"
        policy = QLearningPolicy(
            search_param_configs=cfg["execution"]["search_param_configs"],
            price_tiers=cfg["pricing"]["tiers"],
            model_path="models/qnet_distilled_v1.pt",
            temperature=0.1
        )
    # 
    execution = ExecutionAgent(index, cfg["cost_model"])

    # Output dirs
    run_id = f"run_{int(time.time())}"
    log_dir = os.path.join(cfg["logging"]["output_dir"], run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_writer = LogWriter(log_dir, flush_every_n=cfg["logging"]["flush_every_n"])

    shadow = ShadowSampler(
        base_vectors=xb,
        sample_rate=cfg["shadow"]["sample_rate"],
        max_workers=2,
        on_recall_computed=log_writer.record_recall,
        seed=seed,
    )

    buyer = BuyerSimulator(
        seed=seed,
        best_dist_anchor=cfg.get("buyer", {}).get("best_dist_anchor", 40000.0),
        worst_dist_anchor=cfg.get("buyer", {}).get("worst_dist_anchor", 150000.0),
    )
    ctx = ContextCache(window_size=100)
    orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ctx)

    # --- Run ---
    rng = np.random.default_rng(seed)
    # Cycle through query vectors if we request more than available
    n_qv = xq.shape[0]
    print(f"\nRunning {n_queries} queries (logs to {log_dir})...")
    start = time.time()
    accept_count = 0
    revenue_total = 0.0
    for i in tqdm(range(n_queries)):
        v = xq[i % n_qv]
        q = build_query(i, v, rng)
        outcome = orch.handle_query(q, buyer)
        # ── 新增：LinUCB 在线学习 ──
        # policy.update(outcome.R_t)
        if hasattr(policy, 'update'): policy.update(outcome.R_t)
        if outcome.A_t:
            accept_count += 1
            revenue_total += outcome.R_t

    elapsed = time.time() - start
    qps = n_queries / elapsed if elapsed > 0 else float("inf")

    # Drain shadow + close log
    print("\nDraining shadow sampler...")
    shadow.drain(timeout=60)
    shadow.shutdown()
    log_writer.close()

    print(f"\nDone in {elapsed:.1f}s ({qps:.0f} qps)")
    print(f"  accepts: {accept_count}/{n_queries} ({100*accept_count/n_queries:.1f}%)")
    print(f"  revenue: ${revenue_total:.4f}")

    # --- Learner summary ---
    print()
    learner = LearnerAgent(log_dir)
    learner.periodic_update()
    # ── W3: Bandit diagnostics ──
    if hasattr(policy, 'action_counts'):
        print()
        print("=" * 60)
        print("Bandit diagnostics")
        print("=" * 60)
        counts = policy.action_counts()
        n_untouched = int((counts == 0).sum())
        print(f"  actions never selected: {n_untouched}/{policy.n_actions}")
        print(f"  most selected: action {counts.argmax()} ({counts.max()} times)")
        print(f"  least selected: action {counts.argmin()} ({counts.min()} times)")
        print(f"  mean ||theta||: {policy.mean_theta_norm():.4f}")


if __name__ == "__main__":
    main()
