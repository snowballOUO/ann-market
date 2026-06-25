"""
W3+W6 comparison: FixedPolicy vs LinUCB vs Distilled Q-Net.

Runs policies on identical query/buyer sequences.
Plots cumulative revenue curves.

Usage:
    python scripts/compare_bandit.py --config configs/base.yaml --n-queries 10000
    python scripts/compare_bandit.py --config configs/base.yaml --policies fixed,linucb,qnet
"""
import argparse
import os
import yaml
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm

from src.data.datasets import load_dataset
from src.data.buyer_simulator import BuyerSimulator
from src.agents.difficulty_estimator import DifficultyEstimator, MLPDifficultyEstimator
from src.agents.policy_agent import FixedPolicy
from src.agents.bandit_policy import LinUCBPolicy
from src.agents.q_learning_policy import QLearningPolicy
from src.agents.execution_agent import ExecutionAgent
from src.agents.shadow_sampler import ShadowSampler
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.system.orchestrator import Orchestrator
from src.system.types import Query


def build_query(qid: int, v: np.ndarray, rng: np.random.Generator) -> Query:
    k = int(rng.choice([10, 20, 50, 100]))
    sla = float(rng.choice([0.020, 0.050, 0.100]))
    budget = float(rng.choice([0.005, 0.010, 0.020]))
    return Query(
        id=f"q_{qid:06d}",
        v_t=v.copy(),
        k_t=k,
        filter_t={},
        sla_t=sla,
        budget_t=budget,
    )


def run_policy(policy, orch, buyer, xq, n_queries, seed):
    """Run N queries, return cumulative revenue, accept rate, total revenue."""
    rng = np.random.default_rng(seed)
    n_qv = xq.shape[0]
    cumulative = np.zeros(n_queries)
    total = 0.0
    accepts = 0

    for i in tqdm(range(n_queries), desc=policy.version, leave=False):
        v = xq[i % n_qv]
        q = build_query(i, v, rng)

        buyer.rng = np.random.default_rng(seed + i)
        if hasattr(buyer, 'market_sentiment'):
            buyer.market_sentiment = 0.8

        outcome = orch.handle_query(q, buyer)

        if hasattr(policy, 'update'):
            policy.update(outcome.R_t)

        total += outcome.R_t
        accepts += 1 if outcome.A_t else 0
        cumulative[i] = total

    return cumulative, accepts / n_queries, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n-queries", type=int, default=10000)
    ap.add_argument("--index-path", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--qnet-temp", type=float, default=0.1)
    ap.add_argument("--output-dir", default="reports/figs")
    ap.add_argument("--mlp", action="store_true", help="Use MLP difficulty estimator")
    ap.add_argument("--margin", type=float, default=20.0,
                    help="Cost-based pricing margin (default: 20 = 2000%%)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override experiment seed")
    ap.add_argument("--qnet-model", default=None,
                    help="Path to Q-Net distilled model")
    ap.add_argument("--naive-dqn-model", default=None,
                    help="Path to Naive DQN distilled model")
    ap.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    ap.add_argument("--results-json", default=None, help="Write results to JSON file")
    ap.add_argument("--policies", type=str, default="fixed,linucb,qnet",
                    help="Comma-separated list: fixed,linucb,qnet,sla,cost,naive_dqn")
    args = ap.parse_args()

    selected = [p.strip() for p in args.policies.split(",")]
    valid = {"fixed", "linucb", "qnet", "sla", "cost", "naive_dqn"}
    for p in selected:
        if p not in valid:
            raise ValueError(f"Unknown policy '{p}'. Choices: {valid}")

    cfg = yaml.safe_load(open(args.config))
    n_queries = args.n_queries
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]

    # ── Load data ──
    print(f"Loading {cfg['dataset']['name']}...")
    xb, xq, xt, gt = load_dataset(cfg)

    index_path = args.index_path or os.path.join(cfg["dataset"]["path"], "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    # ── Shared components ──
    if args.mlp:
        diff_est = MLPDifficultyEstimator(
            onnx_path="models/difficulty_v1.onnx",
            sample_vectors=xt[:5000],
        )
    else:
        diff_est = DifficultyEstimator(sample_vectors=xt[:5000])
    execution = ExecutionAgent(index, cfg["cost_model"])
    buyer = BuyerSimulator(
        seed=seed,
        best_dist_anchor=cfg.get("buyer", {}).get("best_dist_anchor", 40000.0),
        worst_dist_anchor=cfg.get("buyer", {}).get("worst_dist_anchor", 150000.0),
    )
    log_dir = cfg["logging"]["output_dir"]
    z_configs = cfg["execution"]["search_param_configs"]
    price_tiers = cfg["pricing"]["tiers"]

    # ── Build & run each policy ──
    results = {}

    if "fixed" in selected:
        print("\n=== FixedPolicy (W1) ===")
        policy = FixedPolicy(z_configs, price_tiers,
                             default_z_index=2, default_p_index=2,
                             epsilon=0.1, seed=seed)
        log_writer = LogWriter(os.path.join(log_dir, "cmp_fixed"), flush_every_n=1000)
        shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                               max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
        orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
        cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
        shadow.drain(); shadow.shutdown(); log_writer.close()
        results["fixed"] = {"cum": cum, "ar": ar, "rev": rev, "label": "FixedPolicy (ε=0.1)"}

    if "linucb" in selected:
        print("\n=== LinUCB (W3) ===")
        policy = LinUCBPolicy(z_configs, price_tiers,
                              alpha=args.alpha, temperature=args.temperature, seed=seed)
        log_writer = LogWriter(os.path.join(log_dir, "cmp_linucb"), flush_every_n=1000)
        shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                               max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
        orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
        cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
        shadow.drain(); shadow.shutdown(); log_writer.close()
        results["linucb"] = {"cum": cum, "ar": ar, "rev": rev,
                             "label": f"LinUCB (α={args.alpha}, τ={args.temperature})"}

    if "qnet" in selected:
        model_path = args.qnet_model or "models/qnet_distilled_v1.pt"
        if not os.path.exists(model_path):
            print(f"\n⚠ Q-net model not found at {model_path}, skipping. Run train_qnet.py first.")
        else:
            print("\n=== Distilled Q-Net (W6) ===")
            policy = QLearningPolicy(z_configs, price_tiers,
                                     model_path=model_path, temperature=args.qnet_temp)
            log_writer = LogWriter(os.path.join(log_dir, "cmp_qnet"), flush_every_n=1000)
            shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                                   max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
            orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
            cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
            shadow.drain(); shadow.shutdown(); log_writer.close()
            results["qnet"] = {"cum": cum, "ar": ar, "rev": rev,
                               "label": f"Distilled Q-Net (τ={args.qnet_temp})"}

    if "sla" in selected:
        from src.agents.sla_heuristic_policy import SLAHeuristicPolicy
        print("\n=== SLA Heuristic ===")
        policy = SLAHeuristicPolicy(z_configs, price_tiers, seed=seed)
        log_writer = LogWriter(os.path.join(log_dir, "cmp_sla"), flush_every_n=1000)
        shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                               max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
        orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
        cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
        shadow.drain(); shadow.shutdown(); log_writer.close()
        results["sla"] = {"cum": cum, "ar": ar, "rev": rev, "label": "SLA Heuristic"}

    if "cost" in selected:
        from src.agents.cost_based_policy import CostBasedPolicy
        print("\n=== Cost-Based ===")
        policy = CostBasedPolicy(z_configs, price_tiers, cfg["cost_model"],
                                  margin=args.margin, seed=seed)
        log_writer = LogWriter(os.path.join(log_dir, "cmp_cost"), flush_every_n=1000)
        shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                               max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
        orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
        cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
        shadow.drain(); shadow.shutdown(); log_writer.close()
        results["cost"] = {"cum": cum, "ar": ar, "rev": rev,
                            "label": f"Cost-Based (margin={args.margin:.0f}x)"}

    if "naive_dqn" in selected:
        model_path = args.naive_dqn_model or "models/qnet_naive_dqn_v1.pt"
        if not os.path.exists(model_path):
            print(f"\n⚠ Naive DQN model not found. Train with: train_qnet.py --no-ut --output {model_path}")
        else:
            from src.agents.naive_dqn_policy import NaiveDQNPolicy
            print("\n=== Naive DQN (No Deconfounding) ===")
            policy = NaiveDQNPolicy(z_configs, price_tiers, model_path=model_path,
                                    temperature=args.qnet_temp)
            log_writer = LogWriter(os.path.join(log_dir, "cmp_naive_dqn"), flush_every_n=1000)
            shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                                   max_workers=2, on_recall_computed=log_writer.record_recall, seed=seed)
            orch = Orchestrator(diff_est, policy, execution, shadow, log_writer, ContextCache(100))
            cum, ar, rev = run_policy(policy, orch, buyer, xq, n_queries, seed)
            shadow.drain(); shadow.shutdown(); log_writer.close()
            results["naive_dqn"] = {"cum": cum, "ar": ar, "rev": rev,
                                    "label": "Naive DQN (no U_t)"}

    if not results:
        print("No policies ran.")
        return

    # ── Results table ──
    print(f"\n{'='*70}")
    print(f"Results ({n_queries} queries)")
    print(f"{'='*70}")
    for key, r in results.items():
        print(f"  {r['label']:>35s}  revenue: ${r['rev']:.4f}  accept: {r['ar']:.2%}")

    if args.results_json:
        import json
        out = {k: {"revenue": round(float(v["rev"]), 6), "accept_rate": round(float(v["ar"]), 6)}
               for k, v in results.items()}
        with open(args.results_json, "w") as f:
            json.dump(out, f)

    # ── Plot ──
    if not args.no_plot:
        colors = {"fixed": "#1f77b4", "linucb": "#ff7f0e", "qnet": "#2ca02c",
                  "sla": "#d62728", "cost": "#9467bd", "naive_dqn": "#8c564b"}
        os.makedirs(args.output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 6))
        queries = np.arange(1, n_queries + 1)
        for key, r in results.items():
            ax.plot(queries, r["cum"], label=r["label"], linewidth=1.5,
                    color=colors.get(key), alpha=0.85)
        ax.set_xlabel("Number of Queries")
        ax.set_ylabel("Cumulative Revenue (USD)")
        ax.set_title("Policy Comparison: FixedPolicy vs LinUCB vs Distilled Q-Net")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, "policy_comparison.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
