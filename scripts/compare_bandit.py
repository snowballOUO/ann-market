"""
W3 comparison: FixedPolicy (W1 baseline) vs LinUCB (W3).

Runs both policies on identical query/buyer sequences.
Plots cumulative revenue curves.

Usage:
    python scripts/compare_bandit.py --config configs/base.yaml --n-queries 10000
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

from src.data.datasets import load_sift1m, load_hdf5
from src.agents.difficulty_estimator import DifficultyEstimator, MLPDifficultyEstimator
from src.agents.policy_agent import FixedPolicy  # W1 baseline
from src.agents.bandit_policy import LinUCBPolicy  # W3 new
from src.agents.execution_agent import ExecutionAgent
from src.agents.shadow_sampler import ShadowSampler
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.system.orchestrator import Orchestrator
from src.system.types import Query


def build_query(qid: int, v: np.ndarray, rng: np.random.Generator) -> Query:
    """Identical to run_experiment.py's build_query."""
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
    """
    用指定 policy 跑 n_queries 条查询。

    关键设计：buyer 的随机决策在两组实验里必须一致。
    做法：每个 query i 用 buyer_rng = np.random.default_rng(seed + i)
    来播种，保证两组实验的 buyer 产生相同的随机数序列。

    Returns:
        cumulative:  每条查询后的累计收益 (shape n_queries,)
        accept_rate: 最终接受率
        total_rev:   累计总收入
    """
    rng = np.random.default_rng(seed)
    n_qv = xq.shape[0]
    cumulative = np.zeros(n_queries)
    total = 0.0
    accepts = 0

    for i in tqdm(range(n_queries), desc=policy.version, leave=False):
        v = xq[i % n_qv]
        q = build_query(i, v, rng)

        # 重置 buyer 状态（两组实验同一个查询的 buyer 行为一致）
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
    ap.add_argument("--output-dir", default="reports/figs")
    ap.add_argument("--mlp", action="store_true", help="Use MLP difficulty estimator")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_queries = args.n_queries
    seed = cfg["experiment"]["seed"]

    # ── 加载数据 ──
    name = cfg["dataset"]["name"]
    print(f"Loading {name}...")
    if name == "ag_news":
        filepath = os.path.join(cfg["dataset"]["path"], cfg["dataset"]["file"])
        xb, xq, xt, gt = load_hdf5(filepath)
    else:
        xb, xq, xt, gt = load_sift1m(cfg["dataset"]["path"])

    index_path = args.index_path or os.path.join(cfg["dataset"]["path"], "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    # ── 共享组件（两组实验一致）──
    if args.mlp:
        diff_est = MLPDifficultyEstimator(
            onnx_path="models/difficulty_v1.onnx",
            sample_vectors=xt[:5000],
        )
    else:
        diff_est = DifficultyEstimator(sample_vectors=xt[:5000])
    execution = ExecutionAgent(index, cfg["cost_model"])
    from src.data.buyer_simulator import BuyerSimulator
    buyer = BuyerSimulator(seed=seed)

    # ── 跑 FixedPolicy ──
    print("\n=== FixedPolicy (W1) ===")
    fixed = FixedPolicy(
        cfg["execution"]["search_param_configs"],
        cfg["pricing"]["tiers"],
        default_z_index=2, default_p_index=2,
        epsilon=0.1, seed=seed,
    )

    log_dir = cfg["logging"]["output_dir"]
    log_fixed = LogWriter(os.path.join(log_dir, "cmp_fixed"), flush_every_n=1000)
    shadow_fixed = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                                max_workers=2, on_recall_computed=log_fixed.record_recall, seed=seed)
    orch_fixed = Orchestrator(diff_est, fixed, execution,
                                shadow_fixed, log_fixed, ContextCache(100))

    cum_fixed, ar_fixed, rev_fixed = run_policy(
        fixed, orch_fixed, buyer, xq, n_queries, seed)
    shadow_fixed.drain()
    shadow_fixed.shutdown()
    log_fixed.close()

    # ── 跑 LinUCB ──
    print("\n=== LinUCB (W3) ===")
    linucb = LinUCBPolicy(
        cfg["execution"]["search_param_configs"],
        cfg["pricing"]["tiers"],
        alpha=args.alpha, temperature=args.temperature, seed=seed,
    )

    log_linucb = LogWriter(os.path.join(log_dir, "cmp_linucb"), flush_every_n=1000)
    shadow_linucb = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                                    max_workers=2, on_recall_computed=log_linucb.record_recall, seed=seed)
    orch_linucb = Orchestrator(diff_est, linucb, execution,
                                shadow_linucb, log_linucb, ContextCache(100))

    cum_linucb, ar_linucb, rev_linucb = run_policy(
        linucb, orch_linucb, buyer, xq, n_queries, seed)
    shadow_linucb.drain()
    shadow_linucb.shutdown()
    log_linucb.close()

    # ── 结果 ──
    improvement = (rev_linucb - rev_fixed) / abs(rev_fixed) * 100
    print(f"\n{'='*60}")
    print(f"Results ({n_queries} queries)")
    print(f"{'='*60}")
    print(f"  FixedPolicy  revenue: ${rev_fixed:.4f}  accept: {ar_fixed:.2%}")
    print(f"  LinUCB       revenue: ${rev_linucb:.4f}  accept: {ar_linucb:.2%}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  {'PASS' if improvement > 15 else 'FAIL'} (threshold: >15%)")

    # ── 画图 ──
    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    queries = np.arange(1, n_queries + 1)
    ax.plot(queries, cum_fixed, label='FixedPolicy (ε=0.1)', linewidth=1.5, alpha=0.85)
    ax.plot(queries, cum_linucb, label=f'LinUCB (α={args.alpha}, τ={args.temperature})',
            linewidth=1.5, alpha=0.85)
    ax.fill_between(queries, cum_fixed, cum_linucb,
                    alpha=0.15, color='green' if rev_linucb > rev_fixed else 'red')
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Cumulative Revenue (USD)')
    ax.set_title(f'FixedPolicy vs LinUCB\nImprovement: {improvement:+.1f}%')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.05,
            f'Δ = {improvement:+.1f}%\nFixed: ${rev_fixed:.2f}\nLinUCB: ${rev_linucb:.2f}',
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'bandit_vs_fixed_revenue.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
      main()