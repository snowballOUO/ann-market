"""
  Validate DR estimator against online ground truth — Week 5.

  Usage:
      python scripts/validate_dr.py --config configs/base.yaml
          --log-dir logs/run_xxx --n-queries 5000

  For 3 test policies:
    1. Compute V̂_DR from historical LinUCB logs (offline)
    2. Run the policy online on the same buyer/environment (online)
    3. Compare — relative error must be < 5%
"""
import argparse
import os
import time
import yaml
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm

from src.data.datasets import load_sift1m, load_hdf5
from src.data.buyer_simulator import BuyerSimulator
from src.agents.difficulty_estimator import DifficultyEstimator
from src.agents.execution_agent import ExecutionAgent
from src.agents.shadow_sampler import ShadowSampler
from src.system.context_cache import ContextCache
from src.system.orchestrator import Orchestrator
from src.system.log_writer import LogWriter
from src.system.types import Query, Action
from src.causal.dr_estimator import (
      load_logs, build_state, build_action_index,
      RewardModel, dr_estimate,
      LowPricePolicy, QualityFocusPolicy, BudgetAwarePolicy,SafeDefaultPolicy,
)


class PolicyWrapper:
    """把 TestPolicy 包装成 Orchestrator 可用的 Policy。"""

    def __init__(self, test_policy, z_configs, price_tiers, seed=42):
        self.policy = test_policy
        self.z_configs = z_configs
        self.prices = price_tiers
        self.n_p = len(price_tiers)
        self.rng = np.random.default_rng(seed)
        self.version = test_policy.name

    def _build_features(self, query, U_t, h_t):
        """和 LinUCB._build_features() 完全一样。"""
        return np.array([
            U_t,
            h_t.get("recent_accept_rate", 0.5),
            h_t.get("recent_mean_latency", 0.0) * 1000,
            query.k_t / 100.0,
            query.sla_t * 1000,
            query.budget_t * 1000,
        ], dtype=np.float64)

    def decide(self, query, U_t, h_t):
        """Orchestrator 调用的接口。"""
        s = self._build_features(query, U_t, h_t)
        probs = self.policy.action_probs(s)

        action_idx = int(self.rng.choice(25, p=probs))
        propensity = float(probs[action_idx])

        z_idx = action_idx // self.n_p
        p_idx = action_idx % self.n_p
        z_t = dict(self.z_configs[z_idx])
        p_t = self.prices[p_idx]

        return Action(z_t=z_t, p_t=p_t), propensity, self.version


def online_evaluate(
        policy_wrapper,
        orchestrator,
        buyer,
        xq: np.ndarray,
        n_queries: int,
        seed: int,
) -> dict:
    """
    用 Orchestrator 在线跑 n_queries 条查询，返回累计收益和接受率。

    buyer 状态每条查询重置（和 compare_bandit 一样）。
    """
    rng = np.random.default_rng(seed)
    n_qv = xq.shape[0]
    total_revenue = 0.0
    total_accepts = 0

    for i in tqdm(range(n_queries), desc=policy_wrapper.version, leave=False):
        v = xq[i % n_qv]

        # 构造查询（和 run_experiment 一样）
        k = int(rng.choice([10, 20, 50, 100]))
        sla = float(rng.choice([0.020, 0.050, 0.100]))
        budget = float(rng.choice([0.005, 0.010, 0.020]))
        q = Query(id=f"q_{i:06d}", v_t=v.copy(),
                  k_t=k, filter_t={}, sla_t=sla, budget_t=budget)

        # 重置 buyer（保证和 DR 日志中的 buyer 行为分布一致）
        buyer.rng = np.random.default_rng(seed + i)
        if hasattr(buyer, 'market_sentiment'):
            buyer.market_sentiment = 0.8

        outcome = orchestrator.handle_query(q, buyer)
        total_revenue += outcome.R_t
        total_accepts += 1 if outcome.A_t else 0

    return {
        "total_revenue": total_revenue,
        "V_true": total_revenue / n_queries,
        "accept_rate": total_accepts / n_queries,
    }


def offline_evaluate(
        log_dir: str,
        policy,
        z_configs,
        price_tiers,
) -> dict:
    """从日志里用 DR 估计新策略的 V̂。"""
    df = load_logs(log_dir)
    S = build_state(df)
    A = build_action_index(df)
    R = df["R_t"].values
    pi_b = df["propensity"].values

    # 训 Q̂
    q_hat = RewardModel().fit(S, A, R)

    # 新策略的动作概率矩阵 (N, 25)
    pi_new = policy.get_action_probs_batch(S)

    # DR 估计
    result = dr_estimate(S, A, R, pi_b, pi_new, q_hat, clip_max=20)

    return {
        **result,
        "n_logs": len(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--index-path", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_dir = cfg["dataset"]["path"]
    seed = cfg["experiment"]["seed"]

    # ── 1. 加载数据 ──
    print(f"Loading dataset from {data_dir}...")
    name = cfg["dataset"]["name"]
    if name == "ag_news":
        xb, xq, xt, gt = load_hdf5(os.path.join(data_dir, cfg["dataset"]["file"]))
    else:
        xb, xq, xt, gt = load_sift1m(data_dir)

    index_path = args.index_path or os.path.join(data_dir, "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    # ── 2. 构造共享组件 ──
    diff_est = DifficultyEstimator(sample_vectors=xt[:5000])
    execution = ExecutionAgent(index, cfg["cost_model"])
    buyer = BuyerSimulator(seed=seed)

    z_configs = cfg["execution"]["search_param_configs"]
    price_tiers = cfg["pricing"]["tiers"]

    # ── 3. 定义 3 个测试策略 ──
    test_policies = [
        LowPricePolicy("LowPrice", z_configs, price_tiers),
        QualityFocusPolicy("QualityFocus", z_configs, price_tiers),
        SafeDefaultPolicy("SafeDefault", z_configs, price_tiers),
    ]

    # ── 4. 对每个策略：离线 DR + 在线真实 ──
    print(f"\n{'=' * 70}")
    print(f"DR Validation ({args.n_queries} online queries per policy)")
    print(f"{'=' * 70}")

    results = []
    for policy in test_policies:
        print(f"\n--- {policy.name} ---")

        # ── Offline DR 估计 ──
        print("  Computing DR estimate from logs...")
        offline = offline_evaluate(args.log_dir, policy, z_configs, price_tiers)
        V_dr = offline["V_dr"]
        V_dm = offline["V_dm"]
        V_ips = offline["V_ips"]
        print(f"    V_dr  = {V_dr:.6f}")
        print(f"    V_dm  = {V_dm:.6f}")
        print(f"    V_ips = {V_ips:.6f}")

        # ── Online 真实收益 ──
        print(f"  Running online ({args.n_queries} queries)...")
        wrapper = PolicyWrapper(policy, z_configs, price_tiers, seed=seed)

        # 为在线运行建独立的 Orchestrator
        log_dir_tmp = os.path.join(cfg["logging"]["output_dir"], f"validate_{policy.name}")
        log_writer = LogWriter(log_dir_tmp, flush_every_n=1000)
        shadow = ShadowSampler(xb, cfg["shadow"]["sample_rate"],
                               max_workers=2,
                               on_recall_computed=log_writer.record_recall,
                               seed=seed)
        orch = Orchestrator(diff_est, wrapper, execution,
                            shadow, log_writer, ContextCache(100))

        online = online_evaluate(wrapper, orch, buyer, xq, args.n_queries, seed)
        V_true = online["V_true"]

        shadow.drain()
        shadow.shutdown()
        log_writer.close()

        # ── 对比 ──
        rel_err = abs(V_dr - V_true) / max(abs(V_true), 1e-6)
        print(f"    V_true = {V_true:.6f}  (accept: {online['accept_rate']:.2%})")
        print(f"    Rel. error: {rel_err:.2%}  {'✅ PASS' if rel_err < 0.05 else '❌ FAIL'}")

        results.append({
            "policy": policy.name,
            "V_dr": V_dr,
            "V_dm": V_dm,
            "V_ips": V_ips,
            "V_true": V_true,
            "rel_error": rel_err,
            "accept_rate": online["accept_rate"],
            "mean_rho": offline["mean_rho"],
            "rho_p99": offline["rho_p99"],
        })

    # ── 5. 结果表 ──
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"{'Policy':<15} {'V_dr':>8} {'V_dm':>8} {'V_ips':>8} {'V_true':>8} {'Error':>8} {'PASS?':>6}")
    print(f"{'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
    for r in results:
        print(f"{r['policy']:<15} {r['V_dr']:>8.4f} {r['V_dm']:>8.4f} "
              f"{r['V_ips']:>8.4f} {r['V_true']:>8.4f} "
              f"{r['rel_error']:>7.2%} {'OK' if r['rel_error'] < 0.05 else 'FAIL':>6}")

    print(f"\nDR estimator bias check:")
    all_pass = all(r["rel_error"] < 0.05 for r in results)
    print(f"  {'✅ All 3 policies pass' if all_pass else '❌ Some policies fail'}")

    # IPS vs DR variance check
    ips_errors = [abs(r["V_ips"] - r["V_true"]) / max(abs(r["V_true"]), 1e-6) for r in results]
    dr_errors = [r["rel_error"] for r in results]
    print(f"\n  Avg DR  error: {np.mean(dr_errors):.2%}")
    print(f"  Avg IPS error: {np.mean(ips_errors):.2%}")
    print(f"  DR improvement: {'✅ IPS errors larger' if np.mean(ips_errors) > np.mean(dr_errors) else '⚠ Check unexpected'}")

    # Mean importance weight
    print(f"\nImportance weight diagnostics:")
    for r in results:
        print(f"{r['policy']:<15} mean ρ={r['mean_rho']:.2f}  p99 ρ={r['rho_p99']:.1f}")

if __name__ == "__main__":
    main()

