#!/usr/bin/env python3
"""
异质市场实验：验证动态定价能否击败强 Fixed baseline。

四维体系化设计：
  1. 战场 — PersonaWorkloadBuilder（企业/预算/质量 + easy/hard 成本分化）
  2. 武器 — (nprobe, price) 套餐组合动作空间
  3. 眼睛 — 10 维 HeteroLinUCB 状态（含预估成本 + persona）
  4. 指挥棒 — satisfaction_retention 奖励塑形

Usage:
    BUYER_VERSION=hetero python scripts/run_hetero_experiment.py --config configs/sift1m_hetero.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime

import faiss
import numpy as np
import yaml
from tqdm import tqdm

# Must set before importing BuyerSimulator
os.environ.setdefault("BUYER_VERSION", "hetero")

from src.agents.difficulty_estimator import MLPDifficultyEstimator
from src.agents.execution_agent import ExecutionAgent
from src.agents.hetero_bandit_policy import HeteroLinUCBPolicy
from src.agents.policy_agent import FixedPolicy
from src.agents.shadow_sampler import ShadowSampler
from src.data.buyer_simulator import BuyerSimulator, BUYER_VERSION
from src.data.datasets import load_dataset
from src.data.persona_workload import PersonaWorkloadBuilder, summarize_persona_workload
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.system.orchestrator import Orchestrator
from src.system.types import Query


def make_difficulty_fn(diff_est):
    def _fn(v: np.ndarray, k_t: int) -> float:
        q = Query(
            id="_probe", v_t=v.copy(), k_t=k_t, filter_t={},
            sla_t=0.05, budget_t=0.01,
        )
        return float(diff_est.estimate(q))
    return _fn


def run_policy(
    policy,
    orch: Orchestrator,
    buyer: BuyerSimulator,
    queries: list[Query],
    seed: int,
    gt,
) -> dict:
    n = len(queries)
    total = 0.0
    accepts = 0
    prices, nprobes, margins = [], [], []
    persona_stats: dict[str, dict] = {}

    for i in tqdm(range(n), desc=policy.version, leave=False):
        q = queries[i]
        buyer.rng = np.random.default_rng(seed + i)
        outcome = orch.handle_query(q, buyer, gt_ids=gt[i % len(gt)] if gt is not None else None)
        if hasattr(policy, "update"):
            policy.update(outcome.R_t)
        total += outcome.R_t
        accepts += int(outcome.A_t)
        if hasattr(orch, "_last_action") and orch._last_action:
            p = orch._last_action.p_t
            np_ = orch._last_action.z_t.get("nprobe", 0)
            prices.append(p)
            nprobes.append(np_)
            margins.append(p - outcome.C_t)
            ps = persona_stats.setdefault(q.persona_t, {"rev": 0.0, "n": 0, "acc": 0})
            ps["rev"] += outcome.R_t
            ps["n"] += 1
            ps["acc"] += int(outcome.A_t)

    return {
        "revenue": total,
        "accept_rate": accepts / n,
        "avg_price": float(np.mean(prices)) if prices else 0.0,
        "avg_nprobe": float(np.mean(nprobes)) if nprobes else 0.0,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "persona_stats": persona_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sift1m_hetero.yaml")
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--policies", default="fixed,hetero_linucb")
    ap.add_argument("--reward-mode", default=None,
                    help="hard | satisfaction | satisfaction_retention")
    ap.add_argument("--output", default=None)
    ap.add_argument("--index-path", default=None)
    ap.add_argument("--shadow-rate", type=float, default=0.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_queries = args.n_queries or cfg["experiment"]["n_queries"]
    reward_mode = args.reward_mode or cfg.get("pricing", {}).get("reward_mode", "satisfaction_retention")
    seed_list = [int(s) for s in args.seeds.split(",")]

    if BUYER_VERSION != "hetero":
        print(f"Warning: BUYER_VERSION={BUYER_VERSION}, set BUYER_VERSION=hetero for full design")

    print(f"Loading {cfg['dataset']['name']}...")
    xb, xq, xt, gt = load_dataset(cfg)

    index_path = args.index_path or os.path.join(cfg["dataset"]["path"], "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    diff_est = MLPDifficultyEstimator(
        onnx_path=cfg.get("models", {}).get("difficulty", "models/sift1m/difficulty_v1.onnx"),
        sample_vectors=xt[:5000],
    )
    difficulty_fn = make_difficulty_fn(diff_est)

    z_cfgs = cfg["execution"]["search_param_configs"]
    prices = cfg["pricing"]["tiers"]
    builder = PersonaWorkloadBuilder(cfg["cost_model"], prices)
    execution = ExecutionAgent(index, cfg["cost_model"])

    policy_names = [p.strip() for p in args.policies.split(",")]
    rows = []

    for seed in seed_list:
        queries, metas = builder.generate_sequence(xq, n_queries, seed, difficulty_fn)
        wstats = summarize_persona_workload(queries, metas)
        print(f"\n[seed={seed}] workload: {wstats}")

        buyer = BuyerSimulator(
            seed=seed,
            best_dist_anchor=cfg.get("buyer", {}).get("best_dist_anchor", 40000.0),
            worst_dist_anchor=cfg.get("buyer", {}).get("worst_dist_anchor", 150000.0),
            nprobe_recall=cfg.get("buyer", {}).get("nprobe_recall"),
        )

        for pname in policy_names:
            if pname == "fixed":
                policy = FixedPolicy(
                    z_cfgs, prices,
                    default_z_index=cfg.get("policies", {}).get("fixed", {}).get("z_index", 2),
                    default_p_index=cfg.get("policies", {}).get("fixed", {}).get("p_index", 2),
                    epsilon=cfg.get("policies", {}).get("fixed", {}).get("epsilon", 0.1),
                    seed=seed,
                )
            elif pname == "hetero_linucb":
                hc = cfg.get("policies", {}).get("hetero_linucb", {})
                policy = HeteroLinUCBPolicy(
                    z_cfgs, prices, cfg["cost_model"],
                    alpha=hc.get("alpha", 0.5),
                    temperature=hc.get("temperature", 0.4),
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown policy: {pname}")

            log_dir = os.path.join(cfg["logging"]["output_dir"], f"hetero_{pname}_{seed}_{int(time.time())}")
            log_writer = LogWriter(log_dir, flush_every_n=1000)
            shadow = ShadowSampler(xb, args.shadow_rate, max_workers=2,
                                   on_recall_computed=log_writer.record_recall, seed=seed)
            orch = Orchestrator(
                diff_est, policy, execution, shadow, log_writer,
                ContextCache(100), reward_mode=reward_mode,
            )
            print(f"\n--- {pname} seed={seed} reward={reward_mode} ---")
            metrics = run_policy(policy, orch, buyer, queries, seed, gt)
            shadow.drain(timeout=30)
            shadow.shutdown()
            log_writer.close()

            print(
                f"  revenue=${metrics['revenue']:.4f} accept={metrics['accept_rate']:.2%} "
                f"avg_p=${metrics['avg_price']:.4f} avg_nprobe={metrics['avg_nprobe']:.1f}"
            )
            rows.append({
                "seed": seed,
                "policy": pname,
                "reward_mode": reward_mode,
                "n_queries": n_queries,
                **{k: v for k, v in metrics.items() if k != "persona_stats"},
                **{f"w_{k}": v for k, v in wstats.items()},
            })

    out = args.output or f"reports/hetero_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out}")

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["policy"]].append(r["revenue"])
    print("\n=== Summary ===")
    for pname, revs in agg.items():
        print(f"  {pname:14s}  mean_revenue=${np.mean(revs):.4f}  (n_seeds={len(revs)})")


if __name__ == "__main__":
    main()
