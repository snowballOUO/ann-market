#!/usr/bin/env python3
"""
三档分层实验：验证自适应定价是否具备「看菜下碟」能力。

  Tier 1 — 随机极低价期望（恶劣环境，测算法是否因环境输）
  Tier 2 — 期望价围绕公允价微小波动（测小范围定价精度）
  Tier 3 — 期望价与成本强相关、两极分化（测能否区分易/难 query）

Usage:
    python scripts/run_tiered_experiment.py --config configs/sift1m.yaml --tiers all
    python scripts/run_tiered_experiment.py --config configs/sift1m.yaml --tiers 2,3 \\
        --policies fixed,linucb,qnet --n-queries 5000 --seeds 42,43,44
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

from src.agents.bandit_policy import LinUCBPolicy
from src.agents.cost_based_policy import CostBasedPolicy
from src.agents.difficulty_estimator import MLPDifficultyEstimator
from src.agents.execution_agent import ExecutionAgent
from src.agents.policy_agent import FixedPolicy
from src.agents.q_learning_policy import QLearningPolicy
from src.agents.shadow_sampler import ShadowSampler
from src.data.buyer_simulator import BuyerSimulator
from src.data.datasets import load_dataset
from src.data.tiered_workload import (
    TIER_DESCRIPTIONS,
    TIER_LABELS,
    TieredWorkloadBuilder,
    WorkloadTier,
    summarize_workload_with_queries,
)
from src.system.context_cache import ContextCache
from src.system.log_writer import LogWriter
from src.system.orchestrator import Orchestrator
from src.system.types import Query


def make_difficulty_fn(diff_est, xt_sample: np.ndarray):
    """U_t from vector only (no query contract fields)."""

    def _fn(v: np.ndarray, k_t: int) -> float:
        q = Query(
            id="_probe",
            v_t=v.copy(),
            k_t=k_t,
            filter_t={},
            sla_t=0.05,
            budget_t=0.01,
        )
        return float(diff_est.estimate(q))

    return _fn


def run_policy_on_sequence(
    policy,
    orch: Orchestrator,
    buyer: BuyerSimulator,
    queries: list[Query],
    seed: int,
    gt,
    update: bool = True,
) -> dict:
    n = len(queries)
    total = 0.0
    accepts = 0
    prices: list[float] = []
    budget_gaps: list[float] = []

    for i, q in enumerate(tqdm(range(n), desc=policy.version, leave=False)):
        query = queries[i]
        buyer.rng = np.random.default_rng(seed + i)
        outcome = orch.handle_query(
            query,
            buyer,
            gt_ids=gt[i % len(gt)] if gt is not None else None,
        )
        if update and hasattr(policy, "update"):
            policy.update(outcome.R_t)
        total += outcome.R_t
        if outcome.A_t:
            accepts += 1
        if hasattr(orch, "_last_action") and orch._last_action is not None:
            p = orch._last_action.p_t
            prices.append(p)
            budget_gaps.append(p - query.budget_t)

    return {
        "revenue": total,
        "accept_rate": accepts / n if n else 0.0,
        "avg_price": float(np.mean(prices)) if prices else 0.0,
        "avg_price_minus_budget": float(np.mean(budget_gaps)) if budget_gaps else 0.0,
    }


def build_policy(name: str, z_cfgs, prices, cfg, seed: int, args):
    if name == "fixed":
        return FixedPolicy(
            z_cfgs,
            prices,
            default_z_index=2,
            default_p_index=2,
            epsilon=args.epsilon,
            seed=seed,
        )
    if name == "linucb":
        return LinUCBPolicy(
            z_cfgs,
            prices,
            alpha=args.linucb_alpha,
            temperature=args.linucb_temperature,
            seed=seed,
        )
    if name == "qnet":
        model_path = args.qnet_model or cfg.get("models", {}).get(
            "qnet", "models/sift1m/qnet_distilled_v1.pt"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Q-Net model not found: {model_path}")
        return QLearningPolicy(
            z_cfgs,
            prices,
            model_path=model_path,
            temperature=args.qnet_temperature,
            seed=seed,
        )
    if name == "cost":
        return CostBasedPolicy(
            z_cfgs,
            prices,
            cfg["cost_model"],
            margin=args.cost_margin,
            seed=seed,
        )
    raise ValueError(f"Unknown policy: {name}")


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="三档分层定价实验")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tiers", default="all", help="1,2,3 或 all")
    ap.add_argument("--policies", default="fixed,linucb,qnet,cost")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seeds", default="42", help="逗号分隔，多 seed 取均值")
    ap.add_argument("--index-path", default=None)
    ap.add_argument("--output", default=None, help="CSV 输出路径")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--linucb-alpha", type=float, default=0.3)
    ap.add_argument("--linucb-temperature", type=float, default=0.5)
    ap.add_argument("--qnet-model", default=None)
    ap.add_argument("--qnet-temperature", type=float, default=0.1)
    ap.add_argument("--cost-margin", type=float, default=0.5)
    ap.add_argument("--u-easy-threshold", type=float, default=0.45)
    ap.add_argument("--shadow-rate", type=float, default=0.0,
                    help="分层实验默认关闭 shadow 以加速")
    ap.add_argument("--skip-missing-qnet", action="store_true",
                    help="Q-Net 模型缺失时跳过而非报错")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅生成负载统计，不跑策略（无需 FAISS 索引）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_list = parse_int_list(args.seeds)
    if args.tiers.strip().lower() == "all":
        tier_list = [1, 2, 3]
    else:
        tier_list = parse_int_list(args.tiers)

    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    valid = {"fixed", "linucb", "qnet", "cost"}
    for p in policy_names:
        if p not in valid:
            raise ValueError(f"Unknown policy '{p}'. Choices: {valid}")

    print(f"Loading {cfg['dataset']['name']}...")
    xb, xq, xt, gt = load_dataset(cfg)

    diff_est = MLPDifficultyEstimator(
        onnx_path=cfg.get("models", {}).get(
            "difficulty", "models/sift1m/difficulty_v1.onnx"
        ),
        sample_vectors=xt[:5000],
    )
    difficulty_fn = make_difficulty_fn(diff_est, xt[:5000])

    z_cfgs = cfg["execution"]["search_param_configs"]
    prices = cfg["pricing"]["tiers"]
    builder = TieredWorkloadBuilder(
        cfg["cost_model"],
        prices,
        margin=args.cost_margin,
        u_easy_threshold=args.u_easy_threshold,
    )

    if args.dry_run:
        for tier in tier_list:
            tier_enum = WorkloadTier(tier)
            print(f"\n{'=' * 70}")
            print(f"Tier {tier}: {TIER_LABELS[tier_enum]} — {TIER_DESCRIPTIONS[tier_enum]}")
            for seed in seed_list:
                queries, metas = builder.generate_sequence(
                    tier_enum, xq, args.n_queries, seed, difficulty_fn
                )
                wstats = summarize_workload_with_queries(queries, metas)
                print(f"  seed={seed}: {wstats}")
        return

    index_path = args.index_path or os.path.join(
        cfg["dataset"]["path"], "index_ivfpq.faiss"
    )
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run scripts/build_index.py first."
        )
    index = faiss.read_index(index_path)

    execution = ExecutionAgent(index, cfg["cost_model"])

    rows: list[dict] = []
    workload_stats: list[dict] = []

    for tier in tier_list:
        tier_enum = WorkloadTier(tier)
        label = TIER_LABELS[tier_enum]
        desc = TIER_DESCRIPTIONS[tier_enum]
        print(f"\n{'=' * 70}")
        print(f"Tier {tier}: {label}")
        print(f"  {desc}")
        print(f"{'=' * 70}")

        for seed in seed_list:
            queries, metas = builder.generate_sequence(
                tier_enum, xq, args.n_queries, seed, difficulty_fn
            )
            wstats = summarize_workload_with_queries(queries, metas)
            wstats["tier"] = tier
            wstats["seed"] = seed
            workload_stats.append(wstats)
            print(
                f"  [seed={seed}] budget_mean=${wstats['budget_mean']:.4f} "
                f"fair_mean=${wstats['fair_price_mean']:.4f} "
                f"ratio={wstats['budget_fair_ratio_mean']:.2f} "
                f"under80%={wstats['budget_below_80pct_fair_frac']:.1%}"
            )

            buyer = BuyerSimulator(
                seed=seed,
                best_dist_anchor=cfg.get("buyer", {}).get("best_dist_anchor", 40000.0),
                worst_dist_anchor=cfg.get("buyer", {}).get("worst_dist_anchor", 150000.0),
                nprobe_recall=cfg.get("buyer", {}).get("nprobe_recall"),
            )

            for pname in policy_names:
                if pname == "qnet":
                    model_path = args.qnet_model or cfg.get("models", {}).get(
                        "qnet", "models/sift1m/qnet_distilled_v1.pt"
                    )
                    if not os.path.exists(model_path):
                        if args.skip_missing_qnet:
                            print(f"  skip qnet (no model at {model_path})")
                            continue
                        raise FileNotFoundError(model_path)

                print(f"\n  --- {pname} (seed={seed}) ---")
                policy = build_policy(pname, z_cfgs, prices, cfg, seed, args)
                log_dir = os.path.join(
                    cfg["logging"]["output_dir"],
                    f"tier{tier}_{pname}_{seed}_{int(time.time())}",
                )
                log_writer = LogWriter(log_dir, flush_every_n=1000)
                shadow = ShadowSampler(
                    xb,
                    args.shadow_rate,
                    max_workers=2,
                    on_recall_computed=log_writer.record_recall,
                    seed=seed,
                )
                orch = Orchestrator(
                    diff_est,
                    policy,
                    execution,
                    shadow,
                    log_writer,
                    ContextCache(100),
                )
                metrics = run_policy_on_sequence(
                    policy, orch, buyer, queries, seed, gt,
                    update=(pname == "linucb"),
                )
                shadow.drain(timeout=30)
                shadow.shutdown()
                log_writer.close()

                print(
                    f"    revenue=${metrics['revenue']:.4f} "
                    f"accept={metrics['accept_rate']:.2%} "
                    f"avg_p=${metrics['avg_price']:.4f} "
                    f"p-budget={metrics['avg_price_minus_budget']:+.4f}"
                )
                rows.append({
                    "tier": tier,
                    "tier_label": label,
                    "seed": seed,
                    "policy": pname,
                    "n_queries": args.n_queries,
                    **metrics,
                    **{f"w_{k}": v for k, v in wstats.items() if k not in ("tier", "seed")},
                })

    out_path = args.output
    if out_path is None:
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"reports/tiered_experiment_{ts}.csv"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved results: {out_path}")

    print(f"\n{'=' * 70}")
    print("Summary (mean revenue by tier × policy)")
    print(f"{'=' * 70}")
    from collections import defaultdict
    agg: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["tier"], r["policy"])].append(r["revenue"])
    for tier in sorted({t for t, _ in agg}):
        print(f"\nTier {tier} ({TIER_LABELS[WorkloadTier(tier)]}):")
        for pname in policy_names:
            key = (tier, pname)
            if key not in agg:
                continue
            revs = agg[key]
            print(f"  {pname:8s}  revenue=${np.mean(revs):.4f}  (n_seeds={len(revs)})")


if __name__ == "__main__":
    main()
