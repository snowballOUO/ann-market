#!/usr/bin/env python3
"""
Must-serve 场景实验：S1–S6 + S_mix，Fixed vs HeteroLinUCB。

规则：永不拒单，策略仅在可行域 (cost<=budget, p>=cost) 内选套餐。

Usage:
    BUYER_VERSION=must_serve python scripts/run_scenario_experiment.py \\
        --config configs/sift1m_scenario.yaml --scenarios all --seeds 42,123,456

    # 多进程并行（默认用满 CPU，每个 worker 内 FAISS 单线程避免过度抢占）
    python scripts/run_scenario_experiment.py --workers 32
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import faiss
import numpy as np
import yaml

os.environ.setdefault("BUYER_VERSION", "must_serve")

# 进程池全局上下文（每个 worker 初始化一次）
_POOL: dict[str, Any] = {}


def _init_worker(cfg_path: str, shadow_rate: float, faiss_threads: int) -> None:
    os.environ["BUYER_VERSION"] = "must_serve"
    if faiss_threads > 0:
        faiss.omp_set_num_threads(faiss_threads)

    from src.agents.difficulty_estimator import MLPDifficultyEstimator
    from src.data.datasets import load_dataset
    from src.data.scenario_workload import ScenarioWorkloadBuilder
    from src.system.types import Query

    cfg = yaml.safe_load(open(cfg_path))
    xb, xq, xt, gt = load_dataset(cfg)
    index_path = os.path.join(cfg["dataset"]["path"], "index_ivfpq.faiss")
    index = faiss.read_index(index_path)

    diff_est = MLPDifficultyEstimator(
        onnx_path=cfg.get("models", {}).get("difficulty", "models/sift1m/difficulty_v1.onnx"),
        sample_vectors=xt[:5000],
    )

    def difficulty_fn(v: np.ndarray, k_t: int) -> float:
        q = Query(
            id="_p", v_t=v.copy(), k_t=k_t, filter_t={},
            sla_t=0.05, budget_t=0.01,
        )
        return float(diff_est.estimate(q))

    builder = ScenarioWorkloadBuilder(cfg["cost_model"], cfg["pricing"]["tiers"])
    _POOL.clear()
    _POOL.update({
        "cfg": cfg,
        "cfg_path": cfg_path,
        "xb": xb,
        "xq": xq,
        "gt": gt,
        "index": index,
        "diff_est": diff_est,
        "difficulty_fn": difficulty_fn,
        "builder": builder,
        "shadow_rate": shadow_rate,
    })


def _run_one_job(job: dict) -> dict:
    """单个 (scenario, seed, policy) 实验 — 在 worker 进程中执行。"""
    from src.agents.execution_agent import ExecutionAgent
    from src.agents.feasible_fixed_policy import FeasibleFixedPolicy
    from src.agents.hetero_bandit_policy import HeteroLinUCBPolicy
    from src.agents.shadow_sampler import ShadowSampler
    from src.data.buyer_simulator import BuyerSimulator, BUYER_VERSION
    from src.system.context_cache import ContextCache
    from src.system.log_writer import LogWriter
    from src.system.orchestrator import Orchestrator

    scenario = job["scenario"]
    seed = job["seed"]
    pname = job["policy"]
    n_queries = job["n_queries"]

    cfg = _POOL["cfg"]
    xb, xq, gt = _POOL["xb"], _POOL["xq"], _POOL["gt"]
    index = _POOL["index"]
    builder = _POOL["builder"]
    difficulty_fn = _POOL["difficulty_fn"]
    diff_est = _POOL["diff_est"]

    queries, _ = builder.generate_sequence(scenario, xq, n_queries, seed, difficulty_fn)

    z_cfgs = cfg["execution"]["search_param_configs"]
    prices = cfg["pricing"]["tiers"]
    fp = cfg.get("policies", {}).get("fixed", {})
    hp = cfg.get("policies", {}).get("hetero_linucb", {})

    if pname == "fixed":
        policy = FeasibleFixedPolicy(
            z_cfgs, prices, cfg["cost_model"],
            default_z_index=fp.get("z_index", 2),
            default_p_index=fp.get("p_index", 2),
            epsilon=fp.get("epsilon", 0.1),
            seed=seed,
        )
    elif pname == "hetero_linucb":
        policy = HeteroLinUCBPolicy(
            z_cfgs, prices, cfg["cost_model"],
            alpha=hp.get("alpha", 0.5),
            temperature=hp.get("temperature", 0.4),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown policy: {pname}")

    buyer = BuyerSimulator(
        seed=seed,
        best_dist_anchor=cfg.get("buyer", {}).get("best_dist_anchor", 40000.0),
        worst_dist_anchor=cfg.get("buyer", {}).get("worst_dist_anchor", 150000.0),
        nprobe_recall=cfg.get("buyer", {}).get("nprobe_recall"),
    )
    execution = ExecutionAgent(index, cfg["cost_model"])

    log_dir = os.path.join(
        cfg["logging"]["output_dir"],
        f"scenario_{scenario}_{pname}_{seed}_{os.getpid()}",
    )
    lw = LogWriter(log_dir, flush_every_n=2000)
    shadow = ShadowSampler(
        xb, _POOL["shadow_rate"], max_workers=1,
        on_recall_computed=lw.record_recall, seed=seed,
    )
    orch = Orchestrator(
        diff_est, policy, execution, shadow, lw,
        ContextCache(100), reward_mode="hard",
    )

    total, s_sum, accepts = 0.0, 0.0, 0
    prices_out, nprobes, margins = [], [], []
    for i in range(n_queries):
        q = queries[i]
        buyer.rng = np.random.default_rng(seed + i)
        outcome = orch.handle_query(
            q, buyer, gt_ids=gt[i % len(gt)] if gt is not None else None,
        )
        if hasattr(policy, "update"):
            policy.update(outcome.R_t)
        total += outcome.R_t
        s_sum += outcome.S_t or 0.0
        accepts += int(outcome.A_t)
        if hasattr(orch, "_last_action") and orch._last_action:
            p = orch._last_action.p_t
            prices_out.append(p)
            nprobes.append(orch._last_action.z_t.get("nprobe", 0))
            margins.append(p - outcome.C_t)

    shadow.drain(timeout=30)
    shadow.shutdown()
    lw.close()

    return {
        "scenario": scenario,
        "seed": seed,
        "policy": pname,
        "n_queries": n_queries,
        "buyer_version": BUYER_VERSION,
        "revenue": total,
        "accept_rate": accepts / n_queries,
        "mean_satisfaction": s_sum / n_queries,
        "avg_price": float(np.mean(prices_out)) if prices_out else 0.0,
        "avg_nprobe": float(np.mean(nprobes)) if nprobes else 0.0,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sift1m_scenario.yaml")
    ap.add_argument("--scenarios", default="all", help="S1,S2,... or all")
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--policies", default="fixed,hetero_linucb")
    ap.add_argument("--output", default=None)
    ap.add_argument("--shadow-rate", type=float, default=0.0)
    ap.add_argument(
        "--workers", type=int, default=0,
        help="并行进程数，0=自动(min(任务数, CPU核数))",
    )
    ap.add_argument(
        "--faiss-threads", type=int, default=1,
        help="每个 worker 内 FAISS OpenMP 线程数（建议 1–2，避免 86 核过载）",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = yaml.safe_load(open(cfg_path))
    n_queries = args.n_queries or cfg["experiment"]["n_queries"]
    seeds = [int(s) for s in args.seeds.split(",")]

    from src.data.scenario_workload import SCENARIO_IDS, SCENARIO_SPEC

    if args.scenarios.strip().lower() == "all":
        scenarios = [s for s in SCENARIO_IDS if s != "S_mix"] + ["S_mix"]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    policy_names = [p.strip() for p in args.policies.split(",")]
    jobs = [
        {"scenario": sc, "seed": sd, "policy": pn, "n_queries": n_queries}
        for sc in scenarios
        for sd in seeds
        for pn in policy_names
    ]

    n_cpu = os.cpu_count() or 4
    workers = args.workers or min(len(jobs), n_cpu)
    workers = max(1, min(workers, len(jobs)))

    print(f"Dataset: {cfg['dataset']['name']}  queries/job={n_queries}")
    print(f"Jobs: {len(jobs)}  workers={workers}  faiss_threads/worker={args.faiss_threads}")
    print(f"CPUs available: {n_cpu}")

    t0 = time.time()
    rows: list[dict] = []

    if workers == 1:
        _init_worker(cfg_path, args.shadow_rate, args.faiss_threads)
        for j, job in enumerate(jobs):
            print(f"[{j+1}/{len(jobs)}] {job['scenario']} seed={job['seed']} {job['policy']}")
            row = _run_one_job(job)
            rows.append(row)
            print(
                f"  → rev=${row['revenue']:.4f} accept={row['accept_rate']:.1%} "
                f"S={row['mean_satisfaction']:.3f}"
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg_path, args.shadow_rate, args.faiss_threads),
        ) as pool:
            futures = {pool.submit(_run_one_job, job): job for job in jobs}
            done = 0
            for fut in as_completed(futures):
                job = futures[fut]
                row = fut.result()
                rows.append(row)
                done += 1
                print(
                    f"[{done}/{len(jobs)}] {row['scenario']} seed={row['seed']} "
                    f"{row['policy']} rev=${row['revenue']:.4f} "
                    f"accept={row['accept_rate']:.1%} S={row['mean_satisfaction']:.3f}"
                )

    elapsed = time.time() - t0
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["policy"]))

    out = args.output or f"reports/sift1m_scenario_must_serve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out}  ({elapsed:.1f}s total, {len(jobs)/elapsed:.2f} jobs/s)")

    print(f"\n{'='*60}\nRevenue summary (mean over seeds)\n{'='*60}")
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["scenario"], r["policy"])].append(r["revenue"])
    for scenario in scenarios:
        print(f"\n{scenario} ({SCENARIO_SPEC.get(scenario, ('mix',))}):")
        revs = {}
        for pname in policy_names:
            key = (scenario, pname)
            if key in agg:
                revs[pname] = float(np.mean(agg[key]))
                print(f"  {pname:14s} ${revs[pname]:.4f}")
        if len(revs) == 2 and policy_names[0] in revs and policy_names[1] in revs:
            f, h = policy_names[0], policy_names[1]
            if revs[f] > 0:
                delta = (revs[h] - revs[f]) / revs[f] * 100
                winner = h if revs[h] > revs[f] else f
                print(f"  → winner: {winner} ({delta:+.1f}%)")


if __name__ == "__main__":
    main()
