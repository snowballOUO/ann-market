"""
Diagnose: does the learned policy actually pick different actions for different buyers?
Runs Q-Net on queries from each buyer type, reports action distribution.
"""
import argparse, os, yaml, numpy as np, faiss
from collections import Counter
from src.data.datasets import load_dataset
from src.data.buyer_simulator import BuyerSimulator
from src.agents.difficulty_estimator import MLPDifficultyEstimator
from src.agents.q_learning_policy import QLearningPolicy
from src.system.context_cache import ContextCache
from src.system.types import Query

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sift1m.yaml")
    ap.add_argument("--model", default="models/sift1m/qnet.pt")
    ap.add_argument("--n-queries", type=int, default=2000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    xb, xq, xt, gt = load_dataset(cfg)
    seed = cfg["experiment"]["seed"]
    rng = np.random.default_rng(seed)

    diff_est = MLPDifficultyEstimator(
        onnx_path=cfg.get("models", {}).get("difficulty", "models/sift1m/difficulty_v1.onnx"),
        sample_vectors=xt[:5000],
    )
    z_cfgs = cfg["execution"]["search_param_configs"]
    prices = cfg["pricing"]["tiers"]
    policy = QLearningPolicy(z_cfgs, prices, model_path=args.model, temperature=0.03)

    ctx = ContextCache(window_size=100)
    buyer = BuyerSimulator(seed=seed,
        best_dist_anchor=cfg.get("buyer",{}).get("best_dist_anchor",40000),
        worst_dist_anchor=cfg.get("buyer",{}).get("worst_dist_anchor",150000))

    n_qv = xq.shape[0]
    results = {name: {"actions": [], "prices": [], "nprobes": [], "accepts": 0, "revenue": 0.0}
               for name in ["BudgetBuyer", "LatencyBuyer", "QualityBuyer"]}

    index = faiss.read_index(os.path.join(cfg["dataset"]["path"], "index_ivfpq.faiss"))
    from src.agents.execution_agent import ExecutionAgent
    exec_agent = ExecutionAgent(index, cfg["cost_model"])

    for i in range(args.n_queries):
        v = xq[i % n_qv]
        profile = buyer.get_profile(seed + i)
        name = profile.name

        # build query with correlated features
        k = int(rng.choice([10,20,50,100]))
        if name == "BudgetBuyer":
            sla = float(rng.choice([0.050, 0.100, 0.200]))
            budget = float(rng.choice([0.001, 0.002, 0.005]))
        elif name == "LatencyBuyer":
            sla = float(rng.choice([0.005, 0.010, 0.020]))
            budget = float(rng.choice([0.010, 0.020, 0.050]))
        else:
            sla = float(rng.choice([0.020, 0.050, 0.100]))
            budget = float(rng.choice([0.005, 0.010, 0.020]))
        q = Query(id=f"q_{i:06d}", v_t=v.copy(), k_t=k, filter_t={}, sla_t=sla, budget_t=budget)

        U_t = diff_est.estimate(q)
        h_t = ctx.get_features()
        h_t["market_sentiment"] = buyer.market_sentiment

        action, _, _ = policy.decide(q, U_t, h_t)
        nprobe = action.z_t.get("nprobe", 0)
        price = action.p_t

        results[name]["actions"].append(f"np{nprobe}_p{price:.3f}")
        results[name]["prices"].append(price)
        results[name]["nprobes"].append(nprobe)

        buyer.rng = np.random.default_rng(seed + i)
        a_res, L_t, C_t = exec_agent.search(q, action.z_t)
        A_t, S_t = buyer.respond(q, a_res, price, L_t)
        R_t = (price - C_t) if A_t else (-C_t)
        ctx.update(type('Outcome',(),{'A_t':A_t,'L_t':L_t,'R_t':R_t})())
        results[name]["accepts"] += int(A_t)
        results[name]["revenue"] += R_t

    print(f"\n{'='*70}")
    print(f"Q-Net action distribution by buyer type ({args.n_queries} queries each)")
    print(f"{'='*70}")
    for name, r in results.items():
        n = args.n_queries
        avg_price = np.mean(r["prices"])
        avg_nprobe = np.mean(r["nprobes"])
        top = Counter(r["actions"]).most_common(3)
        print(f"\n  {name}:")
        print(f"    avg price=${avg_price:.4f}  avg nprobe={avg_nprobe:.0f}"
              f"  accept={r['accepts']/n:.1%}  revenue=${r['revenue']:.2f}")
        print(f"    top actions: {top}")

if __name__ == "__main__":
    main()
