"""Calibrate buyer distance anchors across 5 random query subsets."""
import yaml, faiss, numpy as np, sys

DATASETS = ["sift1m", "deep1m", "gist1m", "ag_news"]
N_SEEDS = 5
N_QUERY = 500
NPROBES = [8, 16, 32, 64, 128]

for ds in DATASETS:
    cfg = yaml.safe_load(open(f"configs/{ds}.yaml"))
    from src.data.datasets import load_dataset
    xb, xq, xt, gt = load_dataset(cfg)

    idx = faiss.read_index(f"{cfg['dataset']['path']}/index_ivfpq.faiss")
    n_total = xq.shape[0]

    best_vals, worst_vals = [], []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(42 + s)
        indices = rng.choice(n_total, size=min(N_QUERY, n_total), replace=False)
        all_dists = []
        for nprobe in NPROBES:
            idx.nprobe = nprobe
            D, I = idx.search(xq[indices], 50)
            all_dists.extend(D.mean(axis=1).tolist())
        dists = np.array(all_dists)
        best_vals.append(np.percentile(dists, 10))
        worst_vals.append(np.percentile(dists, 90))

    b_mean, b_std = np.mean(best_vals), np.std(best_vals)
    w_mean, w_std = np.mean(worst_vals), np.std(worst_vals)
    current_b = cfg["buyer"]["best_dist_anchor"]
    current_w = cfg["buyer"]["worst_dist_anchor"]
    ok = (0.7 * b_mean <= current_b <= 1.3 * b_mean) and (0.7 * w_mean <= current_w <= 1.3 * w_mean)

    print(f"\n{ds:10s} (dim={xq.shape[1]}, {N_QUERY}q × {N_SEEDS} seeds)")
    print(f"  best_anchor:  {b_mean:.2f} ± {b_std:.2f}  (current: {current_b})")
    print(f"  worst_anchor: {w_mean:.1f} ± {w_std:.1f}  (current: {current_w})")
    print(f"  suggested:    best={b_mean:.2f}  worst={w_mean:.1f}")
    print(f"  {'✅ OK' if ok else '❌ NEEDS FIX'}")
