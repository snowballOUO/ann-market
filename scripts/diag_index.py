"""
Diagnose index quality: per-nprobe recall@10, distance distribution, anchor suggestions.
Usage: python scripts/diag_index.py --config configs/sift1m.yaml
"""
import argparse, os, yaml, numpy as np, faiss, time
from src.data.datasets import load_dataset


def recall_at_k(approx_ids, gt_ids, k):
    if gt_ids is None or len(gt_ids) == 0:
        return None
    inter = len(set(approx_ids[:k]) & set(gt_ids[:k]))
    return inter / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sift1m.yaml")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--nprobes", type=str, default="1,2,4,8,16,32,64,128,256,512,1024")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]
    print(f"Dataset: {ds['name']}  dim={ds.get('dim','?')}  format={ds.get('format','?')}")

    xb, xq, xt, gt = load_dataset(cfg)
    print(f"Base: {xb.shape}  Query: {xq.shape}  Train: {xt.shape}  GT: {gt.shape if gt is not None else 'None'}")

    index_path = os.path.join(ds["path"], "index_ivfpq.faiss")
    if not os.path.exists(index_path):
        alt = args.config.replace(".yaml", "").replace("configs/", "")
        index_path = f"data/{alt}/index_ivfpq.faiss"
    print(f"Index: {index_path}  (exists={os.path.exists(index_path)})")
    index = faiss.read_index(index_path)
    nlist = index.nlist if hasattr(index, 'nlist') else 4096
    print(f"Index nlist={nlist}")

    # Auto-generate nprobe levels: powers of 2 up to nlist
    nprobes = []
    n = 1
    while n <= nlist:
        nprobes.append(n)
        n *= 2
    if nprobes[-1] != nlist:
        nprobes.append(nlist)
    print(f"Testing nprobes: {nprobes}")

    nq = min(args.n_queries, xq.shape[0])
    k_default = 10

    print(f"\n{'nprobe':>8s}  {'recall@10':>10s}  {'mean_dist':>10s}  {'p50_dist':>10s}  {'p90_dist':>10s}  {'latency_ms':>10s}")
    print("-" * 70)

    all_distances = []
    best_recall = 0.0
    best_nprobe = 0

    for nprobe in nprobes:
        if hasattr(index, 'nprobe'):
            index.nprobe = nprobe

        recalls = []
        distances = []
        latencies = []

        for i in range(nq):
            v = xq[i].reshape(1, -1).astype(np.float32)
            t0 = time.perf_counter()
            D, I = index.search(v, k_default)
            latencies.append(time.perf_counter() - t0)
            if gt is not None:
                r = recall_at_k(I[0], gt[i], k_default)
                recalls.append(r)
            distances.extend(D[0].tolist())

        avg_recall = np.mean(recalls) if recalls else float('nan')
        dists = np.array(distances)
        mean_d = dists.mean()
        p50_d = np.percentile(dists, 50)
        p90_d = np.percentile(dists, 90)
        avg_lat = np.mean(latencies) * 1000

        print(f"{nprobe:8d}  {avg_recall:10.4f}  {mean_d:10.1f}  {p50_d:10.1f}  {p90_d:10.1f}  {avg_lat:10.3f}")

        if avg_recall > best_recall:
            best_recall = avg_recall
            best_nprobe = nprobe

        if nprobe == 32:  # default nprobe, collect for anchor
            all_distances = dists

    print(f"\nBest recall@10: {best_recall:.4f} at nprobe={best_nprobe}")

    # Anchor calibration
    if len(all_distances) > 0:
        p10 = np.percentile(all_distances, 10)
        p90 = np.percentile(all_distances, 90)
        p99 = np.percentile(all_distances, 99)
        print(f"\nDistance distribution at nprobe=32 ({len(all_distances)} samples):")
        print(f"  min={all_distances.min():.1f}  p10={p10:.1f}  p50={np.percentile(all_distances,50):.1f}  p90={p90:.1f}  p99={p99:.1f}  max={all_distances.max():.1f}")
        print(f"\n  Suggested buyer anchors for {ds['name']}:")
        print(f"    best_dist_anchor: {p10:.1f}        # p10: 'good' results")
        print(f"    worst_dist_anchor: {p99:.1f}        # p99: 'bad' results")

    # What does it take to get 0.95 recall?
    if best_recall < 0.95:
        if best_nprobe >= nlist:
            print(f"\n  max recall = {best_recall:.4f} at nprobe={best_nprobe} (nlist, full search)")
            print(f"  Gap from 0.95: {(0.95-best_recall)*100:.1f} percentage points — this is PQ compression loss, NOT nprobe limit.")
            print(f"  To reach 0.95: rebuild index with less compression (larger m, more nbits, or use IndexIVFFlat).")
        else:
            print(f"\n  max recall = {best_recall:.4f} at nprobe={best_nprobe} (nlist={nlist})")
            print(f"  Still has headroom — higher nprobe might help, but PQ compression is the ceiling.")
            print(f"  Practical: set theta_recall ≤ {best_recall:.2f} for QualityBuyer.")


if __name__ == "__main__":
    main()
