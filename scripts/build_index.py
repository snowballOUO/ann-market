"""
Build an IVF-PQ index on the SIFT1M base set and persist it to disk.

Usage:
    python scripts/build_index.py --config configs/base.yaml

Output:
    data/sift1m/index_ivfpq.faiss
"""
import argparse
import os
import time
import yaml
import numpy as np
import faiss

from src.data.datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--output", default=None, help="output index path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_dir = cfg["dataset"]["path"]
    out_path = args.output or os.path.join(data_dir, "index_ivfpq.faiss")

    name = cfg["dataset"]["name"]
    print(f"Loading {name} from {data_dir}...")
    xb, xq, xt, gt = load_dataset(cfg)  # max_base + normalize handled in loader
    print(f"  base:    {xb.shape}")
    print(f"  query:   {xq.shape}")
    print(f"  train:   {xt.shape}")

    dim = cfg["dataset"]["dim"]
    nlist = cfg["index"]["nlist"]
    m = cfg["index"]["m"]
    nbits = cfg["index"]["nbits"]
    assert xb.shape[1] == dim, f"dim mismatch: {xb.shape[1]} vs {dim}"

    print(f"\nBuilding IVF-PQ index: nlist={nlist}, m={m}, nbits={nbits}")
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    t0 = time.time()
    print(f"  training on {xt.shape[0]} vectors...")
    index.train(xt)
    print(f"  training took {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"  adding {xb.shape[0]} vectors...")
    index.add(xb)
    print(f"  add took {time.time() - t0:.1f}s")

    faiss.write_index(index, out_path)
    print(f"\nWrote index to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # Quick sanity check (skip out-of-range gt IDs from max_base truncation)
    print("\nSanity check with nprobe=16, 10 queries:")
    index.nprobe = 16
    n_check = min(100, xq.shape[0])
    recalls = []
    for i in range(n_check):
        D, I = index.search(xq[i:i+1], 10)
        gt_i = gt[i]
        valid = gt_i[gt_i >= 0]
        if len(valid) > 0:
            overlap = len(set(I[0].tolist()) & set(valid[:10].tolist()))
            recalls.append(overlap / min(10, len(valid)))
    if recalls:
        print(f"  mean recall@10: {np.mean(recalls):.3f} (over {len(recalls)} valid queries)")
    else:
        print(f"  (no valid ground truth — all neighbors outside subset)")


if __name__ == "__main__":
    main()
