"""
Main experiment runner (Week 7).

Pipeline per dataset:
  1. LinUCB 20K → behavior logs
  2. Train Q-Net → models/{ds}/qnet_distilled_v1.pt
  3. Train Naive DQN → models/{ds}/qnet_naive_dqn_v1.pt
  4. For each seed: evaluate all 6 policies → collect metrics

Usage:
    python scripts/run_main_experiment.py                    # all datasets, all seeds
    python scripts/run_main_experiment.py --dataset sift1m    # single dataset
    python scripts/run_main_experiment.py --skip-train        # skip LinUCB+training
"""
import argparse
import os
import sys
import subprocess
import time
import csv
import pandas as pd
import glob
import numpy as np

DATASETS = ["sift1m", "deep1m", "gist1m", "ag_news"]
SEEDS = [42, 123, 456, 789, 101112]
METHODS = ["fixed", "sla", "cost", "linucb", "naive_dqn", "qnet"]


def run(cmd: str, desc: str = "") -> int:
    """Run a shell command, print output. Returns exit code."""
    if desc:
        print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    print(f"  $ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}, {elapsed:.0f}s)")
    else:
        print(f"  OK ({elapsed:.0f}s)")
    return result.returncode


def get_latest_log_dir():
    dirs = sorted(glob.glob("logs/run_*"))
    return dirs[-1] if dirs else None


def evaluate_policies(config: str, seed: int, dataset: str,
                      qnet_model: str, naive_model: str) -> dict:
    """Run compare_bandit — output directly to terminal, read JSON result."""
    import json
    json_path = f"logs/_eval_{dataset}_seed{seed}.json"
    cmd = (
        f'python scripts/compare_bandit.py '
        f'--config {config} --n-queries 10000 --seed {seed} '
        f'--policies fixed,sla,cost,linucb,naive_dqn,qnet '
        f'--qnet-model {qnet_model} --naive-dqn-model {naive_model} '
        f'--no-plot --results-json {json_path}'
    )
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, shell=True, cwd=os.getcwd(), env=env)
    print(f"    (seed {seed} eval done in {time.time()-t0:.0f}s)")
    if os.path.exists(json_path):
        with open(json_path) as f:
            metrics = json.load(f)
        os.remove(json_path)
        return metrics
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Single dataset name")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip LinUCB + model training (use existing)")
    ap.add_argument("--output", default="reports/main_results.csv")
    args = ap.parse_args()

    ds_list = [args.dataset] if args.dataset else DATASETS
    if args.dataset and args.dataset not in DATASETS:
        print(f"Unknown dataset '{args.dataset}'. Choices: {DATASETS}")
        sys.exit(1)

    os.makedirs("reports", exist_ok=True)
    all_rows = []

    for ds in ds_list:
        config = f"configs/{ds}.yaml"
        model_dir = f"models/{ds}"
        os.makedirs(model_dir, exist_ok=True)
        qnet_model = f"{model_dir}/qnet_distilled_v1.pt"
        naive_model = f"{model_dir}/qnet_naive_dqn_v1.pt"

        if not args.skip_train:
            # Step 1: LinUCB behavior logs
            rc = run(
                f"python scripts/run_experiment.py --config {config} --policy linucb --n-queries 20000",
                f"[{ds}] Step 1: LinUCB behavior logs"
            )
            if rc != 0:
                print(f"  SKIPPING {ds} — LinUCB failed")
                continue

            log_dir = get_latest_log_dir()
            if not log_dir:
                print(f"  SKIPPING {ds} — no log dir found")
                continue
            print(f"  Log dir: {log_dir}")

            # Step 2: Train Q-Net
            rc = run(
                f"python scripts/train_qnet.py --log-dir {log_dir} --output {qnet_model}",
                f"[{ds}] Step 2: Train Q-Net"
            )
            if rc != 0:
                print(f"  Q-Net training failed, continuing with other methods")

            # Step 3: Train Naive DQN
            rc = run(
                f"python scripts/train_qnet.py --log-dir {log_dir} --no-ut --output {naive_model}",
                f"[{ds}] Step 3: Train Naive DQN"
            )
            if rc != 0:
                print(f"  Naive DQN training failed, continuing with other methods")

        # Step 4: Evaluate all seeds
        for seed in SEEDS:
            print(f"\n  [{ds}] Seed {seed} — evaluating all 6 policies...")
            metrics = evaluate_policies(config, seed, ds, qnet_model, naive_model)

            for method in METHODS:
                if method in metrics:
                    all_rows.append({
                        "dataset": ds,
                        "method": method,
                        "seed": seed,
                        "revenue": metrics[method]["revenue"],
                        "accept_rate": metrics[method]["accept_rate"],
                    })
                    print(f"    {method:12s}  revenue=${metrics[method]['revenue']:.4f}  accept={metrics[method]['accept_rate']:.3f}")

    # Save results (merge with existing)
    if all_rows:
        df = pd.DataFrame(all_rows)
        if os.path.exists(args.output):
            existing = pd.read_csv(args.output)
            # Deduplicate: remove old rows for the same (dataset, seed, method)
            existing = existing[~existing['dataset'].isin(df['dataset'].unique())]
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {args.output} ({len(df)} rows)")

        # Summary table: mean ± std per dataset/method
        print(f"\n{'='*60}")
        print("Summary (mean revenue ± std over {len(SEEDS)} seeds)")
        print(f"{'='*60}")
        summary = df.groupby(["dataset", "method"])["revenue"].agg(["mean", "std"]).round(4)
        print(summary.to_string())
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
