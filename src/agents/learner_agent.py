"""
Agent 5: LearnerAgent

Week 1: STUB. Just reads trajectories and prints summary stats.
Week 5+: implements Causal DR-Bellman update.

The reason this exists in Week 1 is to lock down the I/O contract with
the log store. Once you can read trajectories back from parquet correctly,
the actual learning algorithm is a contained change.
"""
import glob
import os
import pyarrow.parquet as pq
import pandas as pd
from typing import Optional


class LearnerAgent:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def read_trajectories(self) -> pd.DataFrame:
        """Concatenate all parquet shards into one DataFrame."""
        files = sorted(glob.glob(os.path.join(self.log_dir, "*.parquet")))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def summary(self) -> Optional[dict]:
        df = self.read_trajectories()
        if df.empty:
            return None
        return {
            "n_trajectories": len(df),
            "n_shadow_sampled": df["Q_t"].notna().sum(),
            "mean_recall_when_sampled": df["Q_t"].dropna().mean() if df["Q_t"].notna().any() else None,
            "mean_latency_ms": df["L_t"].mean() * 1000,
            "p99_latency_ms": df["L_t"].quantile(0.99) * 1000,
            "accept_rate": df["A_t"].mean(),
            "sla_violation_rate": df["sla_violated"].mean() if "sla_violated" in df.columns else None,
            "total_revenue": df["R_t"].sum(),
            "mean_propensity": df["propensity"].mean(),
            "min_propensity": df["propensity"].min(),
            "policy_versions": df["policy_version"].unique().tolist(),
        }

    def periodic_update(self):
        """Placeholder for Week 5. For now, just print stats."""
        s = self.summary()
        if s is None:
            print("No trajectories logged yet.")
            return
        print("=" * 60)
        print("Learner summary")
        print("=" * 60)
        for k, v in s.items():
            print(f"  {k}: {v}")
