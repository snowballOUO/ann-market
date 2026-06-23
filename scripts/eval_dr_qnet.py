"""
W6 DR offline evaluation: LinUCB vs Distilled Q-Net.

Uses LinUCB logs as behavior policy data, trains a reward model Q̂,
then computes V̂_DR for both LinUCB and Q-Net policies.

Usage:
    python scripts/eval_dr_qnet.py --log-dir logs/run_1782182484 --qnet-model models/qnet_distilled_v1.pt
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from src.causal.dr_estimator import (
    load_logs, build_state, build_action_index,
    RewardModel, dr_estimate,
)
from src.agents.bandit_policy import LinUCBPolicy
from src.agents.q_learning_policy import QLearningPolicy


class LinUCBPolicyEvaluator:
    """Wrap LinUCB for offline evaluation: compute probs for any state without side effects."""

    def __init__(self, policy: LinUCBPolicy):
        self.policy = policy
        self.n_actions = policy.n_actions
        self.temperature = policy.temperature

    def action_probs(self, s: np.ndarray) -> np.ndarray:
        """Given state s (6,), return softmax probs over 25 actions."""
        ucbs = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.policy.A[a])
            theta = A_inv @ self.policy.b[a]
            point_est = float(theta @ s)
            bonus = self.policy.alpha * np.sqrt(float(s @ A_inv @ s))
            ucbs[a] = point_est + bonus
        ucbs_shifted = ucbs - ucbs.max()
        exp_ucbs = np.exp(ucbs_shifted / self.temperature)
        probs = exp_ucbs / exp_ucbs.sum()
        return probs.astype(np.float64)

    def get_action_probs_batch(self, S: np.ndarray) -> np.ndarray:
        """Batch version: (N, 6) → (N, 25)."""
        probs = np.zeros((len(S), self.n_actions), dtype=np.float64)
        for i in range(len(S)):
            probs[i] = self.action_probs(S[i])
        return probs


class QNetPolicyEvaluator:
    """Wrap Q-net for offline evaluation: compute Boltzmann probs for any state."""

    def __init__(self, model_path: str, temperature: float = 0.1):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.temperature = temperature
        self.n_actions = 25

    def get_action_probs_batch(self, S: np.ndarray) -> np.ndarray:
        """Given states (N, 6), return Boltzmann probs (N, 25)."""
        S_tensor = torch.tensor(S, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(S_tensor)  # (N, 25)
        probs = F.softmax(q_values / self.temperature, dim=1).numpy()
        probs = np.clip(probs, 1e-5, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="LinUCB behavior log directory")
    ap.add_argument("--qnet-model", default="models/qnet_distilled_v1.pt")
    ap.add_argument("--clip-max", type=float, default=20.0)
    ap.add_argument("--linucb-temp", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.exists(args.qnet_model):
        print(f"ERROR: Q-net model not found at {args.qnet_model}")
        print("Run: python scripts/train_qnet.py first")
        sys.exit(1)

    # ── 1. Load behavior logs (LinUCB) ──
    print(f"Loading behavior logs from {args.log_dir}...")
    df = load_logs(args.log_dir)
    S = build_state(df)
    A_log = build_action_index(df)
    R = df["R_t"].values
    pi_b = df["propensity"].values
    print(f"  {len(df)} trajectories")

    for name, val in [("S", S), ("A_log", A_log), ("R", R), ("pi_b", pi_b)]:
        if np.any(np.isnan(val)):
            raise ValueError(f"NaN detected in {name}")

    # ── 2. Train reward model Q̂ on behavior data ──
    print("Training reward model Q̂...")
    q_hat = RewardModel().fit(S, A_log, R)
    preds = q_hat.predict(S, A_log)
    print(f"  Q̂ train R² ≈ {np.corrcoef(preds, R)[0, 1]**2:.4f}")

    # ── 3. Build LinUCB evaluator from the trained policy ──
    print("Building LinUCB policy for offline evaluation...")
    linucb = LinUCBPolicy(
        search_param_configs=[
            {"nprobe": 8,  "rerank_k": 50,  "early_stop": False},
            {"nprobe": 16, "rerank_k": 100, "early_stop": False},
            {"nprobe": 32, "rerank_k": 200, "early_stop": False},
            {"nprobe": 64, "rerank_k": 400, "early_stop": False},
            {"nprobe": 128,"rerank_k": 800, "early_stop": False},
        ],
        price_tiers=[0.001, 0.002, 0.005, 0.01, 0.02],
        alpha=1.0, temperature=args.linucb_temp, seed=42,
    )
    # Warm up LinUCB by replaying logs (reconstruct its belief state)
    for i in range(len(S)):
        linucb._last_s = S[i].astype(np.float64)
        linucb._last_action_idx = int(A_log[i])
        linucb.update(R[i])

    linucb_eval = LinUCBPolicyEvaluator(linucb)

    # ── 4. Build Q-Net evaluator ──
    print(f"Loading Q-Net from {args.qnet_model}...")
    qnet_eval = QNetPolicyEvaluator(args.qnet_model, temperature=0.1)

    # ── 5. Compute π_new for both policies ──
    print("Computing π_new for LinUCB...")
    pi_linucb = linucb_eval.get_action_probs_batch(S)

    print("Computing π_new for Q-Net...")
    pi_qnet = qnet_eval.get_action_probs_batch(S)

    # ── 6. DR estimates ──
    print(f"Computing DR estimates (clip_max={args.clip_max})...")
    dr_linucb = dr_estimate(S, A_log, R, pi_b, pi_linucb, q_hat, clip_max=args.clip_max)
    dr_qnet   = dr_estimate(S, A_log, R, pi_b, pi_qnet,   q_hat, clip_max=args.clip_max)

    V_linucb = dr_linucb["V_dr"]
    V_qnet   = dr_qnet["V_dr"]

    # If V values are negative (costs exceed revenue per query), use absolute threshold
    delta = V_qnet - V_linucb
    abs_denom = max(abs(V_linucb), 1e-12)
    improvement_pct = delta / abs_denom * 100

    # ── 7. Report ──
    print()
    print("=" * 60)
    print("W6 DR Offline Evaluation")
    print("=" * 60)
    print(f"  Behavior log:        {args.log_dir}")
    print(f"  N trajectories:      {len(df)}")
    print(f"  Clipping M:          {args.clip_max}")
    print()
    print(f"  {'':20s} {'V̂_DR':>10s} {'V̂_DM':>10s} {'V̂_IPS':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'LinUCB (behavior)':20s} {V_linucb:>10.6f} {dr_linucb['V_dm']:>10.6f} {dr_linucb['V_ips']:>10.6f}")
    print(f"  {'Q-Net (new)':20s} {V_qnet:>10.6f} {dr_qnet['V_dm']:>10.6f} {dr_qnet['V_ips']:>10.6f}")
    print()
    print(f"  Δ = {delta:+.6f}  ({improvement_pct:+.1f}%)")
    print(f"  Threshold: ≥ 10%")
    print(f"  Result:    {'✅ PASS' if improvement_pct >= 10 else '❌ FAIL'}")

    # ── Diagnostics ──
    print()
    print("Importance weight diagnostics:")
    for name, dr in [("LinUCB", dr_linucb), ("Q-Net", dr_qnet)]:
        print(f"  {name:<10s}  mean ρ={dr['mean_rho']:.2f}  p99 ρ={dr['rho_p99']:.1f}")


if __name__ == "__main__":
    main()
