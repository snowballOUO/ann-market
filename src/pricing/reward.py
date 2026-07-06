"""
Reward shaping for heterogeneous marketplace experiments.

Modes:
  hard                  — (p-c) if accept else -c
  satisfaction          — S_t-weighted margin with soft reject penalty
  satisfaction_retention — adds streak bonus and churn penalty
"""
from __future__ import annotations


def compute_reward(
    price: float,
    cost: float,
    accepted: bool,
    satisfaction: float,
    *,
    mode: str = "hard",
    accept_streak: int = 0,
    market_sentiment: float = 0.8,
    reject_cost_weight: float = 0.5,
    streak_bonus: float = 0.001,
    churn_penalty: float = 0.002,
    sentiment_churn_threshold: float = 0.4,
) -> float:
    margin = price - cost
    s = float(satisfaction) if satisfaction is not None else (1.0 if accepted else 0.0)

    if mode == "hard":
        return margin if accepted else (-cost)

    if mode == "satisfaction":
        return s * margin - (1.0 - s) * cost * reject_cost_weight

    if mode == "satisfaction_retention":
        base = s * margin - (1.0 - s) * cost * reject_cost_weight
        bonus = streak_bonus if accept_streak >= 3 else 0.0
        penalty = churn_penalty if (not accepted and market_sentiment < sentiment_churn_threshold) else 0.0
        return base + bonus - penalty

    raise ValueError(f"Unknown reward mode: {mode}")
