"""
Feasible action filtering for must-serve marketplace.

Rules (成本内必交付):
  1. est_cost(nprobe) <= budget_t     — 算力成本在用户预算包络内
  2. p_t >= est_cost                  — 报价不低于预估成本（不亏本卖）
"""
from __future__ import annotations

import numpy as np

from src.data.tiered_workload import estimate_service_cost
from src.system.types import Query


def estimate_action_cost(
    nprobe: int,
    cost_model: dict,
    *,
    k_t: int = 10,
    cost_slack: float = 1.0,
) -> float:
    base = estimate_service_cost(nprobe, cost_model)
    k_mult = 1.0 + 0.1 * (k_t / 100.0)
    return base * k_mult * cost_slack


def is_action_feasible(
    query: Query,
    z_idx: int,
    p_idx: int,
    configs: list[dict],
    prices: list[float],
    cost_model: dict,
    *,
    min_margin: float = 0.0,
) -> bool:
    nprobe = configs[z_idx].get("nprobe", 32)
    est_c = estimate_action_cost(nprobe, cost_model, k_t=query.k_t)
    p = prices[p_idx]
    if est_c > query.budget_t:
        return False
    if p < est_c * (1.0 + min_margin):
        return False
    return True


def list_feasible_actions(
    query: Query,
    configs: list[dict],
    prices: list[float],
    cost_model: dict,
    *,
    min_margin: float = 0.0,
) -> list[tuple[int, int]]:
    """Return list of (z_idx, p_idx) feasible packages."""
    out: list[tuple[int, int]] = []
    for z_idx in range(len(configs)):
        for p_idx in range(len(prices)):
            if is_action_feasible(
                query, z_idx, p_idx, configs, prices, cost_model, min_margin=min_margin
            ):
                out.append((z_idx, p_idx))
    return out


def fallback_action_indices(
    query: Query,
    configs: list[dict],
    prices: list[float],
    cost_model: dict,
) -> tuple[int, int]:
    """Cheapest feasible delivery: min est_cost, then min price >= cost."""
    best: tuple[int, int] | None = None
    best_cost = float("inf")
    best_price = float("inf")
    for z_idx, cfg in enumerate(configs):
        est_c = estimate_action_cost(cfg.get("nprobe", 32), cost_model, k_t=query.k_t)
        for p_idx, p in enumerate(prices):
            if p < est_c:
                continue
            if est_c < best_cost or (est_c == best_cost and p < best_price):
                best = (z_idx, p_idx)
                best_cost = est_c
                best_price = p
    if best is not None:
        return best
    return 0, 0


def to_action_index(z_idx: int, p_idx: int, n_prices: int) -> int:
    return z_idx * n_prices + p_idx


def mask_to_probs(
    scores: np.ndarray,
    feasible_indices: list[int],
    temperature: float,
    rng,
) -> tuple[int, float]:
    """Softmax over feasible actions only; uniform fallback if empty."""
    if not feasible_indices:
        return 0, 1.0 / len(scores)
    sub = scores[feasible_indices]
    sub = sub - sub.max()
    exp_s = np.exp(sub / max(temperature, 1e-6))
    probs = exp_s / exp_s.sum()
    local_idx = int(rng.choice(len(feasible_indices), p=probs))
    action_idx = feasible_indices[local_idx]
    return action_idx, float(probs[local_idx])
