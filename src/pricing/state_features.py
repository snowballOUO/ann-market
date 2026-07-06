"""
Unified policy state features for heterogeneous pricing experiments.

Extends the original 6-dim LinUCB state with:
  - estimated query cost (pre-decision)
  - buyer persona one-hot (enterprise, budget)
"""
from __future__ import annotations

import numpy as np

from src.data.tiered_workload import estimate_service_cost
from src.system.types import Query

PERSONA_ENTERPRISE = "enterprise"
PERSONA_BUDGET = "budget"
PERSONA_QUALITY = "quality"

# Reference nprobe by difficulty bucket for cost estimation
BUCKET_REF_NPROBE = {
    "easy": 8,
    "medium": 32,
    "hard": 128,
}


def estimate_query_cost(
    query: Query,
    U_t: float,
    cost_model: dict,
) -> float:
    """Pre-decision cost estimate from difficulty bucket + U_t."""
    bucket = query.difficulty_bucket or ("hard" if U_t >= 0.6 else "easy")
    ref_nprobe = BUCKET_REF_NPROBE.get(bucket, 32)
    base = estimate_service_cost(ref_nprobe, cost_model)
    k_mult = 1.0 + 0.2 * (query.k_t / 100.0)
    u_mult = 1.0 + 0.3 * float(U_t)
    return base * k_mult * u_mult


def build_hetero_state(
    query: Query,
    U_t: float,
    h_t: dict,
    cost_model: dict,
    *,
    sentiment: float = 0.8,
) -> np.ndarray:
    """
    10-dim state for HeteroLinUCB:
      [U_t×100, accept_rate, latency_ms, k/100, sla_ms, budget_ms,
       sentiment, est_cost×1000, is_enterprise, is_budget]
    """
    est_cost = estimate_query_cost(query, U_t, cost_model)
    persona = (query.persona_t or "").lower()
    return np.array(
        [
            U_t * 100.0,
            h_t.get("recent_accept_rate", 0.5),
            h_t.get("recent_mean_latency", 0.0) * 1000.0,
            query.k_t / 100.0,
            query.sla_t * 1000.0,
            query.budget_t * 1000.0,
            sentiment,
            est_cost * 1000.0,
            1.0 if persona == PERSONA_ENTERPRISE else 0.0,
            1.0 if persona == PERSONA_BUDGET else 0.0,
        ],
        dtype=np.float64,
    )


def build_legacy_state(query: Query, U_t: float, h_t: dict) -> np.ndarray:
    """Original 6-dim LinUCB state (backward compatible)."""
    return np.array(
        [
            U_t * 100.0,
            h_t.get("recent_accept_rate", 0.5),
            h_t.get("recent_mean_latency", 0.0) * 1000.0,
            query.k_t / 100.0,
            query.sla_t * 1000.0,
            query.budget_t * 1000.0,
        ],
        dtype=np.float64,
    )
