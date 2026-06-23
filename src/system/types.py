"""
Core data structures used across all agents.
Keep these as plain dataclasses for transparency and easy serialization.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import numpy as np


@dataclass
class Query:
    """A buyer's ANN query."""
    id: str
    v_t: np.ndarray            # query vector, shape (dim,)
    k_t: int                   # number of neighbors requested
    filter_t: dict             # optional metadata filter
    sla_t: float               # latency budget in seconds (e.g. 0.050 = 50ms)
    budget_t: float            # buyer's price ceiling in USD

    def to_dict(self):
        d = asdict(self)
        d["v_t"] = self.v_t.tolist() if isinstance(self.v_t, np.ndarray) else self.v_t
        return d


@dataclass
class Action:
    """Platform's joint decision: search params + price."""
    z_t: dict                  # e.g. {"nprobe": 32, "rerank_k": 200, "early_stop": False}
    p_t: float                 # quoted price in USD


@dataclass
class Outcome:
    """Result of executing an action against an index, plus buyer response."""
    results: list              # list of (neighbor_id, distance) tuples
    L_t: float                 # observed latency (seconds)
    C_t: float                 # platform's cost for this query (USD)
    Q_t: Optional[float]       # ground-truth recall@k (None unless shadow-sampled)
    A_t: bool                  # buyer accepted?
    S_t: Optional[float]       # buyer satisfaction score (None if not measured)
    R_t: float                 # platform revenue = (p_t - C_t) if A_t else -C_t
    sla_violated: bool = False # query latency exceeded SLA constraint


@dataclass
class Trajectory:
    """One complete query lifecycle. This is what gets logged to disk."""
    query: Query
    U_t: float                 # difficulty estimate (observed confounder)
    h_t: dict                  # context features
    action: Action
    propensity: float          # pi(a|s) at decision time — CRITICAL for off-policy learning
    policy_version: str
    outcome: Outcome
    timestamp: float           # unix seconds, decision time

    def flatten_for_log(self) -> dict:
        """Flatten into a dict suitable for parquet."""
        return {
            "query_id": self.query.id,
            "k_t": self.query.k_t,
            "sla_t": self.query.sla_t,
            "budget_t": self.query.budget_t,
            "filter_t": str(self.query.filter_t),
            "U_t": self.U_t,
            "h_t": str(self.h_t),
            "z_nprobe": self.action.z_t.get("nprobe"),
            "z_rerank_k": self.action.z_t.get("rerank_k"),
            "z_early_stop": self.action.z_t.get("early_stop"),
            "p_t": self.action.p_t,
            "propensity": self.propensity,
            "policy_version": self.policy_version,
            "L_t": self.outcome.L_t,
            "C_t": self.outcome.C_t,
            "Q_t": self.outcome.Q_t,
            "A_t": self.outcome.A_t,
            "S_t": self.outcome.S_t,
            "R_t": self.outcome.R_t,
            "sla_violated": self.outcome.sla_violated,
            "timestamp": self.timestamp,
        }
