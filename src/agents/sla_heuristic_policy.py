"""
B1: SLA-only Heuristic Policy.

Selects the highest nprobe that fits within SLA budget,
uses a fixed mid-tier price. Returns valid propensity (> 0).

Paper baseline: static SLA-aware configuration without economic optimization.
"""
import numpy as np
from typing import Tuple
from src.system.types import Query, Action


class SLAHeuristicPolicy:
    """Pick max nprobe under SLA, fixed price."""

    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        default_price_idx: int = 2,    # $0.005
        seed: int = 42,
    ):
        self.configs = list(search_param_configs)
        self.prices = list(price_tiers)
        self.price_idx = default_price_idx
        self.n_actions = len(self.configs) * len(self.prices)
        self.version = "sla-heuristic-v1"

    def _latency_estimate(self, nprobe: int) -> float:
        """Rough latency est (seconds). Calibrated per nprobe level."""
        return 0.0005 + 0.0001 * nprobe  # ~0.5ms base + 0.1ms/nprobe

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        # Pick the highest nprobe whose estimated latency fits within SLA
        z_idx = 0
        for i, cfg in enumerate(self.configs):
            est_lat = self._latency_estimate(cfg["nprobe"])
            if est_lat <= query.sla_t * 0.5:  # safety margin
                z_idx = i

        p_idx = self.price_idx
        action_idx = z_idx * len(self.prices) + p_idx

        # Softmax-style propensity: most mass on chosen action, small rest
        probs = np.full(self.n_actions, 1e-5)
        probs[action_idx] = 1.0 - 1e-5 * (self.n_actions - 1)
        propensity = float(probs[action_idx])

        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, propensity, self.version
