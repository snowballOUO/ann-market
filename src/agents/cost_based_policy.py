"""
B2: Cost-Based Pricing Policy.

Fixed nprobe, price = estimated_cost × (1 + margin).
Simple cost-plus model — baseline for economic optimization comparison.
"""
import numpy as np
from typing import Tuple
from src.system.types import Query, Action


class CostBasedPolicy:
    """Fixed nprobe, cost-plus pricing."""

    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        cost_model: dict,
        fixed_z_index: int = 2,     # nprobe=32
        margin: float = 0.5,        # 50% markup
        seed: int = 42,
    ):
        self.configs = list(search_param_configs)
        self.prices = list(price_tiers)
        self.z_idx = fixed_z_index
        self.margin = margin
        self.base_per_ms = cost_model["base_per_ms"]
        self.fixed_overhead = cost_model["fixed_overhead"]
        self.n_actions = len(self.configs) * len(self.prices)
        self.version = "cost-based-v1"

    def _estimate_cost(self, nprobe: int) -> float:
        """Estimate cost based on nprobe (rough latency model)."""
        est_lat = 0.0005 + 0.0001 * nprobe  # seconds
        est_lat_ms = est_lat * 1000
        return self.fixed_overhead + self.base_per_ms * est_lat_ms

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        z_idx = self.z_idx
        cost = self._estimate_cost(self.configs[z_idx]["nprobe"])
        target_price = cost * (1.0 + self.margin)

        # Snap to nearest price tier
        p_idx = min(range(len(self.prices)),
                    key=lambda i: abs(self.prices[i] - target_price))

        # Don't quote above buyer's budget
        while p_idx > 0 and self.prices[p_idx] > query.budget_t:
            p_idx -= 1

        action_idx = z_idx * len(self.prices) + p_idx
        probs = np.full(self.n_actions, 1e-5)
        probs[action_idx] = 1.0 - 1e-5 * (self.n_actions - 1)
        propensity = float(probs[action_idx])

        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, propensity, self.version
