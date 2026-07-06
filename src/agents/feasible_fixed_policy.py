"""
FixedPolicy with feasible-action filtering (must-serve marketplace).
"""
import random
from typing import Tuple

from src.pricing.feasible_actions import (
    fallback_action_indices,
    list_feasible_actions,
    to_action_index,
)
from src.system.types import Query, Action


class FeasibleFixedPolicy:
    """Fixed default package, restricted to cost-feasible actions."""

    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        cost_model: dict,
        default_z_index: int = 2,
        default_p_index: int = 2,
        epsilon: float = 0.1,
        seed: int = 42,
        min_margin: float = 0.0,
    ):
        self.configs = list(search_param_configs)
        self.prices = list(price_tiers)
        self.cost_model = cost_model
        self.default_z = default_z_index
        self.default_p = default_p_index
        self.epsilon = epsilon
        self.min_margin = min_margin
        self.rng = random.Random(seed)
        self.version = f"feasible-fixed-eps{epsilon}"
        self.n_actions = len(self.configs) * len(self.prices)

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        feasible = list_feasible_actions(
            query, self.configs, self.prices, self.cost_model, min_margin=self.min_margin
        )
        if not feasible:
            z_idx, p_idx = fallback_action_indices(
                query, self.configs, self.prices, self.cost_model
            )
            action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
            return action, 1.0 / self.n_actions, self.version

        feasible_set = set(feasible)
        default_pair = (self.default_z, self.default_p)
        if default_pair not in feasible_set:
            default_pair = min(
                feasible,
                key=lambda zp: (
                    abs(zp[0] - self.default_z),
                    abs(zp[1] - self.default_p),
                ),
            )

        explore = self.rng.random() < self.epsilon
        if explore:
            z_idx, p_idx = self.rng.choice(feasible)
            propensity = self.epsilon / len(feasible)
            if (z_idx, p_idx) == default_pair:
                propensity += (1.0 - self.epsilon)
        else:
            z_idx, p_idx = default_pair
            propensity = (1.0 - self.epsilon) + self.epsilon / len(feasible)

        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, float(propensity), self.version
