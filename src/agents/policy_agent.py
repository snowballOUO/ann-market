"""
Agent 2: PolicyAgent

Week 1 version: a fixed policy with optional epsilon-exploration.
This is INTENTIONALLY DUMB. We need a working pipeline first.

In Week 3, this becomes a contextual bandit.
In Week 5, this becomes a Q-learner with DR correction.

CRITICAL invariant: every call to decide() must return a non-zero propensity.
A zero propensity means the action would never have been chosen by this policy,
which breaks importance weighting in off-policy learning.
"""
import random
import numpy as np
from typing import Tuple
from src.system.types import Query, Action


class FixedPolicy:
    """Always picks a fixed (z, p), with small epsilon-uniform exploration.

    This is the Week 1 placeholder. Its purpose is to generate a trajectory log
    with valid propensities, so downstream learning code can be developed.
    """

    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        default_z_index: int = 2,
        default_p_index: int = 2,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        self.configs = search_param_configs
        self.prices = price_tiers
        self.default_z = default_z_index
        self.default_p = default_p_index
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.version = f"fixed-v1-eps{epsilon}"

        self.n_actions = len(self.configs) * len(self.prices)

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        """
        Returns:
            action: Action to take
            propensity: pi(a|s) — probability that this policy would pick this action
            version: policy identifier
        """
        # epsilon-uniform mixture: with prob (1-eps) pick default, else uniform random
        explore = self.rng.random() < self.epsilon

        if explore:
            z_idx = self.rng.randrange(len(self.configs))
            p_idx = self.rng.randrange(len(self.prices))
            # propensity: this specific (z,p) under uniform exploration
            propensity_explore = 1.0 / self.n_actions
            # total propensity is the mixture probability of picking this action
            if z_idx == self.default_z and p_idx == self.default_p:
                propensity = (1 - self.epsilon) + self.epsilon * propensity_explore
            else:
                propensity = self.epsilon * propensity_explore
        else:
            z_idx = self.default_z
            p_idx = self.default_p
            propensity_explore = 1.0 / self.n_actions
            propensity = (1 - self.epsilon) + self.epsilon * propensity_explore

        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, propensity, self.version
