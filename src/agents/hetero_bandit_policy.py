"""
HeteroLinUCB — contextual bandit with 10-dim heterogeneous state.

Action space: (nprobe tier, price tier) package — 5×5 = 25 combos.
Learns to pair low nprobe + low price for budget personas and
high nprobe + high price for enterprise personas.
"""
import math
import numpy as np
from typing import Tuple

from src.pricing.feasible_actions import (
    fallback_action_indices,
    list_feasible_actions,
    to_action_index,
)
from src.pricing.state_features import build_hetero_state
from src.system.types import Query, Action


class HeteroLinUCBPolicy:
    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        cost_model: dict,
        alpha: float = 0.5,
        temperature: float = 0.4,
        min_margin: float = 0.0,
        seed: int = 42,
    ):
        self.configs = list(search_param_configs)
        self.prices = list(price_tiers)
        self.cost_model = cost_model
        self.min_margin = min_margin
        self.n_z = len(self.configs)
        self.n_p = len(self.prices)
        self.n_actions = self.n_z * self.n_p
        self.alpha = alpha
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.version = f"hetero-linucb-a{alpha}-t{int(temperature * 100)}"
        self.d = 10
        self.A = [np.eye(self.d) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.d) for _ in range(self.n_actions)]
        self.counts = np.zeros(self.n_actions, dtype=int)
        self._last_s = None
        self._last_action_idx = None

    def _build_features(self, query: Query, U_t: float, h_t: dict) -> np.ndarray:
        sentiment = h_t.get("market_sentiment", 0.8)
        return build_hetero_state(query, U_t, h_t, self.cost_model, sentiment=sentiment)

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        s = self._build_features(query, U_t, h_t)
        feasible_pairs = list_feasible_actions(
            query, self.configs, self.prices, self.cost_model, min_margin=self.min_margin
        )
        if feasible_pairs:
            feasible_idx = [
                to_action_index(z, p, self.n_p) for z, p in feasible_pairs
            ]
        else:
            z_fb, p_fb = fallback_action_indices(
                query, self.configs, self.prices, self.cost_model
            )
            feasible_idx = [to_action_index(z_fb, p_fb, self.n_p)]

        ucbs = np.full(self.n_actions, -1e9)
        for a in feasible_idx:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            point_est = float(theta @ s)
            explore_bonus = self.alpha * np.sqrt(float(s @ A_inv @ s))
            ucbs[a] = point_est + explore_bonus

        sub_ucbs = ucbs[feasible_idx]
        sub_shifted = sub_ucbs - sub_ucbs.max()
        exp_ucbs = np.exp(sub_shifted / self.temperature)
        probs_sub = exp_ucbs / exp_ucbs.sum()

        local = int(self.rng.choice(len(feasible_idx), p=probs_sub))
        action_idx = feasible_idx[local]
        propensity = float(probs_sub[local])
        self._last_s = s
        self._last_action_idx = action_idx

        z_idx = action_idx // self.n_p
        p_idx = action_idx % self.n_p
        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, propensity, self.version

    def update(self, reward: float):
        a = self._last_action_idx
        s = self._last_s
        if a is None or s is None:
            return
        self.A[a] += np.outer(s, s)
        self.b[a] += reward * s
        self.counts[a] += 1

    def action_counts(self) -> np.ndarray:
        return self.counts.copy()

    def mean_theta_norm(self) -> float:
        norms = []
        for a in range(self.n_actions):
            if self.counts[a] > 0:
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                norms.append(float(np.linalg.norm(theta)))
        return float(np.mean(norms)) if norms else 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(min(x, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-x))
