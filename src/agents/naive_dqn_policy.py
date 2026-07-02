"""
B3: Naive DQN Policy (No Deconfounding).

Same architecture as QLearningPolicy, but state excludes U_t.
Tests whether deconfounding matters for offline learning.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from src.system.types import Query, Action


class NaiveDQNPolicy:
    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        model_path: str = "models/qnet_naive_dqn_v1.pt",
        temperature: float = 0.1,
    ):
        self.configs = search_param_configs
        self.prices = price_tiers
        self.n_actions = len(self.configs) * len(self.prices)
        self.temperature = temperature
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.version = "naive-dqn-v1"

    def _extract_state(self, query: Query, U_t: float, h_t: dict) -> torch.Tensor:
        """7-dim state with U_t replaced by 0, using same normalisation as QLearningPolicy."""
        from src.causal.dr_estimator import make_qnet_state_features
        sentiment = h_t.get("market_sentiment", 0.8)
        feats = make_qnet_state_features(query, U_t, h_t, use_u_t=False, sentiment=sentiment)
        return torch.from_numpy(feats.reshape(1, -1).astype(np.float32))

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        state_tensor = self._extract_state(query, U_t, h_t)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
        probs = F.softmax(q_values / self.temperature, dim=0).numpy()
        probs = np.clip(probs, a_min=1e-5, a_max=1.0)
        probs = probs / probs.sum()
        action_idx = np.random.choice(self.n_actions, p=probs)
        propensity = float(probs[action_idx])
        z_idx = action_idx // len(self.prices)
        p_idx = action_idx % len(self.prices)
        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        return action, propensity, self.version
