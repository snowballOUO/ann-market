import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from src.causal.dr_estimator import make_qnet_state_features
from src.system.types import Query, Action

class QLearningPolicy:
    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        model_path: str = "models/qnet_distilled_v1.pt",
        temperature: float = 0.1,
        use_u_t: bool = True,
        seed: int = 42,
    ):
        self.configs = search_param_configs
        self.prices = price_tiers
        self.n_actions = len(self.configs) * len(self.prices)
        self.temperature = temperature
        self.use_u_t = use_u_t
        self.rng = np.random.default_rng(seed)
        
        # 加载蒸馏后的轻量级 TorchScript 模型，极速推理
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.version = f"qlearning-stable-v1-t{temperature:g}"

    def _extract_state(self, query: Query, U_t: float, h_t: dict) -> torch.Tensor:
        """Construct the normalized QNet state used during offline training."""
        state_features = make_qnet_state_features(query, U_t, h_t, use_u_t=self.use_u_t)
        return torch.from_numpy(state_features.reshape(1, -1).astype(np.float32))

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        """
        使用 Boltzmann Sampling 选择动作，保证 Propensity > 0 的致命不变量。
        """
        state_tensor = self._extract_state(query, U_t, h_t)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
            
        # 波尔兹曼分布提取概率 (Temperature Trick)
        probs = F.softmax(q_values / self.temperature, dim=0).numpy()
        
        # 强制概率下限 (防止浮点精度导致的纯 0，保护下游 IS 权重)
        probs = np.clip(probs, a_min=1e-5, a_max=1.0)
        probs = probs / probs.sum()

        # 根据 Q 值产生的概率分布进行采样
        action_idx = self.rng.choice(self.n_actions, p=probs)
        propensity = float(probs[action_idx])

        # 解码动作
        z_idx = action_idx // len(self.prices)
        p_idx = action_idx % len(self.prices)
        
        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        
        return action, propensity, self.version
