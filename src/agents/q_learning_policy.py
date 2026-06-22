import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from src.system.types import Query, Action

class QLearningPolicy:
    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        model_path: str = "models/qnet_distilled_v1.pt",
        temperature: float = 0.1
    ):
        self.configs = search_param_configs
        self.prices = price_tiers
        self.n_actions = len(self.configs) * len(self.prices)
        self.temperature = temperature
        
        # 加载蒸馏后的轻量级 TorchScript 模型，极速推理
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.version = "qlearning-dr-v1"

    def _extract_state(self, query: Query, U_t: float, h_t: dict) -> torch.Tensor:
        """
        构造状态向量。必须包含 U_t 以完成混淆消除 (Deconfounding)。
        """
        # 解析复杂的标量过滤条件。
        # 如果涉及类似 ACORN 数据集的过滤字段，需将 token 转换为实际的 pct 浮点数比例。
        filter_ratio = 1.0
        if query.filter_t:
            raw_val = query.filter_t.get("filter_ratio_token", "")
            if isinstance(raw_val, str) and "pct" in raw_val:
                filter_ratio = float(raw_val.replace("p", ".").replace("pct", ""))

        state_features = [
            U_t,                        # [极其关键] W4 训练的精确难度评估
            float(np.linalg.norm(query.v_t)),
            query.k_t / 100.0,
            query.sla_t,
            query.budget_t,
            filter_ratio,
            h_t.get("recent_accept_rate", 0.5),
            h_t.get("recent_mean_latency", 0.05)
        ]
        return torch.tensor([state_features], dtype=torch.float32)

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
        action_idx = np.random.choice(self.n_actions, p=probs)
        propensity = float(probs[action_idx])

        # 解码动作
        z_idx = action_idx // len(self.prices)
        p_idx = action_idx % len(self.prices)
        
        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        
        return action, propensity, self.version
