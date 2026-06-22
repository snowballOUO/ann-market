"""
W3---- LinUCB 上下文决策agent
将固定策略替换为一种线性置信上界策略，该策略可以根据不同的查询情景学习选择做出不同的(z_t, p_t)
状态特征 s (6 dims):
    [U_t, recent_accept_rate, recent_mean_latency_ms, k_t/100, sla_ms, budget_ms]
行动空间：
    5 z-configs × 5 price-tiers = 25 discrete actions.
线性模型:
    θ_a = A_a^{-1} b_a                          (ridge regression solution)
    UCB_a = θ_a^T s + α · √(s^T A_a^{-1} s)    (point estimate + exploration bonus)
决策流程：
1. 对所有 25 个动作计算 UCB 值
2. Softmax（上界估计值/温度） → 概率分布
3. 分布中的示例动作
4. 返回（动作、softmax概率、策略版本）
在结果确定后进行的在线学习：
    A_a += outer(s, s)
    b_a += reward * s
"""

import numpy as np
from typing import Tuple
from src.system.types import Query, Action
import math

class LinUCBPolicy:
    def __init__(
            self,
            search_param_configs:list[dict],
            price_tiers:list[float],
            alpha:float=1.0,
            temperature:float=0.5,
            seed:int=42,
    ):
        #动作空间
        self.configs = list(search_param_configs)
        self.prices=list(price_tiers)
        self.n_z=len(self.configs)
        self.n_p=len(self.prices)
        self.n_actions=self.n_z*self.n_p
        #超参数
        self.alpha = alpha
        self.temperature = temperature
        #RNG
        self.rng=np.random.default_rng(seed)
        #策略标识
        self.version=f"linucb-a{alpha}-t{int(temperature*100)}"
        #Per-arm state
        self.d=6  #特征维度
        # A[a] 初始化 = I_d（L2 正则化 + 保证可逆）
        self.A=[np.eye(self.d) for _ in range(self.n_actions)]
        # b[a] 初始化 = 0 向量
        self.b=[np.zeros(self.d) for _ in range(self.n_actions)]
        # 每个动作被选了多少次（诊断用）
        self.counts=np.zeros(self.n_actions,dtype=int)
        # ── decide() 和 update() 之间的桥梁 ──
        self._last_s=None
        self._last_action_idx=None


    def _build_features(
            self,query:Query,U_t:float,h_t:dict
    ) -> np.ndarray:
        return np.array(
            [
                U_t,# [0] 查询难度 [0,1]
                h_t.get("recent_accept_rate", 0.5),# [1] 近期接受率 [0,1]
                h_t.get("recent_mean_latency", 0.0)*1000,# [2] 近期延迟 (ms)
                query.k_t/100.0,# [3] 请求 k [0.1,1.0]
                query.sla_t*1000,# [4] SLA (ms)
                query.budget_t*1000,# [5] 预算 (毫美元)
            ],
            dtype=np.float64,
        )

    def decide(
            self,query:Query,U_t:float,h_t:dict
    ) -> Tuple[Action,float,str]:
        """
                    Returns:
                    action:     chosen (z_t, p_t)
                    propensity: softmax probability of sampled action
                    version:    policy identifier string
        """
        s=self._build_features(query,U_t,h_t)
        # ── 步骤 1: 对每个动作算 UCB ──
        ucbs=np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv=np.linalg.inv(self.A[a])  # (6,6)
            theta=A_inv @ self.b[a]  # (6,)
            point_est=float(theta @ s)   # 点估计：θᵀs
            explore_bonus=(
                self.alpha * np.sqrt(float(s @ A_inv @ s))
            )   # 探索奖励
            ucbs[a]=explore_bonus+point_est

        # ── 步骤 2: softmax 转概率 ──
        ucbs_shifted=ucbs-ucbs.max()   # 防 exp 溢出
        exp_ucbs=np.exp(ucbs_shifted/self.temperature)
        probs=exp_ucbs/exp_ucbs.sum()

        # ── 步骤 3: 按概率采样 ──
        action_idx=int(self.rng.choice(self.n_actions,p=probs))
        propensity=float(probs[action_idx])

        # ── 步骤 4: 记实例状态（供 update() 用）──
        self._last_s=s
        self._last_action_idx=action_idx

        # ── 步骤 5: 动作索引 → (z_t, p_t) ──
        z_idx=action_idx//self.n_p
        p_idx=action_idx%self.n_p
        z_t=dict(self.configs[z_idx])
        p_t=self.prices[p_idx]

        action=Action(z_t=z_t, p_t=p_t)
        return action,propensity,self.version

    def update(self,reward:float):
        """
        Online ridge regression: incorporate the observed reward.

        A[a] += outer(s, s)
        b[a] += reward * s
        """
        a=self._last_action_idx
        s=self._last_s

        if a is None or s is None:
            return
        self.A[a] += np.outer(s,s)
        self.b[a] += reward * s
        self.counts[a] += 1

    def set_exploration(self,alpha:float=None,temperature:float=None):
        """Dynamically adjust exploration during training or ablation."""
        if alpha is not None:
            self.alpha = alpha
        if temperature is not None:
            self.temperature = temperature

    def action_counts(self) -> np.ndarray:
        """How many times each action was selected. For diagnostics."""
        return  self.counts.copy()

    def mean_theta_norm(self) -> float:
        """Mean L2 norm of learned theta vectors. For convergence monitoring."""
        norms=[]
        for a in range(self.n_actions):
            if self.counts[a] > 0:
                A_inv=np.linalg.inv(self.A[a])
                theta=A_inv @ self.b[a]
                norms.append(float(np.linalg.norm(theta)))
        return float(np.mean(norms)) if norms else 0.0

    @staticmethod
    def _sigmoid(x:float) -> float:
        x=max(min(x,50.0),-50.0)
        return 1.0/(1.0+math.exp(-x))



