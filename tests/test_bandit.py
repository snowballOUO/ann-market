"""Tests for LinUCBPolicy."""
import numpy as np
import pytest
from src.system.types import Query
from src.agents.bandit_policy import LinUCBPolicy


def make_query(qid=0, k=10, sla=0.05, budget=0.01):
    return Query(
        id=f"q_{qid:06d}",
        v_t=np.zeros(128, dtype=np.float32),
        k_t=k, filter_t={}, sla_t=sla, budget_t=budget,
    )


CONFIGS = [
    {"nprobe": 8, "rerank_k": 50, "early_stop": False},
    {"nprobe": 32, "rerank_k": 200, "early_stop": False},
]
PRICES = [0.001, 0.005, 0.01]
# n_actions = 2 × 3 = 6


class TestLinUCBPolicy:

    def test_propensity_always_positive(self):
        """不变量 #1：propensity 永不为 0"""
        pol = LinUCBPolicy(CONFIGS, PRICES, alpha=1.0, temperature=0.5, seed=42)
        for i in range(500):
            q = make_query(i)
            _, propensity, _ = pol.decide(q, U_t=0.5, h_t={})
            assert propensity > 0.0, f"propensity={propensity} at iter {i}"
            assert propensity <= 1.0

    def test_propensity_sums_to_one(self):
        """Softmax 输出应该是一个合法的概率分布"""
        pol = LinUCBPolicy(CONFIGS, PRICES, alpha=1.0, temperature=0.5, seed=42)

        # 手动获取 softmax 概率（模拟 decide 的前两步但不采样）
        s = pol._build_features(make_query(), U_t=0.5, h_t={})
        ucbs = []
        for a in range(pol.n_actions):
            A_inv = np.linalg.inv(pol.A[a])
            theta = A_inv @ pol.b[a]
            point_est = float(theta @ s)
            bonus = pol.alpha * np.sqrt(float(s @ A_inv @ s))
            ucbs.append(point_est + bonus)
        ucbs = np.array(ucbs)
        probs = np.exp((ucbs - ucbs.max()) / pol.temperature)
        probs /= probs.sum()

        assert abs(probs.sum() - 1.0) < 1e-10
        assert (probs > 0).all()

    def test_temperature_zero_approaches_argmax(self):
        """τ → 0 时 softmax 接近 one-hot"""
        pol = LinUCBPolicy(CONFIGS, PRICES, alpha=1.0, temperature=0.5, seed=42)
        q = make_query()
        s = pol._build_features(q, U_t=0.5, h_t={})

        # 先训练：动作 0 给高 reward，让它成为最优
        pol._last_s = s
        pol._last_action_idx = 0
        for _ in range(200):
            pol.update(1.0)  # 高收益

        # 其他动作给低 reward
        for a in range(1, pol.n_actions):
            pol._last_s = s
            pol._last_action_idx = a
            for _ in range(50):
                pol.update(0.0)  # 零收益

        # 降温到 τ=0.001
        pol.set_exploration(temperature=0.001)

        _, prop, _ = pol.decide(q, U_t=0.5, h_t={})
        assert prop > 0.9, f"After training action 0, expected prop near 1, got {prop:.4f}"

    def test_update_reduces_uncertainty(self):
        """反复选同一动作后，探索 bonus 应该缩小"""
        pol = LinUCBPolicy(CONFIGS, PRICES, alpha=1.0, temperature=0.5, seed=42)
        q = make_query()
        s = pol._build_features(q, U_t=0.5, h_t={})

        # 初始探索 bonus
        A_inv_0 = np.linalg.inv(pol.A[0])
        bonus_0 = np.sqrt(float(s @ A_inv_0 @ s))

        # 对动作 0 更新 100 次
        pol._last_s = s
        pol._last_action_idx = 0
        for _ in range(100):
            pol.update(0.005)

        A_inv_100 = np.linalg.inv(pol.A[0])
        bonus_100 = np.sqrt(float(s @ A_inv_100 @ s))

        assert bonus_100 < bonus_0, (
            f"Bonus should decrease: {bonus_0:.4f} → {bonus_100:.4f}"
        )

    def test_different_seeds_yield_different_sequences(self):
        """独立 RNG：不同 seed 产生不同的动作序列"""
        pol1 = LinUCBPolicy(CONFIGS, PRICES, seed=1)
        pol2 = LinUCBPolicy(CONFIGS, PRICES, seed=2)
        q = make_query()

        actions1 = [pol1.decide(q, 0.5, {})[0].p_t for _ in range(50)]
        actions2 = [pol2.decide(q, 0.5, {})[0].p_t for _ in range(50)]

        # 极低概率下两个 seed 碰巧产生相同序列
        assert actions1 != actions2

    def test_actions_never_selected_have_high_bonus(self):
        """没被选过的动作探索 bonus 应该大于被大量选过的动作"""
        pol = LinUCBPolicy(CONFIGS, PRICES, alpha=1.0, temperature=0.5, seed=42)
        s = pol._build_features(make_query(), U_t=0.5, h_t={})

        # 动作 0：大量更新
        pol._last_s = s
        pol._last_action_idx = 0
        for _ in range(200):
            pol.update(0.005)

        # 动作 1：从未选中（初始状态）
        A_inv_0 = np.linalg.inv(pol.A[0])
        A_inv_1 = np.linalg.inv(pol.A[1])
        bonus_0 = np.sqrt(float(s @ A_inv_0 @ s))
        bonus_1 = np.sqrt(float(s @ A_inv_1 @ s))

        assert bonus_1 > bonus_0 * 2, (
            f"Untouched action should have larger bonus: {bonus_0:.4f} vs {bonus_1:.4f}"
        )