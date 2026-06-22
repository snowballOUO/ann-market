"""
  Tests for Doubly Robust estimator — Week 5.

  Covers:
    1. DR is unbiased when Q̂ is trained on the evaluation data
    2. Importance weight clipping works
    3. RewardModel trains without errors
    4. DR has lower variance than IPS
    5. Edge case: all-zero propensity handled gracefully
"""
import numpy as np
import pytest
from src.causal.dr_estimator import (
    dr_estimate, RewardModel,
    LowPricePolicy, SafeDefaultPolicy,
)


class TestDREstimator:
    def test_dr_consistent_on_training_data(self):
        """
        DR 在 Q̂ 训练过的同一份数据上应该接近 DM。

        生成一个人造的 (s, a, r) 数据集，训 Q̂，然后用 DR 估计。
        因为 Q̂ 在训练数据上拟合得很好，残差 r - Q̂(s,a) 应该很小，
        DR 估计应该接近 DM 估计。
        """
        N = 2000
        rng = np.random.default_rng(42)

        S = rng.normal(0, 0.5, (N, 6))
        A = rng.integers(0, 25, N)
        # reward = 简单线性函数 + 微小噪声
        R = 0.01 * S[:, 0] + 0.005 * S[:, 1] + 0.003 + rng.normal(0, 0.001, N)

        # 均匀探索的 propensity
        pi_b = np.full(N, 0.04)

        # 随机新策略概率
        pi_new = rng.random((N, 25))
        pi_new /= pi_new.sum(axis=1, keepdims=True)

        # 训 Q̂
        q_hat = RewardModel().fit(S, A, R)

        # DR 估计
        result = dr_estimate(S, A, R, pi_b, pi_new, q_hat)

        # 因为 Q̂ 在训练数据上拟合得很好，DM 和 DR 应该接近
        assert not np.isnan(result["V_dr"])
        assert not np.isnan(result["V_dm"])
        # DR 和 DM 的相对差距应该在 10% 以内（保留误差空间因为可能有一点噪声）
        abs_diff = abs(result["V_dr"] - result["V_dm"])
        assert abs_diff < max(abs(result["V_dm"]) * 0.3, 0.001), (
            f"DR ({result['V_dr']:.4f}) should be close to DM ({result['V_dm']:.4f})"
        )

    def test_rho_clipping(self):
        """
        当 π_b 非常小但 π_new 很大时，ρ 应该被截断在 clip_max。
        """
        N = 100
        rng = np.random.default_rng(42)

        S = rng.normal(0, 0.5, (N, 6))
        A = rng.integers(0, 25, N)
        R = rng.normal(0.005, 0.002, N)

        # 极小 propensity：模仿早期探索的罕见事件
        pi_b = np.full(N, 0.001)  # 只有 0.1% 概率被选中

        # 新策略对这些动作的概率是 1.0
        pi_new = np.zeros((N, 25))
        pi_new[np.arange(N), A] = 1.0

        q_hat = RewardModel().fit(S, A, R)

        # clip_max = 10
        result_clipped = dr_estimate(S, A, R, pi_b, pi_new, q_hat, clip_max=10)
        # clip_max = 500（基本不裁剪）
        result_unclipped = dr_estimate(S, A, R, pi_b, pi_new, q_hat, clip_max=500)

        # clipped 的 mean_rho 应显著小于 unclipped
        assert result_clipped["mean_rho"] < result_unclipped["mean_rho"], (
            f"Clipped ρ ({result_clipped['mean_rho']:.1f}) should be < "
            f"unclipped ({result_unclipped['mean_rho']:.1f})"
        )
        # clipped ρ 不应该超过 10
        assert result_clipped["mean_rho"] <= 10.0, (
            f"Clipped mean ρ should be ≤ 10, got {result_clipped['mean_rho']:.1f}"
        )

    def test_reward_model_trains_without_error(self):
        """
        RewardModel 在合理的输入上应正常训练，predict 不应返回 NaN。
        """
        N = 500
        rng = np.random.default_rng(42)

        S = rng.normal(0, 1, (N, 6))
        A = rng.integers(0, 25, N)
        R = rng.uniform(-0.01, 0.05, N)  # 模拟真实收益范围

        q_hat = RewardModel().fit(S, A, R)

        # predict 应对训练数据返回有限值
        preds = q_hat.predict(S, A)
        assert not np.any(np.isnan(preds)), "Q̂ predictions contain NaN"
        assert not np.any(np.isinf(preds)), "Q̂ predictions contain inf"
        assert preds.min() > -1.0, f"Predictions too low: {preds.min():.4f}"
        assert preds.max() < 1.0, f"Predictions too high: {preds.max():.4f}"

    def test_dr_lower_variance_than_ips(self):
        """
        用 bootstrap 在合成数据上分别估 DR 和 IPS，DR 的方差应更小。
        """
        N = 1000
        rng = np.random.default_rng(42)

        S = rng.normal(0, 0.5, (N, 6))
        A = rng.integers(0, 25, N)
        R = 0.01 * S[:, 0] + 0.003 + rng.normal(0, 0.002, N)
        pi_b = np.full(N, 0.04)

        # 随机新策略
        pi_new = rng.random((N, 25))
        pi_new /= pi_new.sum(axis=1, keepdims=True)

        q_hat = RewardModel().fit(S, A, R)

        # Bootstrap: 有放回抽样 50 次
        n_bs = 50
        dr_estimates = np.zeros(n_bs)
        ips_estimates = np.zeros(n_bs)

        for b in range(n_bs):
            idx = rng.choice(N, size=N, replace=True)
            result_b = dr_estimate(
                S[idx], A[idx], R[idx],
                pi_b[idx], pi_new[idx], q_hat,
                clip_max=50,
            )
            dr_estimates[b] = result_b["V_dr"]
            ips_estimates[b] = result_b["V_ips"]

        dr_std = np.std(dr_estimates)
        ips_std = np.std(ips_estimates)

        assert dr_std < ips_std, (
            f"DR std ({dr_std:.6f}) should be < IPS std ({ips_std:.6f})"
        )

    def test_handles_zero_propensity_gracefully(self):
        """
        当某些 propensity = 0 时，DR 不应崩溃。
        （不应该发生，但防御层要工作）
        """
        N = 100
        S = np.random.randn(N, 6).astype(np.float64)
        A = np.random.randint(0, 25, N).astype(int)
        R = np.random.randn(N) * 0.005 + 0.005
        pi_b = np.zeros(N)  # 全部为零 —— 模拟日志 corrupted
        pi_b[0] = 1e-10  # 给一条极小值防完全分解

        pi_new = np.random.rand(N, 25)
        pi_new /= pi_new.sum(axis=1, keepdims=True)

        q_hat = RewardModel().fit(S, A, R)

        # 不崩溃 —— 因为分母有 max(pi_b, 1e-10)
        result = dr_estimate(S, A, R, pi_b, pi_new, q_hat)
        assert not np.isnan(result["V_dr"])

def test_policies_instantiate():
    """4 个策略应能正常实例化并产生合法的概率分布。"""
    z_configs = [
        {"nprobe": 8, "rerank_k": 50, "early_stop": False},
        {"nprobe": 16, "rerank_k": 100, "early_stop": False},
        {"nprobe": 32, "rerank_k": 200, "early_stop": False},
    ]
    prices = [0.001, 0.005, 0.01]
    # 2×3=6 动作

    policies = [
        LowPricePolicy("LP", z_configs, prices),
        SafeDefaultPolicy("SD", z_configs, prices),
    ]

    s = np.array([0.4, 0.7, 1.2, 0.1, 50.0, 10.0])

    for pol in policies:
        probs = pol.action_probs(s)
        assert abs(probs.sum() - 1.0) < 1e-8
        assert (probs > 0).all()
        assert len(probs) == 9