"""
  Doubly Robust Off-Policy Estimator — Week 5.

  Implements the DR estimator from Dudík, Langford, Li (ICML 2011):

      V̂_DR(π_new) = (1/N) Σ_i [ Q̂(s_i, π_new(s_i))
                             + ρ_i × (r_i - Q̂(s_i, a_i)) ]

  where:
      ρ_i = min( π_new(a_i|s_i) / π_b(a_i|s_i), M )   with M=50 (clipping)
      Q̂(s, a) is a LightGBM reward model trained on logged trajectories
      π_b is the behavior policy propensity (from LinUCB logs)
      π_new is the new policy to evaluate

  Also trains Q̂ and validates DR unbiasedness against online ground truth.
  """
import glob
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

def load_logs(log_dir:str) -> pd.DataFrame:
    """读取 parquet 日志目录，返回干净的 DataFrame。"""
    files = sorted(glob.glob(os.path.join(log_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files in {log_dir}")

    df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # 去掉 orphan recall 文件的行（没有 propensity 列值）
    df = df.dropna(subset=["propensity"])
    # 再去掉 policy_version 为空的行（同样是 orphan 或异常记录）
    df = df.dropna(subset=["policy_version"])

    return df.reset_index(drop=True)


def build_state(df: pd.DataFrame) -> np.ndarray:
    """从日志构建 6 维特征，和 LinUCB._build_features() 完全一致。"""
    # h_t 在日志里是字符串（"{"recent_accept_rate": 0.69, ...}"）
    # 需要用 eval 还原成 dict
    h_parsed = df["h_t"].apply(
        lambda x: eval(x) if isinstance(x, str) else {}
    )

    s = np.column_stack([
        df["U_t"].values,  # [0]
        h_parsed.apply(lambda d: d.get("recent_accept_rate", 0.5)).values,  # [1]
        h_parsed.apply(
            lambda d: d.get("recent_mean_latency", 0.0) * 1000  # [2]
        ).values,
        df["k_t"].values / 100.0,  # [3]
        df["sla_t"].values * 1000,  # [4]
        df["budget_t"].values * 1000,  # [5]
    ])
    return s.astype(np.float64)

def build_action_index(df:pd.DataFrame) -> np.ndarray:
    """z_nprobe 和 p_t → 0-24 的动作索引，和 LinUCB 的映射一致。"""
    nprobe_to_z={8:0,16:1,32:2,64:3,128:4}
    price_to_p = {0.001: 0, 0.002: 1, 0.005: 2, 0.01: 3, 0.02: 4}

    z_idx = df["z_nprobe"].map(nprobe_to_z).values
    p_idx = df["p_t"].map(price_to_p).values

    # 处理可能的 NaN（日志中出现过的 unexpected nprobe/p_t）
    if z_idx.dtype == object or p_idx.dtype == object:
        raise ValueError("Unexpected nprobe or price_t value in logs")

    return (z_idx * 5 + p_idx).astype(int)


class RewardModel:
    """用 LightGBM 训练 Q(s, a) = E[r | s, a]"""

    def __init__(self):
        self.model = None

    def fit(self, S: np.ndarray, A: np.ndarray, R: np.ndarray) -> "RewardModel":
        """
        Args:
            S: (N, 6) 状态特征
            A: (N,)   动作索引 0-24
            R: (N,)   实际收益 R_t
        """
        # 动作 One-hot 编码 (N, 25)
        A_onehot = np.zeros((len(A), 25), dtype=np.float64)
        idx = np.arange(len(A))
        A_onehot[idx, A.astype(int)] = 1.0

        # 拼接 → (N, 6+25=31)
        X = np.column_stack([S, A_onehot])

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            num_leaves=127,
            learning_rate=0.03,
            verbose=-1,
            random_state=42,
        )
        self.model.fit(X, R)
        return self

    def predict(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        """预测 Q(s, a) — (N,) 输出"""
        N = len(S)
        A_onehot = np.zeros((N, 25), dtype=np.float64)
        A_onehot[np.arange(N), A.astype(int)] = 1.0
        X = np.column_stack([S, A_onehot])
        return self.model.predict(X)


class TestPolicy:
    """可离线评估、可在线验证的新策略。"""

    def __init__(self, name: str, z_configs: list[dict], price_tiers: list[float],
                 temperature: float = 0.3):
        self.name = name
        self.z_configs = z_configs
        self.prices = price_tiers
        self.n_z = len(z_configs)
        self.n_p = len(price_tiers)
        self.n_actions = self.n_z * self.n_p
        self.temperature = temperature

    def score_actions(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def action_probs(self, s: np.ndarray) -> np.ndarray:
        """给定状态 s，返回 25 个动作的被选概率。"""
        scores = self.score_actions(s)
        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted / self.temperature)
        return exp_scores / exp_scores.sum()

    def get_action_probs_batch(self, S: np.ndarray) -> np.ndarray:
        """对多条日志批量算概率 — 返回 (N, 25)。"""
        N = len(S)
        probs = np.zeros((N, self.n_actions), dtype=np.float64)
        for i in range(N):
            probs[i] = self.action_probs(S[i])
        return probs


class LowPricePolicy(TestPolicy):
    """预算低 → 选低价，不关心 nprobe。"""

    def score_actions(self, s):
        budget = s[5]  # 毫美元
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            p_idx = a % self.n_p
            price_milli = self.prices[p_idx] * 1000  # 转毫美元
            scores[a] = -3.0 * price_milli / max(budget, 1.0)
        return scores


class QualityFocusPolicy(TestPolicy):
    """偏好高 nprobe，价格维度权重低。"""

    def score_actions(self, s):
        budget = s[5]
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            z_idx = a // self.n_p
            p_idx = a % self.n_p
            price_milli = self.prices[p_idx] * 1000
            scores[a] = 2.0 * z_idx - 0.5 * price_milli / max(budget, 1.0)
        return scores


class BudgetAwarePolicy(TestPolicy):
    """低预算→低价+低nprobe；高预算→高价+高nprobe。"""

    def score_actions(self, s):
        budget = s[5]
        is_low = 1.0 if budget < 10.0 else 0.0
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            z_idx = a // self.n_p
            p_idx = a % self.n_p
            price_milli = self.prices[p_idx] * 1000
            if is_low:
                scores[a] = -4.0 * price_milli / max(budget, 1.0) + 1.0 * (4 - z_idx)
            else:
                scores[a] = 2.0 * z_idx + 2.0 * price_milli / max(budget, 1.0)
        return scores


class SafeDefaultPolicy(TestPolicy):
    """永远偏好默认动作（nprobe=32, p=$0.005），带少量探索。"""

    def __init__(self, name, z_configs, price_tiers, temperature=0.3):
        super().__init__(name, z_configs, price_tiers, temperature)
        # 默认动作：z_idx=2, p_idx=2 → 2*5+2 = 12
        self.default_action = 2 * self.n_p + 2

    def score_actions(self, s):
        scores = np.ones(self.n_actions) * 0.1
        scores[self.default_action] = 3.0
        return scores


def dr_estimate(
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        pi_b: np.ndarray,
        pi_new: np.ndarray,
        Q_hat: RewardModel,
        clip_max: float = 50.0,
) -> dict:
    """
    计算 V̂_DR(π_new)。

    Args:
        S:        (N, 6) 状态特征
        A:        (N,)   日志记录的动作索引
        R:        (N,)   实际收益
        pi_b:     (N,)   老策略的 propensity
        pi_new:   (N, 25) 新策略的每个动作概率
        Q_hat:    reward 模型
        clip_max: importance weight 上限

    Returns:
        dict with keys: V_dr, V_dm, V_ips, mean_rho
    """
    N = len(S)

    # ── 1. 新策略选中"日志里实际选的动作"的概率 ──
    pi_new_taken = pi_new[np.arange(N), A.astype(int)]

    # ── 2. Importance weight with clipping ──
    rho = pi_new_taken / np.maximum(pi_b, 1e-10)
    rho = np.clip(rho, 0.0, clip_max)

    # ── 3. Q̂(s_i, π_new(s_i)) — 新策略的期望 Q 值 ──
    # 对每条日志，用新策略的概率向量对 25 个 Q 预测做加权平均
    Q_best = np.zeros(N)
    for a in range(25):
        Q_all_a = Q_hat.predict(S, np.full(N, a))  # 所有状态对动作 a 的 Q 预测
        Q_best += pi_new[:, a] * Q_all_a  # 用新策略的概率加权

    # ── 4. Q̂(s_i, a_i) — 日志里实际动作的 Q 预测 ──
    Q_taken = Q_hat.predict(S, A)

    # ── 5. DR 估计 ──
    V_dr = float(np.mean(Q_best + rho * (R - Q_taken)))

    # ── 6. 顺便算 DM 和 IPS 用来对比 ──
    V_dm = float(np.mean(Q_best))
    V_ips = float(np.mean(rho * R))

    return {
        "V_dr": V_dr,
        "V_dm": V_dm,
        "V_ips": V_ips,
        "mean_rho": float(np.mean(rho)),
        "rho_p99": float(np.percentile(rho, 99)),
    }


