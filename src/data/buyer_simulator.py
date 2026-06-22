"""
BuyerSimulator: Week 2 implementation.

A sophisticated buyer model based on Random Utility Theory.
Simulates a diverse marketplace with Budget, Latency, and Quality sensitive buyers.
Acceptance is modeled as a logistic function over price, latency, and perceived recall.
Includes a stateful "market sentiment" to simulate long-term customer relationships.
"""
import math
import random
import numpy as np
from dataclasses import dataclass
from src.system.types import Query


@dataclass
class BuyerProfile:
    """定义买方类型的效用参数 (Utility Parameters)"""
    name: str
    
    # 敏感度系数 (Alphas): 决定该维度对最终决策的权重和斜率
    alpha_price: float
    alpha_latency: float
    alpha_recall: float
    
    # 阈值 (Thetas): 买家的心理预期锚点
    theta_price: float
    theta_latency: float
    theta_recall: float


class BuyerSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
        # 1. 定义 3 种经典的买方类型与显式参数校准
        self.profiles = [
            # BudgetBuyer: 极度价格敏感 (alpha_price 极大)，对延迟极度包容 (theta_latency=0.2s)，对召回率要求中等
            BuyerProfile(
                name="BudgetBuyer",
                alpha_price=800.0, alpha_latency=20.0, alpha_recall=5.0,
                theta_price=0.005, theta_latency=0.200, theta_recall=0.70
            ),
            # LatencyBuyer: 金融/风控场景。价格脱敏 (alpha_price 极小)，对延迟极其苛刻 (theta_latency=0.02s)，要求高召回
            BuyerProfile(
                name="LatencyBuyer",
                alpha_price=50.0, alpha_latency=500.0, alpha_recall=10.0,
                theta_price=0.050, theta_latency=0.020, theta_recall=0.90
            ),
            # QualityBuyer: 医疗/法律场景。对召回率极度敏感 (alpha_recall 极大)，价格和延迟容忍度中等
            BuyerProfile(
                name="QualityBuyer",
                alpha_price=200.0, alpha_latency=50.0, alpha_recall=30.0,
                theta_price=0.015, theta_latency=0.100, theta_recall=0.95
            )
        ]
        
        # 预设混合比例: 40% Budget, 30% Latency, 30% Quality
        self.mix_ratios = [0.4, 0.3, 0.3]
        
        # 4. 长期状态 (Stateful Tracker)
        # 记录整个市场的历史满意度 (Market Sentiment)，取值 [0, 1]
        # 初始值为 0.8。它将作为 base_utility 影响所有买家的后续接单概率
        self.market_sentiment = 0.8
        self.sentiment_momentum = 0.1 # EWMA 更新动量

    def _estimate_perceived_recall(self, query: Query, results: list) -> float:
        """
        买家在实际环境中并不知道真实的 Ground-Truth Recall。
        这里我们基于返回的结果数量和距离分布来估算一个“感知召回率”。
        在第一周/第二周骨架中，最简单的代理指标就是返回结果数量是否满足 k_t。
        """
        # if query.k_t <= 0:
        #     return 1.0
        # return min(len(results) / query.k_t, 1.0)
        # modified
        """
        修正版：利用 FAISS 返回的真实距离 (distance) 分布来判断感知召回率。
        买家无法知道绝对的召回率，但能通过“结果是不是偏离太远”来察觉质量下降。
        """
        if query.k_t <= 0 or not results:
            return 0.0

        # 提取本次查询所有返回结果的距离 (L2 Squared Distance)
        distances = [res[1] for res in results]
        mean_dist = sum(distances) / len(distances)

        # --- 针对 SIFT1M (L2距离) 的经验锚点 ---
        # 完美暴搜的平均距离通常在 40,000 左右 (最佳体验)
        # 极端欠搜索 (nprobe=1) 的平均距离会飙升到 150,000 以上 (极差体验)
        # 注意：如果后续换了 cosine 距离或别的数据集，这里的锚点需要重新 Calibrate
        best_dist_anchor = 40000.0
        worst_dist_anchor = 150000.0

        # 将距离线性映射到 [0.0, 1.0] 的召回率区间
        # 距离越小 (越接近 best_dist_anchor)，感知召回率越高
        if mean_dist <= best_dist_anchor:
            perceived = 1.0
        elif mean_dist >= worst_dist_anchor:
            perceived = 0.0
        else:
            perceived = 1.0 - ((mean_dist - best_dist_anchor) / (worst_dist_anchor - best_dist_anchor))

        # 结合数量惩罚 (以防未来引入 filter 导致结果数真的不够)
        count_penalty = len(results) / query.k_t

        return float(perceived * count_penalty)

    def respond(self, query: Query, results: list, price: float, latency: float):
        """Returns (A_t, S_t)"""
        # 3. 随机抽取当前 Query 的买家类型
        profile = self.rng.choice(self.profiles, p=self.mix_ratios)
        
        # 估算感知召回率 (Q)
        perceived_recall = self._estimate_perceived_recall(query, results)

        # 2. 计算效用函数 (Utility / Logit)
        # 市场情绪作为基础偏置 (Base sentiment bias)：情绪越好，整体接受基础概率越高
        sentiment_bias = (self.market_sentiment - 0.5) * 2.0 
        
        # U = bias + α_p(θ_p - p) + α_L(θ_L - L) + α_Q(Q - θ_Q)
        utility = (
            sentiment_bias
            + profile.alpha_price * (profile.theta_price - price)
            + profile.alpha_latency * (profile.theta_latency - latency)
            + profile.alpha_recall * (perceived_recall - profile.theta_recall)
        )

        # Sigmoid 转化为接单概率
        # 使用 clip 防止 exp 溢出
        utility_clipped = np.clip(utility, -20.0, 20.0)
        accept_prob = 1.0 / (1.0 + math.exp(-utility_clipped))

        # 掷骰子决定是否接单
        accept = self.rng.random() < accept_prob

        # 4. 计算精细的满意度 S_t 与状态更新
        if accept:
            # 接单后的满意度是一个 0 到 1 的连续值，直接映射自原本的概率 (越超出预期，满意度越接近 1)
            # 但如果踩着红线过关 (utility ≈ 0)，满意度只有 0.5 左右
            S_t = accept_prob
            
            # 更新市场情绪 (正向反馈)
            self.market_sentiment = (1 - self.sentiment_momentum) * self.market_sentiment + self.sentiment_momentum * S_t
        else:
            S_t = 0.0
            # 拒单会导致市场情绪下降 (负向惩罚机制)
            # 引入不对称惩罚：买家对糟糕体验的记忆比良好体验更深
            self.market_sentiment = (1 - self.sentiment_momentum) * self.market_sentiment + self.sentiment_momentum * 0.0

        return accept, S_t
