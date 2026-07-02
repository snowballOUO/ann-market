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
    def __init__(self, seed: int = 42,
                 best_dist_anchor: float = 40000.0,
                 worst_dist_anchor: float = 150000.0,
                 nprobe_recall: dict = None):
        self.rng = np.random.default_rng(seed)
        self.best_dist_anchor = best_dist_anchor
        self.worst_dist_anchor = worst_dist_anchor
        if nprobe_recall:
            probes = sorted(nprobe_recall.keys())
            self._nr_probes = np.array(probes, dtype=np.float64)
            self._nr_recalls = np.array([nprobe_recall[p] for p in probes], dtype=np.float64)
        else:
            self._nr_probes = None
        
        self.profiles = [
            BuyerProfile(
                name="BudgetBuyer",
                alpha_price=800.0, alpha_latency=20.0, alpha_recall=5.0,
                theta_price=0.005, theta_latency=0.200, theta_recall=0.70
            ),
            BuyerProfile(
                name="LatencyBuyer",
                alpha_price=50.0, alpha_latency=500.0, alpha_recall=10.0,
                theta_price=0.050, theta_latency=0.020, theta_recall=0.90
            ),
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

    def get_profile(self, rng_seed: int) -> BuyerProfile:
        """返回给定 seed 下会被选中的 buyer 类型（确定性，不消费 rng）。"""
        rng = np.random.default_rng(rng_seed)
        return rng.choice(self.profiles, p=self.mix_ratios)

    def _estimate_perceived_recall(self, query: Query, results: list, nprobe: int = None) -> float:
        if query.k_t <= 0 or not results:
            return 0.0

        # 优先用 nprobe→recall 查表（精确映射）
        if self._nr_probes is not None and nprobe is not None:
            perceived = float(np.interp(nprobe, self._nr_probes, self._nr_recalls))
        else:
            # 回退到距离锚点映射
            distances = [res[1] for res in results]
            mean_dist = sum(distances) / len(distances)
            if mean_dist <= self.best_dist_anchor:
                perceived = 1.0
            elif mean_dist >= self.worst_dist_anchor:
                perceived = 0.0
            else:
                perceived = 1.0 - ((mean_dist - self.best_dist_anchor) / (self.worst_dist_anchor - self.best_dist_anchor))

        count_penalty = min(len(results) / max(query.k_t, 1), 1.0)
        return float(perceived * count_penalty)

    def respond(self, query: Query, results: list, price: float, latency: float,
                nprobe: int = None, gt_ids: list = None):
        """Returns (A_t, S_t)"""
        # 3. 随机抽取当前 Query 的买家类型
        profile = self.rng.choice(self.profiles, p=self.mix_ratios)

        # 感知召回率 (Q)：优先用 GT 真实 recall，否则查 nprobe 表
        if gt_ids is not None:
            k_t = min(query.k_t, len(gt_ids), 10)
            approx_set = set(r[0] for r in results[:k_t])
            true_set = set(int(x) for x in gt_ids[:k_t])
            perceived_recall = len(approx_set & true_set) / len(true_set) if true_set else 0.5
        else:
            perceived_recall = self._estimate_perceived_recall(query, results, nprobe)

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
            self.market_sentiment = (1 - self.sentiment_momentum) * self.market_sentiment + self.sentiment_momentum * 0.0

        return accept, S_t
