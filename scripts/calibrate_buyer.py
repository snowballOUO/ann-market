import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 引入我们在第 2 周刚刚写好的买家模拟器
from src.data.buyer_simulator import BuyerSimulator

def calculate_accept_prob(profile, price, latency, recall, market_sentiment=0.8):
    """
    辅助函数：提取 BuyerSimulator 中的效用公式，计算理论接单概率。
    这允许我们在不更新实际系统状态（无副作用）的情况下绘制连续的数学曲线。
    """
    sentiment_bias = (market_sentiment - 0.5) * 2.0 
    utility = (
        sentiment_bias
        + profile.alpha_price * (profile.theta_price - price)
        + profile.alpha_latency * (profile.theta_latency - latency)
        + profile.alpha_recall * (recall - profile.theta_recall)
    )
    utility_clipped = np.clip(utility, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-utility_clipped))

def main():
    # 实例化买家模拟器以获取配置好的 Profiles
    simulator = BuyerSimulator()
    profiles = {p.name: p for p in simulator.profiles}
    
    # 获取三种买家的配置
    budget_buyer = profiles["BudgetBuyer"]
    latency_buyer = profiles["LatencyBuyer"]
    quality_buyer = profiles["QualityBuyer"]

    # 统一设置绘图全局参数
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.autolayout": True
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # ---------------------------------------------------------
    # 图 1: BudgetBuyer (价格敏感)
    # 冻结条件: latency = 0.05 (良好), recall = 0.9 (良好)
    # ---------------------------------------------------------
    prices = np.linspace(0.001, 0.020, 100)
    probs_price = [calculate_accept_prob(budget_buyer, p, 0.05, 0.9) * 100 for p in prices]
    
    ax = axes[0]
    ax.plot(prices, probs_price, color='#2ca02c', linewidth=2.5)
    ax.axvline(x=budget_buyer.theta_price, color='gray', linestyle='--', label=f'Threshold (θ={budget_buyer.theta_price})')
    ax.set_title("BudgetBuyer: Price Sensitivity")
    ax.set_xlabel("Price ($p_t$)")
    ax.set_ylabel("Accept Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 图 2: LatencyBuyer (延迟敏感)
    # 冻结条件: price = 0.01 (中等), recall = 0.9 (良好)
    # ---------------------------------------------------------
    latencies = np.linspace(0.005, 0.100, 100)
    probs_lat = [calculate_accept_prob(latency_buyer, 0.01, L, 0.9) * 100 for L in latencies]
    
    ax = axes[1]
    ax.plot(latencies, probs_lat, color='#d62728', linewidth=2.5)
    ax.axvline(x=latency_buyer.theta_latency, color='gray', linestyle='--', label=f'Threshold (θ={latency_buyer.theta_latency})')
    ax.set_title("LatencyBuyer: Latency Sensitivity")
    ax.set_xlabel("Latency (Seconds)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 图 3: QualityBuyer (召回率敏感)
    # 冻结条件: price = 0.01 (中等), latency = 0.05 (良好)
    # ---------------------------------------------------------
    recalls = np.linspace(0.0, 1.0, 100)
    probs_rec = [calculate_accept_prob(quality_buyer, 0.01, 0.05, R) * 100 for R in recalls]
    
    ax = axes[2]
    ax.plot(recalls, probs_rec, color='#1f77b4', linewidth=2.5)
    ax.axvline(x=quality_buyer.theta_recall, color='gray', linestyle='--', label=f'Threshold (θ={quality_buyer.theta_recall})')
    ax.set_title("QualityBuyer: Recall Sensitivity")
    ax.set_xlabel("Perceived Recall ($Q$)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 保存图表
    # ---------------------------------------------------------
    out_dir = "reports/figs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "buyer_calibration.png")
    
    plt.savefig(out_path, format='png', bbox_inches='tight')
    plt.close()
    
    print(f"买家参数校准图已成功生成：{out_path}")

if __name__ == "__main__":
    main()
