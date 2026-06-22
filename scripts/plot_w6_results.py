import matplotlib.pyplot as plt
import numpy as np

def main():
    # 全局字体和清晰度设置
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 创建 1x3 的画布
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Week 6: Causal Q-Learning vs. LinUCB Performance', fontsize=16, fontweight='bold', y=1.05)

    # ==========================================
    # 图 1: Causal DR-Bellman Loss 收敛曲线
    # ==========================================
    epochs = [1, 2, 3, 4, 5]
    losses = [0.0057, 0.0041, 0.0033, 0.0025, 0.0019]
    
    ax1.plot(epochs, losses, marker='o', markersize=8, linewidth=2.5, color='#1f77b4')
    ax1.set_title('Offline Q-Net Training Convergence', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Causal DR-Bellman Loss', fontsize=12)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # ==========================================
    # 图 2: 接单率 (Accept Rate) vs 收益 (Revenue)
    # ==========================================
    labels = ['LinUCB\n(W3 Baseline)', 'Distilled Q-Net\n(W6 Final)']
    x = np.arange(len(labels))
    width = 0.35
    
    accept_rates = [91.18, 95.75]
    revenues = [58.53, 50.20]
    
    # 左 Y 轴：接单率
    color1 = '#2ca02c' # 绿色
    bars1 = ax2.bar(x - width/2, accept_rates, width, label='Accept Rate (%)', color=color1, alpha=0.8)
    ax2.set_ylabel('Accept Rate (%)', fontsize=12, color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.set_ylim(80, 100) # 放大差异
    
    # 右 Y 轴：收益
    ax2_twin = ax2.twinx()
    color2 = '#ff7f0e' # 橙色
    bars2 = ax2_twin.bar(x + width/2, revenues, width, label='Revenue ($)', color=color2, alpha=0.8)
    ax2_twin.set_ylabel('Revenue ($)', fontsize=12, color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2_twin.set_ylim(40, 65)
    
    ax2.set_title('Market Share vs. Revenue Strategy', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    
    # 合并图例
    lines, labels_leg = ax2.get_legend_handles_labels()
    lines2, labels_leg2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels_leg + labels_leg2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # ==========================================
    # 图 3: 蒸馏延迟 (P99 Latency) vs 吞吐量 (QPS)
    # ==========================================
    p99_latency = [2.29, 2.00]
    qps = [329, 391]
    
    # 左 Y 轴：P99 延迟
    color3 = '#d62728' # 红色
    bars3 = ax3.bar(x - width/2, p99_latency, width, label='P99 Latency (ms)', color=color3, alpha=0.8)
    ax3.set_ylabel('P99 Latency (ms)', fontsize=12, color=color3)
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_ylim(0, 3.0)
    
    # 画一条 2.0ms 的 SLA 红线
    ax3.axhline(y=2.0, color='r', linestyle=':', linewidth=2, label='SLA Target (2.0ms)')
    
    # 右 Y 轴：QPS
    ax3_twin = ax3.twinx()
    color4 = '#9467bd' # 紫色
    bars4 = ax3_twin.bar(x + width/2, qps, width, label='System QPS', color=color4, alpha=0.8)
    ax3_twin.set_ylabel('System QPS (queries/sec)', fontsize=12, color=color4)
    ax3_twin.tick_params(axis='y', labelcolor=color4)
    ax3_twin.set_ylim(0, 450)
    
    ax3.set_title('System Efficiency: Distillation Impact', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    
    # 合并图例
    lines3, labels_leg3 = ax3.get_legend_handles_labels()
    lines4, labels_leg4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels_leg3 + labels_leg4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # ==========================================
    # 自动调整布局并保存
    # ==========================================
    plt.tight_layout()
    plt.savefig('w6_evaluation_plots.png', bbox_inches='tight')
    print("✅ 绘图完成！图表已保存为: w6_evaluation_plots.png")
    # 如果是在有界面的系统上，可以取消下一行的注释直接预览
    # plt.show()

if __name__ == '__main__':
    main()
