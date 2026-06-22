import pandas as pd, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 加载日志
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
files = sorted(glob.glob(_os.path.join(PROJECT_ROOT, 'logs/run_*/run_*.parquet')))
df = pd.concat([pd.read_parquet(f) for f in files])

# 过滤掉孤儿记录（没有 policy_version 的）
df = df.dropna(subset=["policy_version"])
# 读孤儿 recall
shadow_sampled = 0
mean_recall = None
orphan_files = sorted(glob.glob(_os.path.join(PROJECT_ROOT, 'logs/run_*/*orphan*.parquet')))
if orphan_files:
    df_orphan = pd.concat([pd.read_parquet(f) for f in orphan_files])
    shadow_sampled = len(df_orphan)
    mean_recall = df_orphan["Q_t"].mean()

print(f"有效记录数: {len(df)}")
print()

# ── 问题 1: propensity 最小值 ──
print("===== 问题 1: propensity 最小值 =====")
print(f"min propensity: {df['propensity'].min()}")
print(f"propensity 为 0 的行数: {(df['propensity'] == 0).sum()}")
print()

# ── 问题 2: shadow 采样数和 mean recall ──
print("===== 问题 2: shadow 采样 =====")
shadow_mask = df["Q_t"].notna()
n_shadow = shadow_mask.sum()
print(f"shadow 采样: {shadow_sampled}/{len(df)} ({100 * shadow_sampled / len(df):.2f}%)")
print(f"mean recall (shadow): {mean_recall:.4f}" if mean_recall is not None else "mean recall (shadow): N/A")
print()

# ── 问题 3: 决策延迟 p99（L_t 即 FAISS 搜索延迟） ──
print("===== 问题 3: 延迟 =====")
print(f"L_t mean: {df['L_t'].mean() * 1000:.2f} ms")
print(f"L_t p50:  {df['L_t'].quantile(0.50) * 1000:.2f} ms")
print(f"L_t p99:  {df['L_t'].quantile(0.99) * 1000:.2f} ms")
print()

# ── 问题 4: 不同价格档位的 accept rate ──
print("===== 问题 4: 各价格档位 accept rate =====")
price_accept = df.groupby("p_t").agg(
    accept_rate=("A_t", "mean"),
    n=("A_t", "count")
).sort_index()
print(price_accept)
print()

# ── 画 accept-rate-by-price 柱状图 ──
os.makedirs("reports/figs", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
labels = [f"${p:.3f}" for p in price_accept.index]
bars = ax.bar(labels, price_accept["accept_rate"])
ax.set_xlabel("Price Tier (USD)")
ax.set_ylabel("Accept Rate")
ax.set_title("Accept Rate by Price Tier (Week 1)")

for bar, rate, n in zip(bars, price_accept["accept_rate"], price_accept["n"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{rate:.1%}\nn={n}", ha="center", fontsize=9)

ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("reports/figs/accept_rate_by_price.png", dpi=150)
print("图表已保存到 reports/figs/accept_rate_by_price.png")

plt.close()

