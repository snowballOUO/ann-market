import argparse
import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.q_net import LargeQNet, causal_dr_bellman_loss
from src.models.distill import SmallQNet, distill_q_net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=None, help="Specific log dir to train on")
    ap.add_argument("--output", default="models/qnet_distilled_v1.pt")
    args = ap.parse_args()

    from src.causal.dr_estimator import load_logs, build_state, build_action_index

    if args.log_dir:
        latest_log_dir = args.log_dir
    else:
        print("1. 正在寻找最新的历史日志...")
        log_dirs = sorted(glob.glob("logs/run_*"))
        if not log_dirs:
            raise FileNotFoundError("No log directories found under logs/")
        latest_log_dir = log_dirs[-1]

    import pandas as pd
    df = load_logs(latest_log_dir)  # 自动清洗 NaN 和 orphan 记录
    print(f"成功加载日志: {latest_log_dir}，共 {len(df)} 条轨迹。")

    # ── 合并孤儿 shadow recall (晚到但 valid) ──
    orphan_files = sorted(glob.glob(os.path.join(latest_log_dir, "*orphan*.parquet")))
    if orphan_files:
        df_orphan = pd.concat([pd.read_parquet(f) for f in orphan_files], ignore_index=True)
        df_orphan = df_orphan.dropna(subset=["Q_t"])
        recall_map = dict(zip(df_orphan["query_id"], df_orphan["Q_t"]))
        # Overwrite Q_t for matching query_ids
        matched = df["query_id"].isin(recall_map)
        df.loc[matched, "Q_t"] = df.loc[matched, "query_id"].map(recall_map)
        print(f"  Merged {len(df_orphan)} orphan shadow recalls, matched {matched.sum()} trajectories")

    print("2. 正在提取 Causal 状态特征 (Deconfounding)...")
    states = build_state(df).astype(np.float32)        # (N, 6) — 所有维度正确填充
    actions = build_action_index(df).astype(np.int64)  # (N,)  — 真实日志动作
    rewards = df["R_t"].values.astype(np.float32)

    # ── Q_t 质量反馈：shadow-sampled 查询用真实 recall 修正 reward ──
    Q_t = df["Q_t"].values.astype(float)  # NaN for non-sampled queries
    mean_q = float(np.nanmean(Q_t)) if np.any(~np.isnan(Q_t)) else 0.5
    target_recall = max(mean_q, 0.3)
    quality_lambda = 0.05 / max(mean_q, 0.3)
    n_quality = int((~np.isnan(Q_t)).sum())
    print(f"  Auto-calibrated: mean_recall={mean_q:.3f}, target={target_recall:.2f}, lambda={quality_lambda:.4f}")
    q_mask = ~np.isnan(Q_t)
    n_quality = int(q_mask.sum())
    if n_quality > 0:
        quality_bonus = quality_lambda * (Q_t[q_mask] - target_recall)
        rewards = rewards.copy()
        rewards[q_mask] += quality_bonus
        print(f"  Quality reward adjustment: {n_quality} queries "
              f"(avg Δ={quality_bonus.mean():+.6f}, range=[{quality_bonus.min():+.4f}, {quality_bonus.max():+.4f}])")

    propensities = np.clip(df["propensity"].values, 1e-4, 1.0)

    n_actions = 25
    state_dim = states.shape[1]  # 6

    # next_state: i → i+1（离线 RL 标准做法），最后一条为终止态
    next_states = np.zeros_like(states, dtype=np.float32)
    next_states[:-1] = states[1:]

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    props_t = torch.tensor(propensities, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)

    print("3. 正在训练 LargeQNet (Causal DR-Bellman)...")
    large_q = LargeQNet(state_dim, n_actions)
    # 致命修复 3：降低学习率，防止在随机数据上梯度暴走
    optimizer = optim.Adam(large_q.parameters(), lr=1e-4) 
    
    dataset = TensorDataset(states_t, actions_t, rewards_t, next_states_t, props_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(5):
        total_loss = 0
        for s, a, r, s_next, p in loader:
            optimizer.zero_grad()
            # 降低 gamma，削弱无意义的自举反馈
            loss = causal_dr_bellman_loss(large_q, s, a, r, s_next, p, gamma=0.5) 
            
            if torch.isnan(loss):
                continue # 如果出现异常 Batch，直接忽略保命
                
            loss.backward()
            # 致命修复 4：梯度裁剪，强制锁死权重的更新幅度
            torch.nn.utils.clip_grad_norm_(large_q.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/5, Loss: {total_loss/max(1, len(loader)):.4f}")

    print("4. 正在蒸馏为 SmallQNet (Hidden=32) 并导出 TorchScript...")
    small_q = SmallQNet(state_dim, n_actions)
    distill_q_net(large_q, small_q, DataLoader(states_t, batch_size=256),
                   state_dim=state_dim, epochs=5)
    
    os.makedirs("models", exist_ok=True)
    dummy_input = torch.randn(1, state_dim)
    traced_model = torch.jit.trace(small_q, dummy_input)
    traced_model.save(args.output)
    print(f"✅ 模型导出成功：{args.output}")

if __name__ == "__main__":
    main()
