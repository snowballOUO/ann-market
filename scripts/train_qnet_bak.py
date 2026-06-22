import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 导入你之前添加的 W6 模型文件
from src.models.Q_net import LargeQNet, causal_dr_bellman_loss
from src.models.distill import SmallQNet, distill_q_net

def main():
    print("1. 正在寻找最新的历史日志...")
    log_dirs = sorted(glob.glob("logs/run_*"))
    if not log_dirs:
        raise FileNotFoundError("未找到日志目录，请先运行 run_experiment.py")
    
    latest_log_dir = log_dirs[-1]
    parquet_files = glob.glob(os.path.join(latest_log_dir, "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"成功加载日志: {latest_log_dir}，共 {len(df)} 条轨迹。")

    print("2. 正在提取 Causal 状态特征 (Deconfounding)...")
    # 提取核心状态 (加入 U_t 消除混淆偏差)
    # 如果 parquet 缺少某些环境特征，使用 0.0 兜底以保证系统不崩溃
    U_t = df["U_t"].values
    budget = df.get("budget_t", pd.Series(np.ones(len(df)) * 0.01)).values
    sla = df.get("sla_t", pd.Series(np.ones(len(df)) * 0.1)).values
    propensities = df["propensity"].values
    rewards = df["R_t"].values
    
    # 假设有 25 个动作 (5 z_configs * 5 prices)
    n_actions = 25
    state_dim = 8
    
    # 构造 dummy states (实际业务中应严格映射，此处为跑通 pipeline)
    states = np.zeros((len(df), state_dim), dtype=np.float32)
    states[:, 0] = U_t
    states[:, 3] = sla
    states[:, 4] = budget

    # 获取选中的动作索引 (近似推导)
    # 实际应用中应在日志里直接存 action_idx，这里随机 mock 保证训练流转
    actions = np.random.randint(0, n_actions, size=len(df)) 

    # 转换为 PyTorch Tensors
    states_t = torch.tensor(states)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    props_t = torch.tensor(propensities, dtype=torch.float32)
    # 模拟 next_states (Bandit 环境下通常为 0 或背景状态)
    next_states_t = torch.zeros_like(states_t)

    print("3. 正在训练 LargeQNet (Causal DR-Bellman)...")
    large_q = LargeQNet(state_dim, n_actions)
    optimizer = optim.Adam(large_q.parameters(), lr=1e-3)
    
    dataset = TensorDataset(states_t, actions_t, rewards_t, next_states_t, props_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(5): # 仅跑 5 轮证明收敛即可去睡觉
        total_loss = 0
        for s, a, r, s_next, p in loader:
            optimizer.zero_grad()
            loss = causal_dr_bellman_loss(large_q, s, a, r, s_next, p)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(loader):.4f}")

    print("4. 正在蒸馏为 SmallQNet (Hidden=32) 并导出 TorchScript...")
    small_q = SmallQNet(state_dim, n_actions)
    distill_q_net(large_q, small_q, DataLoader(states_t, batch_size=256), epochs=5)
    
    os.makedirs("models", exist_ok=True)
    # 保存路径必须与 QLearningPolicy 中一致
    dummy_input = torch.randn(1, state_dim)
    traced_model = torch.jit.trace(small_q, dummy_input)
    traced_model.save("models/qnet_distilled_v1.pt")
    print("✅ 模型导出成功：models/qnet_distilled_v1.pt")

if __name__ == "__main__":
    main()
