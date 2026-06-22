import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.q_net import LargeQNet, causal_dr_bellman_loss
from src.models.distill import SmallQNet, distill_q_net

def main():
    print("1. 正在寻找最新的历史日志...")
    log_dirs = sorted(glob.glob("logs/run_*"))
    latest_log_dir = log_dirs[-1]
    parquet_files = glob.glob(os.path.join(latest_log_dir, "*.parquet"))
    
    # 致命修复 1：加载日志并强制清洗所有 NaN 缺失值
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.fillna(0.0) 
    print(f"成功加载并清洗日志: {latest_log_dir}，共 {len(df)} 条轨迹。")

    print("2. 正在提取 Causal 状态特征 (Deconfounding)...")
    # 防御性读取，确保没有任何 NaN 流入张量
    U_t = df["U_t"].values
    budget = df.get("budget_t", pd.Series(np.ones(len(df)) * 0.01)).values
    sla = df.get("sla_t", pd.Series(np.ones(len(df)) * 0.1)).values
    propensities = np.clip(df["propensity"].values, 1e-4, 1.0) # 强制斩断趋于 0 的极小值
    rewards = df["R_t"].values
    
    n_actions = 25
    state_dim = 8
    
    states = np.zeros((len(df), state_dim), dtype=np.float32)
    states[:, 0] = U_t
    states[:, 3] = sla
    states[:, 4] = budget

    # 随机动作占位（仅为跑通流水线）
    actions = np.random.randint(0, n_actions, size=len(df)) 

    # 致命修复 2：真实系统中的 next_state 应该是时间序列上的下一条日志
    next_states = np.roll(states, shift=-1, axis=0)
    next_states[-1] = np.zeros(state_dim) # 最后一个查询无后续，设为终止态

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
    distill_q_net(large_q, small_q, DataLoader(states_t, batch_size=256), epochs=5)
    
    os.makedirs("models", exist_ok=True)
    dummy_input = torch.randn(1, state_dim)
    traced_model = torch.jit.trace(small_q, dummy_input)
    traced_model.save("models/qnet_distilled_v1.pt")
    print("✅ 模型导出成功：models/qnet_distilled_v1.pt")

if __name__ == "__main__":
    main()
