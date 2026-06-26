import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeQNet(nn.Module):
    """
    离线训练的大型 Q 网络。
    输入：状态 s (必须显式包含 U_t 进行 Deconfounding)
    输出：所有可能离散动作的 Q 值评估
    """
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        # 足够深的网络以拟合复杂的因果奖励曲面
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

def causal_dr_bellman_loss(
    q_net: nn.Module,
    states: torch.Tensor,       # 包含 U_t
    actions: torch.Tensor,      # 采取的动作索引
    rewards: torch.Tensor,      # 修正后的 DR 奖励 (来自 W5)
    next_states: torch.Tensor,
    pi_b_probs: torch.Tensor,   # 日志里的旧策略倾向得分 (propensity)
    gamma: float = 0.99,
    clip_M: float = 50.0
) -> torch.Tensor:
    """
    计算 Causal DR-Bellman 残差损失。
    """
    # 1. 计算当前状态的 Q(s, a)
    q_values = q_net(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. 计算目标 Q 值 (Target): r + gamma * max_a' Q(s', a')
    with torch.no_grad():
        next_q_values = q_net(next_states)
        max_next_q, _ = next_q_values.max(dim=1)
        target_q = rewards + gamma * max_next_q

    # 3. 计算新策略在当前状态下采取动作 a 的概率 pi_new(a|s)
    # 使用与在线 Policy 相同的 Boltzmann softmax 提取概率
    temperature = 0.1
    pi_new_probs = F.softmax(q_values / temperature, dim=1)
    pi_new_a = pi_new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 4. 计算重要性权重 rho，并应用 Clipping 防御 Propensity Collapse
    rho = pi_new_a / (pi_b_probs + 1e-8)
    clipped_rho = torch.clamp(rho, max=clip_M)

    # 5. Causal Bellman 残差
    td_error = target_q - q_sa
    loss = torch.mean(clipped_rho * (td_error ** 2))
    
    return loss


def ips_bellman_loss(
    q_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    pi_b_probs: torch.Tensor,
    gamma: float = 0.99,
    clip_M: float = 50.0
) -> torch.Tensor:
    """
    Naive IPS-weighted Bellman loss used for the no-DR ablation.
    """
    q_values = q_net(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = q_net(next_states)
        max_next_q, _ = next_q_values.max(dim=1)
        target_q = rewards + gamma * max_next_q

    weights = torch.clamp(1.0 / (pi_b_probs + 1e-8), max=clip_M)
    td_error = target_q - q_sa
    return torch.mean(weights * (td_error ** 2))


def stable_bellman_loss(
    q_net: nn.Module,
    target_q_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    pi_b_probs: torch.Tensor,
    gamma: float = 0.5,
    clip_M: float = 50.0,
) -> torch.Tensor:
    """
    Stable behavior-weighted Bellman loss for QNet training.

    Unlike causal_dr_bellman_loss, this does not multiply the TD error by the
    current model's pi_new(a|s). That keeps the model from changing its own
    sample weights during training, which made the offline QNet seed-sensitive.
    """
    q_values = q_net(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_q_net(next_states)
        max_next_q, _ = next_q_values.max(dim=1)
        target_q = rewards + gamma * max_next_q

    weights = torch.clamp(1.0 / (pi_b_probs + 1e-8), max=clip_M)
    td_loss = F.smooth_l1_loss(q_sa, target_q, reduction="none")
    return torch.mean(weights * td_loss)
