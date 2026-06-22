import torch
import torch.nn as nn
import torch.optim as optim

class SmallQNet(nn.Module):
    """
    在线推理专用的轻量级 Q 网络 (hidden=32)。
    """
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

def distill_q_net(teacher: nn.Module, student: nn.Module, dataloader, epochs: int = 10):
    """
    将 LargeQNet 的知识蒸馏到 SmallQNet。
    """
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        for states in dataloader:
            # 取出状态 (无需动作和奖励，纯粹对齐函数空间)
            with torch.no_grad():
                teacher_q = teacher(states)
            
            student_q = student(states)
            
            # 直接回归 Teacher 的 Q 值
            loss = mse_loss(student_q, teacher_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 最终保存为 TorchScript 格式，脱离 Python GIL 进一步加速推理
    student.eval()
    dummy_input = torch.randn(1, states.shape[1])
    traced_script_module = torch.jit.trace(student, dummy_input)
    traced_script_module.save("models/qnet_distilled_v1.pt")
