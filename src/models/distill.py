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

def distill_q_net(teacher: nn.Module, student: nn.Module, dataloader,
                   state_dim: int, epochs: int = 10):
    """
    将 LargeQNet 的知识蒸馏到 SmallQNet。
    """
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        for batch in dataloader:
            states = batch[0] if isinstance(batch, (list, tuple)) else batch
            with torch.no_grad():
                teacher_q = teacher(states)

            student_q = student(states)

            loss = mse_loss(student_q, teacher_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 导出 TorchScript，用显式 state_dim 保证维度正确
    student.eval()
    dummy_input = torch.randn(1, state_dim)
    traced_script_module = torch.jit.trace(student, dummy_input)
    traced_script_module.save("models/qnet_distilled_v1.pt")

    # Add onnx
    torch.onnx.export(
    student,                          # 这里传入你刚训练好的 student 模型
    dummy_input,                      # 给定一个示范形状
    "models/qnet_distilled_v1.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,         # 常量折叠，直接把固定参数算好，极大加速
    input_names=['state'],
    output_names=['q_values']
    )
    print("✅ 成功导出 TorchScript 和 ONNX 双版本模型！")
