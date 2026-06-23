import numpy as np
import onnxruntime as ort
from typing import Tuple
from src.system.types import Query, Action

class QLearningPolicy:
    def __init__(
        self,
        search_param_configs: list[dict],
        price_tiers: list[float],
        model_path: str = "models/qnet_distilled_v1.onnx", # 默认后缀改成 onnx
        temperature: float = 0.1
    ):
        self.configs = search_param_configs
        self.prices = price_tiers
        self.n_actions = len(self.configs) * len(self.prices)
        self.temperature = temperature
        
        # 加载蒸馏后的 ONNX 模型，开启极速推理
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # 极小模型强行绑定单线程最快
        self.ort_session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        
        # 更新版本号以作区分
        self.version = "qlearning-dr-v1-onnx"

    def _extract_state(self, query: Query, U_t: float, h_t: dict) -> np.ndarray:
        """
        构造状态向量。必须包含 U_t 以完成混淆消除 (Deconfounding)。
        直接返回纯 NumPy 数组，彻底摆脱 PyTorch 开销。
        6 维特征，与 LinUCB._build_features() 和 dr_estimator.build_state() 完全一致。
        """
        # 解析复杂的标量过滤条件
        # filter_ratio = 1.0
        # if query.filter_t:
        #     raw_val = query.filter_t.get("filter_ratio_token", "")
        #     if isinstance(raw_val, str) and "pct" in raw_val:
        #         filter_ratio = float(raw_val.replace("p", ".").replace("pct", ""))

        state_features = [
            # U_t,                        # [极其关键] W4 训练的精确难度评估
            # float(np.linalg.norm(query.v_t)),
            # query.k_t / 100.0,
            # query.sla_t,
            # query.budget_t,
            # filter_ratio,
            # h_t.get("recent_accept_rate", 0.5),
            # h_t.get("recent_mean_latency", 0.05)
            U_t,                                          # [0] 查询难度
            h_t.get("recent_accept_rate", 0.5),           # [1] 近期接受率
            h_t.get("recent_mean_latency", 0.0) * 1000,   # [2] 近期延迟 (ms)
            query.k_t / 100.0,                            # [3] 请求 k [0.1, 1.0]
            query.sla_t * 1000,                           # [4] SLA (ms)
            query.budget_t * 1000,                        # [5] 预算 (毫美元)
        ]
        
        # 直接返回 float32 的 numpy array，并加上 batch_size=1 的维度
        return np.array([state_features], dtype=np.float32)

    def decide(self, query: Query, U_t: float, h_t: dict) -> Tuple[Action, float, str]:
        """
        使用 Boltzmann Sampling 选择动作，保证 Propensity > 0 的致命不变量。
        """
        state_array = self._extract_state(query, U_t, h_t)
        
        # 纯 C++ 极速推理，拿到 Q 值 (剥离 batch 维度)
        q_values = self.ort_session.run(None, {self.input_name: state_array})[0][0]
        
        # 手写安全的波尔兹曼分布提取概率 (替代 F.softmax)
        q_shifted = q_values - np.max(q_values) # 防止指数爆炸
        exp_q = np.exp(q_shifted / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        # 强制概率下限 (防止浮点精度导致的纯 0，保护下游 IS 权重)
        probs = np.clip(probs, a_min=1e-5, a_max=1.0)
        probs = probs / probs.sum()

        # 根据 Q 值产生的概率分布进行采样
        action_idx = np.random.choice(self.n_actions, p=probs)
        propensity = float(probs[action_idx])

        # 解码动作
        z_idx = action_idx // len(self.prices)
        p_idx = action_idx % len(self.prices)
        
        action = Action(z_t=dict(self.configs[z_idx]), p_t=self.prices[p_idx])
        
        return action, propensity, self.version
