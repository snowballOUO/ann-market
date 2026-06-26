import numpy as np
import torch

from src.agents.q_learning_policy import QLearningPolicy
from src.system.types import Query


CONFIGS = [
    {"nprobe": 8, "rerank_k": 50, "early_stop": False},
    {"nprobe": 16, "rerank_k": 100, "early_stop": False},
    {"nprobe": 32, "rerank_k": 200, "early_stop": False},
    {"nprobe": 64, "rerank_k": 400, "early_stop": False},
    {"nprobe": 128, "rerank_k": 800, "early_stop": False},
]
PRICES = [0.001, 0.002, 0.005, 0.01, 0.02]


class ConstantQ:
    def eval(self):
        return self

    def __call__(self, state):
        return torch.zeros((state.shape[0], len(CONFIGS) * len(PRICES)))


def make_query() -> Query:
    return Query(
        id="q",
        v_t=np.zeros(4, dtype=np.float32),
        k_t=10,
        filter_t={},
        sla_t=0.05,
        budget_t=0.01,
    )


def action_key(action):
    return action.z_t["nprobe"], action.p_t


def test_q_learning_policy_uses_seeded_rng(monkeypatch):
    monkeypatch.setattr(torch.jit, "load", lambda _: ConstantQ())
    q = make_query()

    p1 = QLearningPolicy(CONFIGS, PRICES, model_path="unused.pt", seed=7)
    p2 = QLearningPolicy(CONFIGS, PRICES, model_path="unused.pt", seed=7)
    p3 = QLearningPolicy(CONFIGS, PRICES, model_path="unused.pt", seed=8)

    seq1 = [action_key(p1.decide(q, U_t=0.1, h_t={})[0]) for _ in range(20)]
    seq2 = [action_key(p2.decide(q, U_t=0.1, h_t={})[0]) for _ in range(20)]
    seq3 = [action_key(p3.decide(q, U_t=0.1, h_t={})[0]) for _ in range(20)]

    assert seq1 == seq2
    assert seq1 != seq3


def test_q_learning_policy_uses_normalized_qnet_state(monkeypatch):
    monkeypatch.setattr(torch.jit, "load", lambda _: ConstantQ())
    q = Query(
        id="q",
        v_t=np.zeros(4, dtype=np.float32),
        k_t=50,
        filter_t={},
        sla_t=0.1,
        budget_t=0.02,
    )
    policy = QLearningPolicy(CONFIGS, PRICES, model_path="unused.pt", seed=7)

    state = policy._extract_state(
        q,
        U_t=0.4,
        h_t={"recent_accept_rate": 0.8, "recent_mean_latency": 0.002},
    )

    expected = torch.tensor([[0.4, 0.8, 0.4, 0.5, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(state, expected)
