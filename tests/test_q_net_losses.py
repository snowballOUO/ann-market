import torch

from src.models.q_net import causal_dr_bellman_loss, ips_bellman_loss, stable_bellman_loss


class FixedQNet(torch.nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer("q_values", torch.tensor(q_values, dtype=torch.float32))

    def forward(self, states):
        return self.q_values.repeat(states.shape[0], 1)


def test_ips_bellman_loss_is_distinct_from_dr_weighting():
    q_net = FixedQNet([[0.0, 1.0, 2.0]])
    states = torch.zeros(4, 2)
    next_states = torch.zeros(4, 2)
    actions = torch.tensor([0, 1, 2, 1])
    rewards = torch.tensor([0.1, 0.2, 0.3, 0.2])
    propensities = torch.tensor([0.2, 0.2, 0.2, 0.2])

    dr_loss = causal_dr_bellman_loss(
        q_net,
        states,
        actions,
        rewards,
        next_states,
        propensities,
        gamma=0.5,
    )
    ips_loss = ips_bellman_loss(
        q_net,
        states,
        actions,
        rewards,
        next_states,
        propensities,
        gamma=0.5,
    )

    assert torch.isfinite(ips_loss)
    assert not torch.isclose(ips_loss, dr_loss)


def test_stable_bellman_loss_does_not_depend_on_unselected_action_logits():
    target_q = FixedQNet([[0.0, 0.0, 0.0]])
    q_with_large_tail = FixedQNet([[0.0, 0.0, 20.0]])
    q_with_different_tail = FixedQNet([[0.0, 20.0, 0.0]])
    states = torch.zeros(2, 2)
    next_states = torch.zeros(2, 2)
    actions = torch.tensor([0, 0])
    rewards = torch.tensor([0.1, 0.1])
    propensities = torch.tensor([0.5, 0.5])

    loss_a = stable_bellman_loss(
        q_with_large_tail,
        target_q,
        states,
        actions,
        rewards,
        next_states,
        propensities,
        gamma=0.5,
    )
    loss_b = stable_bellman_loss(
        q_with_different_tail,
        target_q,
        states,
        actions,
        rewards,
        next_states,
        propensities,
        gamma=0.5,
    )

    assert torch.isfinite(loss_a)
    assert torch.allclose(loss_a, loss_b, atol=1e-7)
