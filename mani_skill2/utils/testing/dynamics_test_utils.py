import torch
import torch.nn as nn


class MockFCNetwork(nn.Module):
    def __init__(self, state_size: int, act_size: int):
        super().__init__()
        self.layer = nn.Linear(state_size + act_size, state_size)

    def forward(self, state: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, act), dim=-1)
        x = self.layer(x)
        return x


def get_mock_fc_batch(
    batch_size: int,
    state_size: int,
    act_size: int,
):
    state = torch.zeros((batch_size, state_size))
    act = torch.zeros((batch_size, act_size))
    new_state = torch.zeros((batch_size, state_size))
    return dict(state=state, actions=act, new_state=new_state)
