import torch
import torch.nn as nn


class MockFCNetwork(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.layer = nn.Linear(obs_size + act_size, obs_size)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, act), dim=-1)
        x = self.layer(x)
        return x


def get_mock_fc_batch(
    batch_size: int,
    obs_size: int,
    act_size: int,
):
    obs = torch.zeros((batch_size, obs_size))
    act = torch.zeros((batch_size, act_size))
    new_obs = torch.zeros((batch_size, obs_size))
    return dict(obs=obs, actions=act, new_obs=new_obs)
