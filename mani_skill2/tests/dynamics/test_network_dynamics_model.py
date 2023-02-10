import pytest
import torch

from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.utils.testing.dynamics_test_utils import MockFCNetwork


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("obs_size", (1, 3))
@pytest.mark.parametrize("act_size", (1, 2))
@pytest.mark.parametrize("is_residual", (True, False))
def test_network_dynamics_model_step(batch_size, obs_size, act_size, is_residual):
    net = MockFCNetwork(obs_size, act_size)
    dyn = NetworkDynamicsModel(
        net,
        is_residual=is_residual,
    )

    obs = torch.zeros((batch_size, obs_size))
    act = torch.zeros((batch_size, act_size))

    next_obs, info = dyn.step(obs, act)
    assert next_obs.shape == (batch_size, obs_size)
