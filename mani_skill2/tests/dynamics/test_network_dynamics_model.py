import pytest
import torch

from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.utils.testing.dynamics_test_utils import MockFCNetwork


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("state_size", (1, 3))
@pytest.mark.parametrize("act_size", (1, 2))
@pytest.mark.parametrize("is_residual", (True, False))
def test_network_dynamics_model_step(batch_size, state_size, act_size, is_residual):
    net = MockFCNetwork(state_size, act_size)
    dyn = NetworkDynamicsModel(
        net,
        is_residual=is_residual,
    )

    state = torch.zeros((batch_size, state_size))
    act = torch.zeros((batch_size, act_size))

    next_state, terminated, info = dyn.step(state, act)
    assert next_state.shape == (batch_size, state_size)
