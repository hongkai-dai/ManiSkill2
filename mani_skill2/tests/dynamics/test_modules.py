import pytest

from mani_skill2.dynamics.modules import DynamicsPLModule
from mani_skill2.dynamics.network_dynamics_model import NetworkDynamicsModel
from mani_skill2.utils.testing.dynamics_test_utils import (
    MockFCNetwork,
    get_mock_fc_batch,
)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDynamicsModelTrainer:
    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize("obs_size", (3,))
    @pytest.mark.parametrize("act_size", (2,))
    def test_steps(self, batch_size, obs_size, act_size):
        net = MockFCNetwork(obs_size, act_size)
        dyn = NetworkDynamicsModel(net, is_residual=True)
        module = DynamicsPLModule(dyn, lr=1e-4)
        batch = get_mock_fc_batch(batch_size, obs_size, act_size)
        loss = module.training_step(batch, 0)
        assert loss is not None
        loss = module.validation_step(batch, 0)
        assert loss is not None
        
