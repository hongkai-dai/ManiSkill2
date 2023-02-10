import pytest
import torch

from mani_skill2.dynamics.networks import UnetFiLM


class TestUnetFiLM:
    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize("cond_size", (1, 3))
    @pytest.mark.parametrize("n_channels", (1, 4))
    @pytest.mark.parametrize(
        "height,width",
        [
            (32, 32),
            (64, 64),
        ],
    )
    def test_forward(self, batch_size, cond_size, n_channels, height, width):
        model = UnetFiLM(n_channels=n_channels, cond_size=cond_size)
        x = torch.zeros((batch_size, n_channels, height, width))
        cond = torch.zeros((batch_size, cond_size))
        output = model.forward(x, cond)
        assert output.shape == (batch_size, n_channels, height, width)

    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize(
        "height,width",
        [
            (32, 32),
            (64, 64),
        ],
    )
    def test_forward_without_channel(
        self,
        batch_size,
        height,
        width,
        cond_size=2,
        n_channels=1,
    ):
        model = UnetFiLM(n_channels=n_channels, cond_size=cond_size)
        # Formulate input without the channel dim.
        x = torch.zeros((batch_size, height, width))
        cond = torch.zeros((batch_size, cond_size))
        output = model.forward(x, cond)
        assert output.shape == (batch_size, height, width)
