import pytest
import numpy as np
import torch


from mani_skill2.dynamics.normalizers import HeightMapNormalizer, AffineNormalizer


def _check_normalizer_round_trip(normalizer, x):
    x_norm = normalizer.normalize(x)
    x_denorm = normalizer.denormalize(x_norm)
    torch.testing.assert_close(x, x_denorm)
    if normalizer.heightmap_min is not None:
        assert torch.all(x_norm >= normalizer.heightmap_min * normalizer.scale)


@pytest.mark.parametrize("scale", (1.0, 10.0))
@pytest.mark.parametrize("batch_size", (0, 1, 2))
@pytest.mark.parametrize("heightmap_min", (None, 0.1))
def test_height_map_normalizer(scale, batch_size, heightmap_min, height=32, width=32):
    normalizer = HeightMapNormalizer(scale, heightmap_min)
    shape = [height, width]
    if batch_size > 0:
        shape = [batch_size] + shape
    x = torch.ones(shape)
    _check_normalizer_round_trip(normalizer, x)


class TestAffineNormalizer:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_to_device(self, device):
        to_device = "cuda" if device == "cpu" else "cpu"
        data_shape = (2,)
        lo = torch.tensor([-2, -5.0], device=device)
        up = torch.tensor([3, 4.0], device=device)
        dut = AffineNormalizer(data_shape, lo, up)
        dut.to(to_device)
        assert dut.lo.device.type == to_device
        assert dut.up.device.type == to_device

    def test_empty_bounds(self):
        data_shape = (2,)
        dut = AffineNormalizer(data_shape, lo=None, up=None)
        assert isinstance(dut.lo, torch.Tensor)
        assert isinstance(dut.up, torch.Tensor)
        assert isinstance(dut.scale_flag, torch.Tensor)
        assert (dut.lo.shape == data_shape)
        assert (dut.up.shape == data_shape)
        assert (dut.scale_flag.shape == data_shape)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_normalize_denormalize(self, device):
        data_shape = (2,)
        lo = torch.tensor([-2, -5.0])
        up = torch.tensor([3, 4.0])
        dut = AffineNormalizer(data_shape, lo, up)
        dut.to(device)

        np.testing.assert_allclose(
            dut.normalize(lo.to(device)).cpu().detach().numpy(), np.array([-1.0, -1.0])
        )
        np.testing.assert_allclose(
            dut.normalize(up.to(device)).cpu().detach().numpy(), np.array([1.0, 1.0])
        )
        np.testing.assert_allclose(
            dut.denormalize(torch.tensor([-1.0, -1], device=device))
            .cpu()
            .detach()
            .numpy(),
            lo.cpu().detach().numpy(),
        )
        np.testing.assert_allclose(
            dut.denormalize(torch.tensor([1.0, 1], device=device))
            .cpu()
            .detach()
            .numpy(),
            up.cpu().detach().numpy(),
        )

        x = torch.tensor([[-3, 5.0], [1, 2], [-3, 7]], device=device)
        x_normalized = dut.normalize(x)
        x_denormalized = dut.denormalize(x_normalized)
        np.testing.assert_allclose(
            x_denormalized.cpu().detach().numpy(), x.cpu().detach().numpy()
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_unscaled(self, device):
        """
        Some entries in lo is the same as up, so we do not scale these entries.
        """
        data_shape = (3,)
        lo = torch.tensor([0, 2, 3.0])
        up = torch.tensor([0, 5.0, 3.0])
        dut = AffineNormalizer(data_shape, lo, up)

        x = torch.tensor([[2, 5, 1], [0, 2, 2.0]], device=device)
        dut.to(device)
        x_normalized = dut.normalize(x)
        np.testing.assert_allclose(
            x_normalized[:, 0].cpu().detach().numpy(), x[:, 0].cpu().detach().numpy()
        )
        np.testing.assert_allclose(
            x_normalized[:, 2].cpu().detach().numpy(), x[:, 2].cpu().detach().numpy()
        )
        np.testing.assert_allclose(
            x_normalized[:, 1].cpu().detach().numpy(), np.array([1, -1.0])
        )
        x_denormalized = dut.denormalize(x_normalized)
        np.testing.assert_allclose(
            x_denormalized.cpu().detach().numpy(), x.cpu().detach().numpy()
        )

    def test_from_data(self):
        data = torch.tensor([[-3.0, 2.0], [-1, 5], [0, 4], [1, 6], [2, 9]])
        dut1 = AffineNormalizer.from_data(data, percentile=0.0)
        np.testing.assert_allclose(dut1.lo.cpu().detach().numpy(), np.array([-3, 2.0]))
        np.testing.assert_allclose(dut1.up.cpu().detach().numpy(), np.array([2.0, 9.0]))
        dut2 = AffineNormalizer.from_data(data, percentile=0.2)
        np.testing.assert_allclose(dut2.lo.cpu().detach().numpy(), np.array([-1, 4.0]))
        np.testing.assert_allclose(dut2.up.cpu().detach().numpy(), np.array([1.0, 6.0]))
