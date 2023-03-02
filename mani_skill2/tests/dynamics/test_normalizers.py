import pytest
import torch


from mani_skill2.dynamics.normalizers import HeightMapNormalizer


def _check_normalizer_round_trip(normalizer, x):
    x_norm = normalizer.normalize(x)
    x_denorm = normalizer.denormalize(x_norm)
    torch.testing.assert_close(x, x_denorm)


@pytest.mark.parametrize("scale", (1.0, 10.0))
@pytest.mark.parametrize("batch_size", (0, 1, 2))
def test_height_map_normalizer(scale, batch_size, height=32, width=32):
    normalizer = HeightMapNormalizer(scale)
    shape = [height, width]
    if batch_size > 0:
        shape = [batch_size] + shape
    x = torch.ones(shape)
    _check_normalizer_round_trip(normalizer, x)
