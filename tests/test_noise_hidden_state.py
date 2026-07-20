import torch

from tests.noise_hidden_state import add_relative_gaussian_noise


def test_relative_noise_is_reproducible_and_scaled():
    hidden = torch.full((64, 64), 2.0)
    noisy_a, std_a = add_relative_gaussian_noise(
        hidden, 0.1, torch.Generator().manual_seed(7)
    )
    noisy_b, std_b = add_relative_gaussian_noise(
        hidden, 0.1, torch.Generator().manual_seed(7)
    )

    assert std_a == std_b == 0.2
    assert torch.equal(noisy_a, noisy_b)
    assert abs((noisy_a - hidden).square().mean().sqrt().item() - 0.2) < 0.01


def test_zero_noise_preserves_values():
    hidden = torch.randn(3, 5, dtype=torch.float16)
    noisy, std = add_relative_gaussian_noise(hidden, 0, torch.Generator())
    assert std == 0
    assert torch.equal(noisy, hidden.float())
