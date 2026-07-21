import pytest
import torch

from src.algorithm import CosineRMSESIPIT
from src.commands.invert_hidden_state_approx import build_parser


def _algorithm(min_cosine=0.78, max_rmse=0.22):
    return CosineRMSESIPIT(
        log_dir='logs',
        log_name=None,
        min_cosine_similarity=min_cosine,
        max_rmse=max_rmse,
    )


def test_requires_both_cosine_and_rmse_thresholds():
    algorithm = _algorithm(min_cosine=0.99, max_rmse=0.2)
    target = torch.tensor([1.0, 0.0])

    assert algorithm.is_match(torch.tensor([1.1, 0.0]), target)
    assert not algorithm.is_match(torch.tensor([2.0, 0.0]), target)
    assert not algorithm.is_match(torch.tensor([0.9, 0.2]), target)


def test_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        _algorithm(min_cosine=1.1)
    with pytest.raises(ValueError):
        _algorithm(max_rmse=-0.1)


def test_parser_defaults_follow_sae_non_bos_baseline():
    args = build_parser().parse_args(['--input', 'in.pt', '--output', 'out.txt'])
    assert args.min_cosine_similarity == 0.78
    assert args.max_rmse == 0.22
    assert args.skip_target_tokens == 0
