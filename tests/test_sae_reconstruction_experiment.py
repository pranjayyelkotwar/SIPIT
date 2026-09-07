import argparse
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments.evaluate_sae_reconstruction import (  # noqa: E402
    SAERoundTripHook,
    compare_teacher_forced_logits,
    load_prompts,
    output_similarity,
    reconstruction_metrics,
)


class _FakeSAE(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def encode(self, hidden):
        return hidden * 2

    def decode(self, features):
        return features + 1


def test_exact_reconstruction_metrics_are_ideal():
    hidden = torch.randn(1, 4, 8)
    features = torch.zeros(1, 4, 16)
    features[..., :3] = 1

    metrics = reconstruction_metrics(hidden, hidden.clone(), features)

    assert metrics["all_cosine"] == pytest.approx(1.0)
    assert metrics["all_relative_l2"] == pytest.approx(0.0)
    assert metrics["all_explained_variance"] == pytest.approx(1.0)
    assert metrics["mean_l0"] == pytest.approx(3.0)


def test_output_similarity_counts_prefix_and_positions():
    clean = torch.tensor([[1, 2, 3, 4]])
    reconstructed = torch.tensor([[1, 2, 9, 4]])

    metrics = output_similarity(clean, reconstructed)

    assert not metrics["exact_token_match"]
    assert metrics["common_prefix_tokens"] == 2
    assert metrics["position_match_rate"] == pytest.approx(0.75)


def test_identical_logits_have_zero_kl_and_full_agreement():
    logits = torch.randn(1, 3, 11)
    targets = torch.tensor([[1, 2, 3]])

    metrics = compare_teacher_forced_logits(logits, logits.clone(), targets)

    assert metrics["mean_kl_clean_to_sae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["top1_agreement_rate"] == pytest.approx(1.0)
    assert metrics["first_token_top1_agrees"]


def test_sae_hook_replaces_module_output():
    block = torch.nn.Identity()
    hidden = torch.randn(1, 3, 5)

    with SAERoundTripHook(block, _FakeSAE(), record_first_call=True) as hook:
        output = block(hidden)

    torch.testing.assert_close(output, hidden * 2 + 1)
    assert hook.calls == 1
    assert hook.first_call_metrics is not None


def test_bundled_sample_contains_five_prompts():
    prompts_path = (
        Path(__file__).parents[1] / "experiments" / "sae_reconstruction_prompts.json"
    )
    args = argparse.Namespace(
        num_prompts=5,
        prompts_file=prompts_path,
        shuffle=False,
        seed=42,
    )

    prompts = load_prompts(args)

    assert len(prompts) == 5
    assert prompts[0]["source_id"] == "seasons-explanation"
    assert all("sentence" in prompt["prompt"].lower() for prompt in prompts)
