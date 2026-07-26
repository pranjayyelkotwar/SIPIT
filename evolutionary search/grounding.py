from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


class FeatureScorer(Protocol):
    def __call__(
        self, features: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor: ...


class GroundingScoreCalculator:
    """The SAE grounding calculation used by sae_exp_1.2."""

    def __init__(self, c_vector: torch.Tensor, eps: float = 1e-6) -> None:
        self.c_vector = c_vector
        self.eps = eps

    @classmethod
    def from_avg_latents(
        cls,
        avg_latents_dir: Path,
        device: torch.device | str,
        dtype: torch.dtype,
        eps: float = 1e-6,
    ) -> "GroundingScoreCalculator":
        arc_path = avg_latents_dir / "avg_latents_arc_easy.pt"
        hle_path = avg_latents_dir / "avg_latents_hle.pt"
        if not arc_path.exists() or not hle_path.exists():
            raise FileNotFoundError(
                "Expected avg_latents_arc_easy.pt and avg_latents_hle.pt"
            )
        arc = torch.load(arc_path, map_location="cpu", weights_only=True).to(
            device=device, dtype=dtype
        )
        hle = torch.load(hle_path, map_location="cpu", weights_only=True).to(
            device=device, dtype=dtype
        )
        if arc.shape != hle.shape:
            raise ValueError(f"ARC {tuple(arc.shape)} != HLE {tuple(hle.shape)}")
        return cls(arc - hle, eps)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.c_vector.shape[-1]:
            raise ValueError(
                f"Grounding vector has {self.c_vector.shape[-1]} features, "
                f"but LlamaScope produced {features.shape[-1]}. Recompute the "
                "average latents with this exact LlamaScope SAE."
            )
        return (self.c_vector * features).sum(dim=-1) / (
            features.sum(dim=-1) + self.eps
        )

    def __call__(
        self, features: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        del reconstructed
        return self.score(features)


@dataclass
class PerplexityRegressionWeights:
    """The perplexity regression calculation used by sae_exp_1.2."""

    weights: torch.Tensor
    bias: float
    target_mean: float
    target_std: float
    target_key: str
    normalized: bool = True
    layer: int | None = None
    token_pos: int | None = None

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.weights.shape[-1]:
            raise ValueError(
                f"Regression has {self.weights.shape[-1]} weights, but LlamaScope "
                f"produced {features.shape[-1]}. Train weights for this SAE."
            )
        values = features.to(dtype=self.weights.dtype, device=self.weights.device)
        prediction = (values * self.weights).sum(dim=-1) + self.bias
        if self.normalized and self.target_std != 0.0:
            prediction = prediction * self.target_std + self.target_mean
        return prediction

    def __call__(
        self, features: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        del reconstructed
        return self.predict(features)


def load_perplexity_regression_weights(
    path: Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> PerplexityRegressionWeights:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return PerplexityRegressionWeights(
        weights=payload["weights"].to(device=device, dtype=dtype),
        bias=float(payload.get("bias", 0.0)),
        target_mean=float(payload.get("target_mean", 0.0)),
        target_std=float(payload.get("target_std", 1.0)),
        target_key=str(payload.get("target_key", "perplexity")),
        normalized=bool(payload.get("normalized", True)),
        layer=payload.get("layer"),
        token_pos=payload.get("token_pos"),
    )


def save_perplexity_regression_weights(
    weights: PerplexityRegressionWeights, path: Path
) -> None:
    payload = {
        "weights": weights.weights.detach().cpu(),
        "bias": float(weights.bias),
        "target_mean": float(weights.target_mean),
        "target_std": float(weights.target_std),
        "target_key": weights.target_key,
        "normalized": weights.normalized,
        "layer": weights.layer,
        "token_pos": weights.token_pos,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def score_stability_delta(
    delta: torch.Tensor, fisher_diag: torch.Tensor
) -> torch.Tensor:
    """Unchanged Fisher-diagonal stability score from sae_exp_1.2."""
    return -torch.sum(fisher_diag * delta.square())
