from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from grounding import FeatureScorer
from llamascope import tensor_for_sae


@dataclass
class SearchConfig:
    num_candidates: int = 16
    min_active: int = 1
    max_active: int = 3
    active_topk: int = 8
    beta: float = 0.01
    max_iters: int = 10
    min_improvement: float = 1e-4
    patience: int = 3
    target_score: float | None = None
    clamp_nonnegative: bool = True
    accept_non_improving: bool = False
    seed: int = 0
    eps: float = 1e-8


@dataclass
class SearchStep:
    step: int
    score: float
    latent_indices: list[int]
    coefficients: list[float]
    improved: bool


@dataclass
class SearchResult:
    original_hidden_state: torch.Tensor
    reconstructed_hidden_state: torch.Tensor
    original_features: torch.Tensor
    final_features: torch.Tensor
    history: list[SearchStep]

    def metadata(self) -> dict[str, Any]:
        best_score = min(step.score for step in self.history)
        return {
            "initial_score": self.history[0].score,
            "final_score": best_score,
            "steps": [asdict(step) for step in self.history],
        }


def _scalar_score(
    scorer: FeatureScorer, features: torch.Tensor, reconstruction: torch.Tensor
) -> float:
    score = scorer(features, reconstruction)
    if score.numel() != 1:
        raise ValueError("The scorer must return exactly one score per hidden state.")
    return float(score.item())


def perturb_hidden_state(
    hidden_state: torch.Tensor,
    *,
    sae,
    scorer: FeatureScorer,
    config: SearchConfig | None = None,
) -> SearchResult:
    """Search LlamaScope latent space and return the decoded hidden state.

    `hidden_state` must be a single vector shaped `(d_model,)` or `(1, d_model)`.
    Lower scorer values are considered better.
    """
    config = config or SearchConfig()
    if hidden_state.ndim == 1:
        hidden = hidden_state.unsqueeze(0)
    elif hidden_state.ndim == 2 and hidden_state.shape[0] == 1:
        hidden = hidden_state
    else:
        raise ValueError(
            f"Expected (d_model,) or (1, d_model), got {tuple(hidden_state.shape)}"
        )
    if config.num_candidates < 1 or config.max_iters < 0:
        raise ValueError("num_candidates must be positive and max_iters nonnegative.")

    hidden = tensor_for_sae(hidden, sae)
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(config.seed)

    with torch.no_grad():
        original_features = sae.encode(hidden)
        current_features = original_features.clone()
        current_reconstruction = sae.decode(current_features)
        current_score = _scalar_score(
            scorer, current_features, current_reconstruction
        )

    history = [SearchStep(0, current_score, [], [], True)]
    best_features = current_features.clone()
    best_reconstruction = current_reconstruction.clone()
    best_score = current_score
    stale_steps = 0

    for step in range(1, config.max_iters + 1):
        if config.target_score is not None and best_score <= config.target_score:
            break

        k_top = min(config.active_topk, current_features.shape[-1])
        active = torch.topk(current_features.abs(), k=k_top, dim=-1).indices[0]
        max_active = min(config.max_active, active.numel())
        min_active = min(config.min_active, max_active)
        if min_active < 1:
            break

        # beta is relative to the RMS magnitude of the currently active features.
        active_values = current_features[0, active]
        sigma = config.beta * (
            active_values.square().mean().sqrt().item() + config.eps
        )
        candidate_best: tuple[
            float, torch.Tensor, torch.Tensor, list[int], list[float]
        ] | None = None

        for _ in range(config.num_candidates):
            count = int(
                torch.randint(
                    min_active,
                    max_active + 1,
                    (1,),
                    generator=generator,
                    device=hidden.device,
                ).item()
            )
            chosen = active[
                torch.randperm(
                    active.numel(), generator=generator, device=hidden.device
                )[:count]
            ]
            coefficients = (
                torch.randn(
                    count,
                    generator=generator,
                    device=hidden.device,
                    dtype=current_features.dtype,
                )
                * sigma
            )
            candidate_features = current_features.clone()
            candidate_features[0, chosen] += coefficients
            if config.clamp_nonnegative:
                candidate_features.clamp_(min=0)

            with torch.no_grad():
                reconstruction = sae.decode(candidate_features)
                score = _scalar_score(scorer, candidate_features, reconstruction)

            if candidate_best is None or score < candidate_best[0]:
                candidate_best = (
                    score,
                    candidate_features,
                    reconstruction,
                    chosen.tolist(),
                    coefficients.float().tolist(),
                )

        if candidate_best is None:
            break
        score, features, reconstruction, indices, coefficients = candidate_best
        improved = score < best_score - config.min_improvement
        accepted = improved or config.accept_non_improving

        if accepted:
            current_features = features
            current_reconstruction = reconstruction
            current_score = score
        if improved:
            best_features = features.clone()
            best_reconstruction = reconstruction.clone()
            best_score = score
            stale_steps = 0
        else:
            stale_steps += 1

        history.append(
            SearchStep(step, current_score, indices, coefficients, improved)
        )
        if stale_steps >= config.patience:
            break

    return SearchResult(
        original_hidden_state=hidden.detach().cpu().squeeze(0),
        reconstructed_hidden_state=best_reconstruction.detach().cpu().squeeze(0),
        original_features=original_features.detach().cpu().squeeze(0),
        final_features=best_features.detach().cpu().squeeze(0),
        history=history,
    )
