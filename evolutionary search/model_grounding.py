"""Model-dependent grounding functions ported from ``sae_exp_1.2``.

Unlike SAE and regression grounding, these functions require a language model,
prompt tokens, and a complete activation override. They are kept outside the
single-hidden-state search API to make that dependency explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


LogitsFunction = Callable[
    [object, list[int], int, torch.Tensor, int], torch.Tensor
]


@dataclass
class StabilityConfig:
    topk: int = 20
    eps: float = 1e-8


def compute_fisher_diag(
    model,
    prompt_tokens: list[int],
    override_layer: int,
    override_activations: torch.Tensor,
    token_pos: int,
    config: StabilityConfig | None = None,
) -> torch.Tensor:
    """Compute the original top-k diagonal Fisher approximation.

    The model must implement ``forward_with_activation_override_grad`` with the
    same contract as the model in ``sae_exp_1.2``.
    """
    config = config or StabilityConfig()
    if override_activations.ndim != 3 or override_activations.shape[0] != 1:
        raise ValueError("override_activations must have shape (1, seq_len, d_model)")
    if override_activations.shape[1] != len(prompt_tokens):
        raise ValueError("Prompt length and override sequence length must match.")

    seq_len = len(prompt_tokens)
    token_pos = seq_len - 1 if token_pos < 0 else min(token_pos, seq_len - 1)
    device = next(model.parameters()).device
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    override = override_activations.detach().clone().requires_grad_(True)
    logits = model.forward_with_activation_override_grad(
        tokens,
        start_pos=0,
        override_layer=override_layer,
        override_activations=override,
    )[:, token_pos].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    top_log_probs, _ = torch.topk(
        log_probs, k=min(config.topk, log_probs.shape[-1]), dim=-1
    )
    fisher = torch.zeros_like(override[:, token_pos], dtype=torch.float32)
    for index in range(top_log_probs.shape[-1]):
        log_probability = top_log_probs[:, index].sum()
        gradient = torch.autograd.grad(
            log_probability, override, retain_graph=True
        )[0][:, token_pos].float()
        fisher += top_log_probs[:, index].exp().unsqueeze(-1) * gradient.square()
    return fisher.squeeze(0)


def score_stability_delta(
    delta: torch.Tensor, fisher_diag: torch.Tensor
) -> torch.Tensor:
    return -torch.sum(fisher_diag * delta.square())


@dataclass
class PseudoCurvConfig:
    topk_vocab: int = 50
    mc_samples: int = 4
    min_active: int = 1
    max_active: int = 3
    beta: float = 0.01
    eps: float = 1e-8
    latent_topk: int = 8
    normalize_entropy: bool = True


def compute_pseudo_curv(
    *,
    model,
    prompt_tokens: list[int],
    override_layer: int,
    base_override_activations: torch.Tensor,
    token_pos: int,
    features: torch.Tensor,
    sae,
    logits_fn: LogitsFunction,
    config: PseudoCurvConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate the original expected-KL pseudo-curvature with LlamaScope.

    ``logits_fn(model, tokens, layer, overrides, token_pos)`` is an adapter for
    whichever Llama implementation owns the activation-override operation.
    Candidate perturbations occur in SAE feature space and are decoded through
    LlamaScope, rather than using the former custom SAE decoder matrix.
    """
    config = config or PseudoCurvConfig()
    if base_override_activations.ndim != 3:
        raise ValueError("base_override_activations must be (1, seq_len, d_model)")
    if base_override_activations.shape[0] != 1:
        raise ValueError("Pseudo-curvature currently supports one sequence.")
    seq_len = base_override_activations.shape[1]
    if seq_len != len(prompt_tokens):
        raise ValueError("Prompt length and override sequence length must match.")
    token_pos = seq_len - 1 if token_pos < 0 else min(token_pos, seq_len - 1)

    base_logits = logits_fn(
        model,
        prompt_tokens,
        override_layer,
        base_override_activations,
        token_pos,
    ).float()
    vocab_k = min(config.topk_vocab, base_logits.shape[-1])
    base_top_logits, top_indices = torch.topk(base_logits, k=vocab_k, dim=-1)
    base_log_probs = torch.log_softmax(base_top_logits, dim=-1)
    base_probs = base_log_probs.exp()
    entropy = -(base_probs * base_log_probs).sum(dim=-1)

    features = features.reshape(1, -1)
    latent_k = min(config.latent_topk, features.shape[-1])
    active = torch.topk(features.abs(), k=latent_k, dim=-1).indices[0]
    active_rms = features[0, active].square().mean().sqrt().item()
    sigma = config.beta * (active_rms + config.eps)
    reconstructed = []
    for _ in range(config.mc_samples):
        count = int(
            torch.randint(
                min(config.min_active, active.numel()),
                min(config.max_active, active.numel()) + 1,
                (1,),
                device=features.device,
                generator=generator,
            ).item()
        )
        chosen = active[
            torch.randperm(
                active.numel(), device=features.device, generator=generator
            )[:count]
        ]
        candidate = features.clone()
        candidate[0, chosen] += (
            torch.randn(
                count,
                device=features.device,
                dtype=features.dtype,
                generator=generator,
            )
            * sigma
        )
        candidate.clamp_(min=0)
        reconstructed.append(sae.decode(candidate)[0])

    override_batch = base_override_activations.repeat(config.mc_samples, 1, 1)
    override_batch[:, token_pos] = torch.stack(reconstructed).to(
        override_batch.dtype
    )
    perturbed_logits = logits_fn(
        model, prompt_tokens, override_layer, override_batch, token_pos
    ).float()
    selected = torch.gather(
        perturbed_logits,
        -1,
        top_indices.repeat(config.mc_samples, 1),
    )
    perturbed_log_probs = torch.log_softmax(selected, dim=-1)
    kl = (
        base_probs.repeat(config.mc_samples, 1)
        * (base_log_probs.repeat(config.mc_samples, 1) - perturbed_log_probs)
    ).sum(dim=-1)
    if config.normalize_entropy:
        kl = kl / (entropy.repeat(config.mc_samples) + config.eps)
    return -kl.mean()

