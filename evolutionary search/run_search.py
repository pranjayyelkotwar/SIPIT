#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from grounding import (
    GroundingScoreCalculator,
    load_perplexity_regression_weights,
)
from llamascope import DEFAULT_RELEASE, DEFAULT_WIDTH, load_llamascope_sae
from search import SearchConfig, perturb_hidden_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perturb a hidden state in LlamaScope SAE latent space."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-key", default="hidden_states")
    parser.add_argument("--token-pos", type=int, default=-1)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    scores = parser.add_mutually_exclusive_group(required=True)
    scores.add_argument("--avg-latents-dir", type=Path)
    scores.add_argument("--perplexity-weights", type=Path)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--min-active", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=3)
    parser.add_argument("--active-topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target-score", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-negative-features", action="store_true")
    parser.add_argument("--accept-non-improving", action="store_true")
    return parser.parse_args()


def load_input(path: Path, key: str) -> tuple[object, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        return payload, payload
    if not isinstance(payload, dict):
        raise TypeError("Input must be a tensor or a dictionary bundle.")
    if key not in payload or not torch.is_tensor(payload[key]):
        raise KeyError(f"Bundle does not contain tensor key {key!r}.")
    return payload, payload[key]


def select_hidden(states: torch.Tensor, token_pos: int) -> tuple[torch.Tensor, int]:
    if states.ndim == 1:
        return states, 0
    if states.ndim == 2:
        pos = token_pos if token_pos >= 0 else states.shape[0] - 1
        if not 0 <= pos < states.shape[0]:
            raise IndexError(f"token-pos {token_pos} is outside sequence length.")
        return states[pos], pos
    if states.ndim == 3 and states.shape[0] == 1:
        pos = token_pos if token_pos >= 0 else states.shape[1] - 1
        if not 0 <= pos < states.shape[1]:
            raise IndexError(f"token-pos {token_pos} is outside sequence length.")
        return states[0, pos], pos
    raise ValueError("Hidden states must have shape (d), (seq,d), or (1,seq,d).")


def main() -> None:
    args = parse_args()
    payload, states = load_input(args.input, args.input_key)
    hidden, resolved_pos = select_hidden(states, args.token_pos)
    loaded = load_llamascope_sae(
        layer=args.layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )
    sae = loaded.sae

    if args.avg_latents_dir is not None:
        scorer = GroundingScoreCalculator.from_avg_latents(
            args.avg_latents_dir, sae.device, sae.dtype
        )
    else:
        scorer = load_perplexity_regression_weights(
            args.perplexity_weights,
            device=sae.device,
            dtype=torch.float32,
        )

    config = SearchConfig(
        num_candidates=args.num_candidates,
        min_active=args.min_active,
        max_active=args.max_active,
        active_topk=args.active_topk,
        beta=args.beta,
        max_iters=args.max_iters,
        min_improvement=args.min_improvement,
        patience=args.patience,
        target_score=args.target_score,
        clamp_nonnegative=not args.allow_negative_features,
        accept_non_improving=args.accept_non_improving,
        seed=args.seed,
    )
    result = perturb_hidden_state(hidden, sae=sae, scorer=scorer, config=config)

    output_states = states.clone()
    if output_states.ndim == 1:
        output_states = result.reconstructed_hidden_state
    elif output_states.ndim == 2:
        output_states[resolved_pos] = result.reconstructed_hidden_state.to(
            output_states.dtype
        )
    else:
        output_states[0, resolved_pos] = result.reconstructed_hidden_state.to(
            output_states.dtype
        )

    if isinstance(payload, dict):
        output = dict(payload)
        output[args.input_key] = output_states
        output["evolutionary_search"] = {
            **result.metadata(),
            "layer": args.layer,
            "token_pos": resolved_pos,
            "release": loaded.release,
            "sae_id": loaded.sae_id,
            "original_features": result.original_features,
            "final_features": result.final_features,
        }
    else:
        output = output_states

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    result_metadata = result.metadata()
    print(
        f"Saved {args.output} | layer={args.layer} token={resolved_pos} "
        f"score={result_metadata['initial_score']:.6g}"
        f"->{result_metadata['final_score']:.6g}"
    )


if __name__ == "__main__":
    main()
