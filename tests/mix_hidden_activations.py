#!/usr/bin/env python3
"""Mix an original hidden-state prefix with a reconstructed suffix."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def load_bundle(path: Path) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not torch.is_tensor(bundle.get("hidden_states")):
        raise TypeError(
            f"{path} must contain a dictionary with a tensor named 'hidden_states'."
        )
    return bundle


def mix_bundles(original: dict, reconstructed: dict, original_fraction: float) -> dict:
    """Return the reconstructed bundle with an original hidden-state prefix."""
    if not math.isfinite(original_fraction) or not 0.0 <= original_fraction <= 1.0:
        raise ValueError("--original-fraction must be between 0.0 and 1.0.")

    original_hidden = original["hidden_states"]
    reconstructed_hidden = reconstructed["hidden_states"]

    if original_hidden.ndim not in (2, 3):
        raise ValueError(
            "Expected hidden_states shaped [tokens, hidden] or "
            f"[batch, tokens, hidden], got {tuple(original_hidden.shape)}."
        )
    if original_hidden.shape != reconstructed_hidden.shape:
        raise ValueError(
            "Original and reconstructed hidden-state shapes differ: "
            f"{tuple(original_hidden.shape)} vs {tuple(reconstructed_hidden.shape)}."
        )
    if original_hidden.dtype != reconstructed_hidden.dtype:
        raise ValueError(
            "Original and reconstructed hidden-state dtypes differ: "
            f"{original_hidden.dtype} vs {reconstructed_hidden.dtype}."
        )

    token_count = original_hidden.shape[-2]
    original_token_count = int(token_count * original_fraction)

    mixed_hidden = reconstructed_hidden.clone()
    mixed_hidden[..., :original_token_count, :] = original_hidden[
        ..., :original_token_count, :
    ]

    # Preserve every reconstructed-bundle field and replace only hidden_states.
    output = dict(reconstructed)
    output["hidden_states"] = mixed_hidden
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the first fraction of tokens from an original hidden-state "
            "bundle and the remaining tokens from a reconstructed bundle."
        )
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--original-fraction",
        type=float,
        required=True,
        help=(
            "Fraction in [0, 1] to take from the start of the original tensor. "
            "The token count is rounded down."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    original = load_bundle(args.original)
    reconstructed = load_bundle(args.reconstructed)
    output = mix_bundles(original, reconstructed, args.original_fraction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    token_count = output["hidden_states"].shape[-2]
    original_token_count = int(token_count * args.original_fraction)
    print(
        f"Saved {args.output}: first {original_token_count}/{token_count} tokens "
        "from original, remainder from reconstructed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
