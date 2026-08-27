#!/usr/bin/env python3
"""Find which tokens in one question activate selected SAE features.

The input is a per-question residual-stream tensor written by
``activation_capture/capture_activations.py``.  Every token state is passed
through the same LlamaScope SAE used elsewhere in this repository, then token
positions are ranked separately for each requested feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EVOLUTIONARY_SEARCH_DIR = REPO_ROOT / "evolutionary search"
if str(EVOLUTIONARY_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(EVOLUTIONARY_SEARCH_DIR))

# Reuse the existing activation-tensor normalization and LlamaScope loader.
from build_artifacts import flatten_states, load_hidden_tensor  # noqa: E402
from llamascope import (  # noqa: E402
    DEFAULT_RELEASE,
    DEFAULT_WIDTH,
    load_llamascope_sae,
)


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode every token state for one captured question and rank the "
            "tokens that activate selected SAE features."
        )
    )
    parser.add_argument("--activation", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Exact prompt used for activation capture.")
    prompt.add_argument(
        "--prompt-file", type=Path, help="UTF-8 file containing the exact prompt."
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        action="append",
        required=True,
        help="SAE feature to inspect; repeat to inspect multiple features.",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tokenizer model ID.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_outputs/exp2/token_feature_rankings.csv"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--top-k-tokens", type=int, default=20)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=4,
        help="Number of neighboring token pieces shown on each side.",
    )
    parser.add_argument(
        "--add-bos-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must match the activation-capture tokenization setting.",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    return args.prompt_file.read_text(encoding="utf-8")


def token_context(token_pieces: list[str], position: int, radius: int) -> str:
    start = max(0, position - radius)
    stop = min(len(token_pieces), position + radius + 1)
    pieces = token_pieces[start:stop]
    pieces[position - start] = f"[{pieces[position - start]}]"
    return "".join(pieces)


def encode_selected_features(
    states: torch.Tensor,
    sae,
    feature_ids: list[int],
    chunk_size: int,
) -> torch.Tensor:
    selected_chunks = []
    for chunk in states.split(chunk_size):
        with torch.inference_mode():
            encoded = sae.encode(chunk.to(device=sae.device, dtype=sae.dtype))
        if max(feature_ids) >= encoded.shape[-1]:
            raise IndexError(
                f"Feature ID {max(feature_ids)} is outside SAE width "
                f"{encoded.shape[-1]}."
            )
        selected_chunks.append(encoded[:, feature_ids].detach().cpu().float())
    return torch.cat(selected_chunks, dim=0)


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if args.top_k_tokens < 1:
        raise ValueError("--top-k-tokens must be at least 1.")
    if args.context_tokens < 0:
        raise ValueError("--context-tokens cannot be negative.")
    if min(args.feature_id) < 0:
        raise ValueError("--feature-id cannot be negative.")
    if not args.activation.exists():
        raise FileNotFoundError(f"Activation tensor not found: {args.activation}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    prompt = read_prompt(args)
    states = flatten_states(load_hidden_tensor(args.activation)).cpu()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    full_token_ids = tokenizer.encode(
        prompt, add_special_tokens=args.add_bos_token, truncation=False
    )
    token_ids = full_token_ids[: states.shape[0]]
    if len(full_token_ids) > states.shape[0]:
        logging.info(
            "Matched the capture by truncating %d prompt tokens to %d positions",
            len(full_token_ids),
            states.shape[0],
        )
    if len(token_ids) != states.shape[0]:
        raise ValueError(
            "Prompt tokenization does not match the activation tensor: "
            f"tokenizer produced {len(token_ids)} tokens but the tensor has "
            f"{states.shape[0]} positions. Use the exact captured prompt and "
            "matching --add-bos-token/--no-add-bos-token setting."
        )
    token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    loaded = load_llamascope_sae(
        layer=args.layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )
    feature_ids = list(dict.fromkeys(args.feature_id))
    activations = encode_selected_features(
        states, loaded.sae, feature_ids, args.chunk_size
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for column, feature_id in enumerate(feature_ids):
        values, positions = torch.topk(
            activations[:, column], k=min(args.top_k_tokens, states.shape[0])
        )
        for rank, (value, position) in enumerate(
            zip(values.tolist(), positions.tolist(), strict=True), start=1
        ):
            rows.append(
                {
                    "feature_id": feature_id,
                    "rank": rank,
                    "token_position": position,
                    "token_id": token_ids[position],
                    "token_text": token_pieces[position],
                    "activation": value,
                    "context": token_context(
                        token_pieces, position, args.context_tokens
                    ),
                }
            )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "activation": str(args.activation),
        "model": args.model,
        "sae_release": loaded.release,
        "sae_id": loaded.sae_id,
        "layer": args.layer,
        "token_count": len(token_ids),
        "feature_ids": feature_ids,
        "top_k_tokens": args.top_k_tokens,
        "ranking_csv": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    logging.info("Saved Experiment 2 rankings to %s", args.output)


if __name__ == "__main__":
    main()
