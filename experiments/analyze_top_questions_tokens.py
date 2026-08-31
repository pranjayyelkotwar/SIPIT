#!/usr/bin/env python3
"""Run token-level SAE analysis on a feature's top-ranked questions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVOLUTIONARY_SEARCH_DIR = REPO_ROOT / "evolutionary search"
if str(EVOLUTIONARY_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(EVOLUTIONARY_SEARCH_DIR))

from build_artifacts import flatten_states, load_hidden_tensor  # noqa: E402
from llamascope import (  # noqa: E402
    DEFAULT_RELEASE,
    DEFAULT_WIDTH,
    load_llamascope_sae,
)
from experiments.inspect_token_feature_activations import (  # noqa: E402
    DEFAULT_MODEL,
    encode_selected_features,
    token_context,
)


DEFAULT_FEATURE_ID = 2548


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 2 on the top-ranked questions for one feature, "
            "loading the tokenizer and SAE only once."
        )
    )
    parser.add_argument(
        "--ranked-json",
        type=Path,
        default=Path(
            "experiment_outputs/exp1_100/top_questions/"
            "top_hle_questions_by_feature.json"
        ),
    )
    parser.add_argument("--feature-id", type=int, default=DEFAULT_FEATURE_ID)
    parser.add_argument("--top-k-questions", type=int, default=5)
    parser.add_argument("--top-k-tokens", type=int, default=20)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--activation-dir",
        type=Path,
        help=(
            "Optional capture directory used to relocate activation files if "
            "the paths stored in the ranking JSON no longer exist."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/exp2/feature_2548_top5"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--add-bos-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must match the original activation-capture setting.",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.ranked_json.exists():
        raise FileNotFoundError(f"Ranking JSON not found: {args.ranked_json}")
    if args.feature_id < 0:
        raise ValueError("--feature-id cannot be negative.")
    if args.top_k_questions < 1:
        raise ValueError("--top-k-questions must be at least 1.")
    if args.top_k_tokens < 1:
        raise ValueError("--top-k-tokens must be at least 1.")
    if args.context_tokens < 0:
        raise ValueError("--context-tokens cannot be negative.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")


def resolve_question_activation(
    question: dict[str, Any], activation_dir: Path | None, layer: int
) -> Path:
    recorded = Path(question["activation_path"])
    candidates = [recorded]
    if activation_dir is not None:
        candidates.extend(
            [
                activation_dir / f"layer_{layer}" / recorded.name,
                activation_dir / recorded.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Activation file not found for question rank {question.get('rank')}: "
        f"{recorded}. Supply --activation-dir if the captures moved."
    )


def aligned_tokens(
    tokenizer, prompt: str, state_count: int, add_bos_token: bool
) -> tuple[list[int], list[str]]:
    full_token_ids = tokenizer.encode(
        prompt, add_special_tokens=add_bos_token, truncation=False
    )
    token_ids = full_token_ids[:state_count]
    if len(token_ids) != state_count:
        raise ValueError(
            f"Prompt produced {len(token_ids)} tokens for {state_count} states. "
            "Check the prompt, tokenizer model, and BOS-token setting."
        )
    token_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
    return token_ids, token_pieces


def analyze_question(
    question: dict[str, Any],
    *,
    activation_path: Path,
    tokenizer,
    sae,
    feature_id: int,
    add_bos_token: bool,
    chunk_size: int,
    top_k_tokens: int,
    context_tokens: int,
) -> list[dict[str, Any]]:
    states = flatten_states(load_hidden_tensor(activation_path)).cpu()
    token_ids, token_pieces = aligned_tokens(
        tokenizer,
        question["prompt_text"],
        states.shape[0],
        add_bos_token,
    )
    feature_activations = encode_selected_features(
        states, sae, [feature_id], chunk_size
    )[:, 0]
    values, positions = torch.topk(
        feature_activations,
        k=min(top_k_tokens, feature_activations.shape[0]),
    )
    rows = []
    for token_rank, (value, position) in enumerate(
        zip(values.tolist(), positions.tolist(), strict=True), start=1
    ):
        rows.append(
            {
                "feature_id": feature_id,
                "question_rank": question["rank"],
                "source_id": question.get("source_id"),
                "final_token_activation": question["activation"],
                "question_text": question.get("question_text"),
                "activation_path": str(activation_path),
                "token_rank": token_rank,
                "token_position": position,
                "token_id": token_ids[position],
                "token_text": token_pieces[position],
                "token_activation": value,
                "context": token_context(token_pieces, position, context_tokens),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ranking_report = json.loads(args.ranked_json.read_text(encoding="utf-8"))
    feature_key = str(args.feature_id)
    available = ranking_report.get("features", {}).get(feature_key)
    if not available:
        raise ValueError(f"Feature {args.feature_id} is absent from {args.ranked_json}.")
    questions = available[: args.top_k_questions]
    layer = args.layer or int(ranking_report["configuration"]["layer"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    loaded = load_llamascope_sae(
        layer=layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )

    rows: list[dict[str, Any]] = []
    for question in questions:
        activation_path = resolve_question_activation(
            question, args.activation_dir, layer
        )
        logging.info(
            "Analyzing feature %d, question rank %d", args.feature_id, question["rank"]
        )
        rows.extend(
            analyze_question(
                question,
                activation_path=activation_path,
                tokenizer=tokenizer,
                sae=loaded.sae,
                feature_id=args.feature_id,
                add_bos_token=args.add_bos_token,
                chunk_size=args.chunk_size,
                top_k_tokens=args.top_k_tokens,
                context_tokens=args.context_tokens,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "token_feature_rankings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "token_feature_rankings.json"
    json_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "feature_id": args.feature_id,
                    "question_count": len(questions),
                    "top_k_tokens": args.top_k_tokens,
                    "context_tokens": args.context_tokens,
                    "layer": layer,
                    "model": args.model,
                    "sae_release": loaded.release,
                    "sae_id": loaded.sae_id,
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Saved %s", csv_path)
    logging.info("Saved %s", json_path)


if __name__ == "__main__":
    main()
