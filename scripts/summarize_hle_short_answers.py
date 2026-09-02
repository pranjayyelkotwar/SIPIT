#!/usr/bin/env python3
"""Show an HLE short-answer example and count Math vs non-Math questions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any


DEFAULT_DATASET = "cais/hle"
DEFAULT_SPLIT = "test"
SHORT_ANSWER_TYPE = "exactMatch"
MATH_CATEGORY = "Math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print one real HLE short-answer question and count short-answer "
            "Math vs non-Math questions."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset ID (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split (default: {DEFAULT_SPLIT}).",
    )
    return parser.parse_args()


def is_short_answer(record: Mapping[str, Any]) -> bool:
    """Return whether the record uses HLE's short-answer format."""
    return record.get("answer_type") == SHORT_ANSWER_TYPE


def is_math(record: Mapping[str, Any]) -> bool:
    """Return whether the record belongs to HLE's broad Math category."""
    return record.get("category") == MATH_CATEGORY


def is_text_only(record: Mapping[str, Any]) -> bool:
    """Match the text-only rule used by HLEDataset.keep_example()."""
    return record.get("image", "") == ""


def update_counts(counts: dict[str, int], record: Mapping[str, Any]) -> None:
    group = "math" if is_math(record) else "non_math"
    counts[group] += 1


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with: "
            "python -m pip install datasets"
        ) from exc

    print(f"Loading {args.dataset} split={args.split}...")
    dataset = load_dataset(args.dataset, split=args.split)

    all_counts = {"math": 0, "non_math": 0}
    text_only_counts = {"math": 0, "non_math": 0}
    example: Mapping[str, Any] | None = None

    for raw_record in dataset:
        record = dict(raw_record)
        if not is_short_answer(record):
            continue

        if example is None and is_text_only(record):
            example = record

        update_counts(all_counts, record)
        if is_text_only(record):
            update_counts(text_only_counts, record)

    if example is None:
        raise SystemExit("No text-only HLE short-answer question was found.")

    print("\nExample real HLE short-answer question")
    print("-" * 40)
    print(f"ID: {example.get('id', '<missing>')}")
    print(f"Category: {example.get('category', '<missing>')}")
    print(f"Answer type: {example.get('answer_type', '<missing>')}")
    print(f"Question:\n{example.get('question', '<missing>')}")

    print("\nShort-answer counts (answer_type == 'exactMatch')")
    print("-" * 54)
    print(
        f"Full HLE split:       Math={all_counts['math']}, "
        f"non-Math={all_counts['non_math']}, "
        f"total={sum(all_counts.values())}"
    )
    print(
        f"Text-only experiment: Math={text_only_counts['math']}, "
        f"non-Math={text_only_counts['non_math']}, "
        f"total={sum(text_only_counts.values())}"
    )


if __name__ == "__main__":
    main()
