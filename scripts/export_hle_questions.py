#!/usr/bin/env python3
"""Export the first N text-only HLE questions to a readable text file."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "cais/hle"
DEFAULT_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HLE and export the first N text-only questions in a "
            "human-readable format."
        )
    )
    parser.add_argument(
        "num_questions",
        type=int,
        help="Number of text-only questions to export.",
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
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Include gold answers. Answers are omitted by default.",
    )
    return parser.parse_args()


def contains_payload(value: Any) -> bool:
    """Return whether an image-like dataset field contains actual content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bytes):
        return bool(value)
    if isinstance(value, dict):
        return any(contains_payload(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_payload(item) for item in value)
    return True


def is_text_only(record: dict[str, Any]) -> bool:
    image_keys = (
        "image",
        "images",
        "figure",
        "figures",
        "image_path",
        "image_url",
    )
    return not any(contains_payload(record.get(key)) for key in image_keys)


def first_present(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "" and value != [] and value != {}:
            return value
    return None


def normalize_choices(raw_choices: Any) -> list[tuple[str, str]]:
    """Normalize common HF choice structures into (label, text) pairs."""
    if not raw_choices:
        return []

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if isinstance(raw_choices, dict):
        raw_labels = raw_choices.get("label") or raw_choices.get("labels")
        raw_texts = raw_choices.get("text") or raw_choices.get("texts")
        if isinstance(raw_labels, list) and isinstance(raw_texts, list):
            return [
                (str(label), str(text))
                for label, text in zip(raw_labels, raw_texts, strict=False)
            ]
        return [(str(label), str(text)) for label, text in raw_choices.items()]

    if isinstance(raw_choices, (list, tuple)):
        normalized = []
        for index, choice in enumerate(raw_choices):
            default_label = labels[index] if index < len(labels) else str(index + 1)
            if isinstance(choice, dict):
                label = first_present(choice, ("label", "key", "id")) or default_label
                text = first_present(choice, ("text", "value", "choice", "option"))
                normalized.append((str(label), str(text if text is not None else choice)))
            else:
                normalized.append((default_label, str(choice)))
        return normalized

    return [("A", str(raw_choices))]


def normalized_record(record: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    choices = normalize_choices(
        first_present(
            record,
            ("choices", "options", "multiple_choice_targets"),
        )
    )
    answer = first_present(
        record,
        ("answer", "answer_key", "correct_answer", "solution"),
    )
    source_id = first_present(
        record,
        ("question_id", "id", "uid"),
    )
    subject = first_present(
        record,
        ("subject", "category", "raw_subject", "field"),
    )
    question_type = first_present(record, ("answer_type",))
    return {
        "source_id": source_id if source_id is not None else str(fallback_index),
        "subject": subject if subject is not None else "unknown",
        "question_type": question_type
        or ("multiple_choice" if choices else "short_answer"),
        "question": str(record.get("question", "")).strip(),
        "choices": choices,
        "answer": answer,
    }


def format_question(
    question_number: int,
    record: dict[str, Any],
    *,
    include_answer: bool,
) -> str:
    separator = "=" * 80
    lines = [
        separator,
        f"HLE QUESTION {question_number}",
        separator,
        f"ID: {record['source_id']}",
        f"Subject: {record['subject']}",
        f"Type: {record['question_type']}",
        "",
        "Question:",
        record["question"],
    ]
    if record["choices"]:
        lines.extend(["", "Choices:"])
        lines.extend(f"{label}. {text}" for label, text in record["choices"])
    if include_answer:
        answer = record["answer"] if record["answer"] is not None else "<not provided>"
        lines.extend(["", "Gold answer:", str(answer)])
    return "\n".join(lines)


def export_questions(args: argparse.Namespace) -> int:
    if args.num_questions < 1:
        raise ValueError("num_questions must be at least 1.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install the SIPIT dependencies "
            "with: python -m pip install -e ."
        ) from exc

    print(f"Loading {args.dataset} split={args.split}...")
    dataset = load_dataset(args.dataset, split=args.split)
    formatted = []
    for dataset_index, raw_record in enumerate(dataset):
        record = dict(raw_record)
        if not is_text_only(record):
            continue
        normalized = normalized_record(record, dataset_index)
        formatted.append(
            format_question(
                len(formatted) + 1,
                normalized,
                include_answer=args.include_answers,
            )
        )
        if len(formatted) == args.num_questions:
            break

    if not formatted:
        raise ValueError("No text-only HLE questions were found.")

    output = Path(f"hle_questions_first_{args.num_questions}.txt")
    output.write_text("\n\n".join(formatted) + "\n", encoding="utf-8")
    print(f"Exported {len(formatted)} question(s) to {output}")
    if len(formatted) < args.num_questions:
        print(
            f"Warning: requested {args.num_questions}, but only "
            f"{len(formatted)} text-only question(s) were available."
        )
    return len(formatted)


def main() -> None:
    export_questions(parse_args())


if __name__ == "__main__":
    main()
