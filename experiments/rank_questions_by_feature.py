#!/usr/bin/env python3
"""Rank HLE questions by selected features' final-token SAE activations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVOLUTIONARY_SEARCH_DIR = REPO_ROOT / "evolutionary search"
if str(EVOLUTIONARY_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(EVOLUTIONARY_SEARCH_DIR))

# Reuse exactly the same tensor loading, final-token selection, dataset-name
# normalization, activation-path resolution, and SAE loading as Experiment 1.
from build_artifacts import (  # noqa: E402
    load_hidden_tensor,
    normalize_dataset_name,
    resolve_activation_path,
    selected_state,
)
from llamascope import (  # noqa: E402
    DEFAULT_RELEASE,
    DEFAULT_WIDTH,
    load_llamascope_sae,
)


DEFAULT_FEATURE_IDS = (
    2548,
    11089,
    14847,
    36468,
    36484,
    57138,
    66188,
    69504,
    106542,
    121471,
)


@dataclass
class QuestionRecord:
    capture_index: int
    activation_path: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the HLE questions with the highest final-token activations "
            "for selected LlamaScope SAE features."
        )
    )
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--feature-id",
        type=int,
        action="append",
        help=(
            "Feature to rank; repeat for multiple features. If omitted, uses "
            f"the requested defaults: {','.join(map(str, DEFAULT_FEATURE_IDS))}."
        ),
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        help=(
            "JSON file containing feature IDs, either as a list or under "
            "--feature-set-key. Overrides the built-in defaults."
        ),
    )
    parser.add_argument(
        "--feature-set-key",
        default="target_high_control_low",
        help="Object key to read from --feature-file.",
    )
    parser.add_argument("--dataset", default="hle")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--minimum-activation",
        type=float,
        default=0.0,
        help="Exclude questions whose selected-feature activation is not above this value.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_outputs/exp1_100/top_questions"),
    )
    parser.add_argument(
        "--output-prefix",
        default="top_hle_questions_by_feature",
        help="Filename prefix for the JSON and CSV reports.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if args.feature_id and args.feature_file:
        raise ValueError("Use either --feature-id or --feature-file, not both.")
    if args.feature_file and not args.feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {args.feature_file}")


def selected_feature_ids(args: argparse.Namespace) -> list[int]:
    if args.feature_file:
        payload = json.loads(args.feature_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if args.feature_set_key not in payload:
                raise KeyError(
                    f"Feature set {args.feature_set_key!r} not found in "
                    f"{args.feature_file}."
                )
            payload = payload[args.feature_set_key]
        if not isinstance(payload, list):
            raise TypeError("Selected feature JSON value must be a list.")
        feature_ids = [int(value) for value in payload]
    else:
        feature_ids = list(args.feature_id or DEFAULT_FEATURE_IDS)

    feature_ids = list(dict.fromkeys(feature_ids))
    if not feature_ids:
        raise ValueError("No feature IDs were selected.")
    if min(feature_ids) < 0:
        raise ValueError("Feature IDs cannot be negative.")
    return feature_ids


def load_question_records(
    metadata_file: Path,
    activation_dir: Path,
    *,
    layer: int,
    dataset: str,
) -> list[QuestionRecord]:
    target_dataset = normalize_dataset_name(dataset)
    records: list[QuestionRecord] = []
    with metadata_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSON on metadata line %d", line_number + 1)
                continue
            record_layer = int(metadata.get("capture_layer", layer))
            if record_layer != layer:
                continue
            source_dataset = normalize_dataset_name(
                str(metadata.get("source_dataset", "unknown"))
            )
            if source_dataset != target_dataset:
                continue
            capture_index = int(metadata.get("capture_dataset_idx", line_number))
            activation_path = resolve_activation_path(
                activation_dir,
                layer,
                capture_index,
                metadata.get("activation_path"),
            )
            records.append(
                QuestionRecord(capture_index, activation_path, metadata)
            )
    return records


def encode_selected_final_activations(
    records: list[QuestionRecord],
    sae,
    feature_ids: list[int],
    batch_size: int,
) -> tuple[list[QuestionRecord], torch.Tensor, int]:
    valid_records: list[QuestionRecord] = []
    activation_batches: list[torch.Tensor] = []
    pending_states: list[torch.Tensor] = []
    pending_records: list[QuestionRecord] = []
    missing = 0

    def encode_pending() -> None:
        if not pending_states:
            return
        states = torch.stack(pending_states).to(device=sae.device, dtype=sae.dtype)
        with torch.inference_mode():
            encoded = sae.encode(states)
        if max(feature_ids) >= encoded.shape[-1]:
            raise IndexError(
                f"Feature ID {max(feature_ids)} is outside SAE width "
                f"{encoded.shape[-1]}."
            )
        activation_batches.append(
            encoded[:, feature_ids].detach().to(device="cpu", dtype=torch.float32)
        )
        valid_records.extend(pending_records)
        pending_states.clear()
        pending_records.clear()

    for record in records:
        if not record.activation_path.exists():
            missing += 1
            continue
        state = selected_state(load_hidden_tensor(record.activation_path), -1)
        pending_states.append(state.cpu())
        pending_records.append(record)
        if len(pending_states) >= batch_size:
            encode_pending()
    encode_pending()

    if not activation_batches:
        raise ValueError("No matching activation files could be encoded.")
    return valid_records, torch.cat(activation_batches, dim=0), missing


def result_for_question(
    record: QuestionRecord,
    *,
    rank: int,
    activation: float,
) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "rank": rank,
        "activation": activation,
        "capture_dataset_idx": record.capture_index,
        "source_id": metadata.get("source_id"),
        "source_split": metadata.get("source_split"),
        "subject": metadata.get("subject"),
        "question_type": metadata.get("question_type"),
        "question_text": metadata.get("question_text"),
        "prompt_text": metadata.get("prompt_text"),
        "choices": metadata.get("choices"),
        "gold_answer": metadata.get("gold_answer"),
        "activation_path": str(record.activation_path),
    }


def rank_questions(
    records: list[QuestionRecord],
    activations: torch.Tensor,
    feature_ids: list[int],
    *,
    top_k: int,
    minimum_activation: float,
) -> dict[str, list[dict[str, Any]]]:
    ranked: dict[str, list[dict[str, Any]]] = {}
    for column, feature_id in enumerate(feature_ids):
        order = torch.argsort(activations[:, column], descending=True)
        feature_results = []
        for row_index in order.tolist():
            activation = float(activations[row_index, column])
            if activation <= minimum_activation:
                continue
            feature_results.append(
                result_for_question(
                    records[row_index],
                    rank=len(feature_results) + 1,
                    activation=activation,
                )
            )
            if len(feature_results) == top_k:
                break
        ranked[str(feature_id)] = feature_results
    return ranked


def write_csv(path: Path, ranked: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = [
        "feature_id",
        "rank",
        "activation",
        "capture_dataset_idx",
        "source_id",
        "source_split",
        "subject",
        "question_type",
        "question_text",
        "prompt_text",
        "choices",
        "gold_answer",
        "activation_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id, questions in ranked.items():
            for question in questions:
                row = {"feature_id": int(feature_id), **question}
                row["choices"] = json.dumps(row["choices"], ensure_ascii=False)
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    feature_ids = selected_feature_ids(args)

    records = load_question_records(
        args.metadata_file,
        args.activation_dir,
        layer=args.layer,
        dataset=args.dataset,
    )
    if not records:
        raise ValueError(
            f"No {args.dataset!r} records for layer {args.layer} were found in "
            f"{args.metadata_file}."
        )
    logging.info("Found %d matching question records", len(records))

    loaded = load_llamascope_sae(
        layer=args.layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )
    valid_records, activations, missing = encode_selected_final_activations(
        records, loaded.sae, feature_ids, args.batch_size
    )
    ranked = rank_questions(
        valid_records,
        activations,
        feature_ids,
        top_k=args.top_k,
        minimum_activation=args.minimum_activation,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_prefix}.json"
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    report = {
        "configuration": {
            "dataset": normalize_dataset_name(args.dataset),
            "layer": args.layer,
            "token_position": -1,
            "feature_ids": feature_ids,
            "top_k": args.top_k,
            "minimum_activation": args.minimum_activation,
            "question_count": len(valid_records),
            "missing_activation_files": missing,
            "sae_release": loaded.release,
            "sae_id": loaded.sae_id,
        },
        "features": ranked,
    }
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(csv_path, ranked)
    logging.info("Saved JSON report to %s", json_path)
    logging.info("Saved CSV report to %s", csv_path)


if __name__ == "__main__":
    main()
