#!/usr/bin/env python3
"""Inspect and rank an averaged SAE latent tensor.

The default input is the HLE average produced by
``evolutionary search/build_artifacts.py averages``. That artifact contains one
mean value per SAE feature; it does not retain individual question or token
activations.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch


DEFAULT_INPUT = Path("hle_sae_averages/avg_latents_hle.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize an averaged SAE latent tensor, print its strongest "
            "features, and optionally export a complete ranking to CSV."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Averaged SAE tensor (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of strongest mean feature activations to print (default: 20).",
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        action="append",
        default=[],
        help="Inspect a particular feature ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "CSV destination. By default, writes INPUT with '_ranked.csv' "
            "appended to its stem."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the complete ranked-feature CSV.",
    )
    return parser.parse_args()


def load_feature_averages(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Averaged SAE tensor not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(payload):
        raise TypeError(f"Expected a tensor in {path}, got {type(payload).__name__}.")
    if payload.ndim != 1:
        raise ValueError(
            f"Expected one mean per feature with shape (d_sae,), got {tuple(payload.shape)}."
        )
    if payload.numel() == 0:
        raise ValueError(f"The tensor in {path} is empty.")
    return payload.detach().cpu().float()


def print_summary(path: Path, values: torch.Tensor) -> None:
    finite = torch.isfinite(values)
    finite_values = values[finite]
    if finite_values.numel() == 0:
        raise ValueError("The tensor contains no finite feature values.")

    quantiles = torch.quantile(
        finite_values,
        torch.tensor([0.25, 0.50, 0.75, 0.90, 0.99]),
    )
    positive = int((values > 0).sum())
    zero = int((values == 0).sum())
    negative = int((values < 0).sum())

    print("SAE average-latent summary")
    print("==========================")
    print(f"File:              {path}")
    print(f"Shape:             {tuple(values.shape)}")
    print(f"Feature count:     {values.numel():,}")
    print(f"Positive means:    {positive:,}")
    print(f"Zero means:        {zero:,}")
    print(f"Negative means:    {negative:,}")
    print(f"Non-finite values: {int((~finite).sum()):,}")
    print(f"Minimum:           {finite_values.min().item():.8g}")
    print(f"Maximum:           {finite_values.max().item():.8g}")
    print(f"Mean:              {finite_values.mean().item():.8g}")
    print(f"Std. deviation:    {finite_values.std(unbiased=False).item():.8g}")
    print(f"25th percentile:   {quantiles[0].item():.8g}")
    print(f"Median:            {quantiles[1].item():.8g}")
    print(f"75th percentile:   {quantiles[2].item():.8g}")
    print(f"90th percentile:   {quantiles[3].item():.8g}")
    print(f"99th percentile:   {quantiles[4].item():.8g}")


def ranked_features(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sortable = torch.nan_to_num(values, nan=-math.inf)
    return torch.sort(sortable, descending=True)


def print_top_features(
    sorted_values: torch.Tensor, sorted_ids: torch.Tensor, top_k: int
) -> None:
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    count = min(top_k, sorted_values.numel())
    print(f"\nTop {count} features by mean activation")
    print("======================================")
    print(" rank  feature_id  mean_activation")
    for rank, (feature_id, value) in enumerate(
        zip(sorted_ids[:count].tolist(), sorted_values[:count].tolist()), start=1
    ):
        print(f"{rank:>5}  {feature_id:>10}  {value:>15.8g}")


def print_requested_features(
    values: torch.Tensor, sorted_ids: torch.Tensor, feature_ids: list[int]
) -> None:
    if not feature_ids:
        return
    ranks = torch.empty_like(sorted_ids)
    ranks[sorted_ids] = torch.arange(sorted_ids.numel())
    print("\nRequested feature IDs")
    print("=====================")
    print("feature_id  overall_rank  mean_activation")
    for feature_id in feature_ids:
        if not 0 <= feature_id < values.numel():
            print(f"{feature_id:>10}  OUT_OF_RANGE")
            continue
        print(
            f"{feature_id:>10}  {int(ranks[feature_id]) + 1:>12}  "
            f"{values[feature_id].item():>15.8g}"
        )


def default_csv_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_ranked.csv")


def write_ranking_csv(
    path: Path, sorted_values: torch.Tensor, sorted_ids: torch.Tensor
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature_id", "mean_activation", "is_positive"])
        for rank, (feature_id, value) in enumerate(
            zip(sorted_ids.tolist(), sorted_values.tolist()), start=1
        ):
            writer.writerow([rank, feature_id, value, value > 0])
    print(f"\nSaved the complete feature ranking to: {path}")


def print_interpretation() -> None:
    print(
        "\nInterpretation\n"
        "==============\n"
        "Each feature_id is a LlamaScope l22r_32x SAE latent. Its value is the\n"
        "mean activation across every non-padding token in every captured HLE\n"
        "prompt. The artifact no longer contains question IDs, token positions,\n"
        "maximum activations, or human-readable feature descriptions. Repeated\n"
        "prompt text and multiple-choice formatting can therefore dominate the\n"
        "highest averages; compare against a baseline dataset or inspect the raw\n"
        "per-question activations before attributing a feature specifically to HLE."
    )


def main() -> None:
    args = parse_args()
    values = load_feature_averages(args.input)
    sorted_values, sorted_ids = ranked_features(values)
    print_summary(args.input, values)
    print_top_features(sorted_values, sorted_ids, args.top_k)
    print_requested_features(values, sorted_ids, args.feature_id)
    if not args.no_csv:
        write_ranking_csv(
            args.csv or default_csv_path(args.input), sorted_values, sorted_ids
        )
    print_interpretation()


if __name__ == "__main__":
    main()
