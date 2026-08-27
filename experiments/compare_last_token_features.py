#!/usr/bin/env python3
"""Compare final-prompt-token SAE features between HLE and ARC.

This experiment consumes the residual-stream tensors and JSONL metadata written
by ``activation_capture/capture_activations.py``.  Only the final token state of
each question is encoded by the SAE; activations from all earlier positions are
ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVOLUTIONARY_SEARCH_DIR = REPO_ROOT / "evolutionary search"
if str(EVOLUTIONARY_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(EVOLUTIONARY_SEARCH_DIR))

# Reuse the activation readers, dataset-name normalization, and SAE loader used
# by the existing evolutionary-search artifact pipeline.
from build_artifacts import (  # noqa: E402
    load_hidden_tensor,
    metadata_records,
    selected_state,
)
from llamascope import (  # noqa: E402
    DEFAULT_RELEASE,
    DEFAULT_WIDTH,
    load_llamascope_sae,
)


@dataclass
class FeatureMoments:
    count: int
    total: torch.Tensor
    total_squared: torch.Tensor
    active_count: torch.Tensor

    @classmethod
    def empty(cls, feature_count: int) -> "FeatureMoments":
        return cls(
            count=0,
            total=torch.zeros(feature_count, dtype=torch.float64),
            total_squared=torch.zeros(feature_count, dtype=torch.float64),
            active_count=torch.zeros(feature_count, dtype=torch.int64),
        )

    def update(self, features: torch.Tensor, activation_threshold: float) -> None:
        values = features.detach().to(device="cpu", dtype=torch.float64)
        self.count += values.shape[0]
        self.total += values.sum(dim=0)
        self.total_squared += values.square().sum(dim=0)
        self.active_count += (values > activation_threshold).sum(dim=0)

    def summarize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise ValueError("Cannot summarize an empty dataset.")
        mean = self.total / self.count
        variance = (self.total_squared / self.count - mean.square()).clamp_min(0)
        prevalence = self.active_count.to(torch.float64) / self.count
        return mean, variance.sqrt(), prevalence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode only the final prompt-token state for each HLE/ARC question "
            "and compare SAE feature activation statistics."
        )
    )
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Capture layer and matching LlamaScope SAE layer.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_outputs/exp1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of final-token states encoded per SAE call.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="A feature is active on a question when activation exceeds this value.",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.10,
        help="Minimum active-question fraction for membership in a dataset set.",
    )
    parser.add_argument(
        "--min-mean",
        type=float,
        default=0.0,
        help="Minimum dataset mean for membership in a dataset set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of largest absolute HLE-minus-ARC differences in the CSV.",
    )
    parser.add_argument(
        "--hle-name",
        default="hle",
        help="Normalized dataset name used for HLE metadata records.",
    )
    parser.add_argument(
        "--arc-name",
        default="arc_easy",
        help="Normalized dataset name used for ARC metadata records.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not 0 <= args.min_prevalence <= 1:
        raise ValueError("--min-prevalence must be between 0 and 1.")
    if not args.metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")


def encode_final_states(
    paths: list[Path],
    sae,
    batch_size: int,
    activation_threshold: float,
) -> tuple[FeatureMoments, int]:
    moments: FeatureMoments | None = None
    missing = 0
    pending: list[torch.Tensor] = []

    def encode_pending() -> None:
        nonlocal moments
        if not pending:
            return
        states = torch.stack(pending).to(device=sae.device, dtype=sae.dtype)
        with torch.inference_mode():
            features = sae.encode(states)
        if moments is None:
            moments = FeatureMoments.empty(features.shape[-1])
        moments.update(features, activation_threshold)
        pending.clear()

    for path in paths:
        if not path.exists():
            missing += 1
            continue
        # selected_state(..., -1) is the only token-selection operation here.
        pending.append(selected_state(load_hidden_tensor(path), -1).cpu())
        if len(pending) >= batch_size:
            encode_pending()
    encode_pending()
    if moments is None:
        raise ValueError("No activation files could be encoded.")
    return moments, missing


def feature_set(
    mean: torch.Tensor,
    prevalence: torch.Tensor,
    *,
    min_mean: float,
    min_prevalence: float,
) -> set[int]:
    selected = (mean > min_mean) & (prevalence >= min_prevalence)
    return set(torch.nonzero(selected, as_tuple=False).flatten().tolist())


def standardized_difference(
    first_mean: torch.Tensor,
    first_std: torch.Tensor,
    second_mean: torch.Tensor,
    second_std: torch.Tensor,
) -> torch.Tensor:
    pooled = ((first_std.square() + second_std.square()) / 2).sqrt()
    difference = first_mean - second_mean
    return torch.where(pooled > 0, difference / pooled, torch.zeros_like(difference))


def write_feature_stats(
    path: Path,
    *,
    hle_mean: torch.Tensor,
    hle_std: torch.Tensor,
    hle_prevalence: torch.Tensor,
    arc_mean: torch.Tensor,
    arc_std: torch.Tensor,
    arc_prevalence: torch.Tensor,
    top_k: int,
) -> None:
    difference = hle_mean - arc_mean
    effect_size = standardized_difference(hle_mean, hle_std, arc_mean, arc_std)
    order = torch.argsort(difference.abs(), descending=True)[:top_k]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "feature_id",
                "hle_mean",
                "arc_mean",
                "mean_difference_hle_minus_arc",
                "standardized_difference",
                "hle_prevalence",
                "arc_prevalence",
            ]
        )
        for rank, feature_id in enumerate(order.tolist(), start=1):
            writer.writerow(
                [
                    rank,
                    feature_id,
                    hle_mean[feature_id].item(),
                    arc_mean[feature_id].item(),
                    difference[feature_id].item(),
                    effect_size[feature_id].item(),
                    hle_prevalence[feature_id].item(),
                    arc_prevalence[feature_id].item(),
                ]
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dataset_paths: dict[str, list[Path]] = {args.hle_name: [], args.arc_name: []}
    for _, dataset, path in metadata_records(
        args.metadata_file, args.activation_dir, args.layer
    ):
        if dataset in dataset_paths:
            dataset_paths[dataset].append(path)
    for dataset, paths in dataset_paths.items():
        if not paths:
            raise ValueError(
                f"No {dataset!r} records found for layer {args.layer} in "
                f"{args.metadata_file}."
            )
        logging.info("Found %d %s activation records", len(paths), dataset)

    loaded = load_llamascope_sae(
        layer=args.layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )
    hle_moments, hle_missing = encode_final_states(
        dataset_paths[args.hle_name],
        loaded.sae,
        args.batch_size,
        args.activation_threshold,
    )
    arc_moments, arc_missing = encode_final_states(
        dataset_paths[args.arc_name],
        loaded.sae,
        args.batch_size,
        args.activation_threshold,
    )
    hle_mean, hle_std, hle_prevalence = hle_moments.summarize()
    arc_mean, arc_std, arc_prevalence = arc_moments.summarize()

    hle_features = feature_set(
        hle_mean,
        hle_prevalence,
        min_mean=args.min_mean,
        min_prevalence=args.min_prevalence,
    )
    arc_features = feature_set(
        arc_mean,
        arc_prevalence,
        min_mean=args.min_mean,
        min_prevalence=args.min_prevalence,
    )
    feature_sets = {
        "hle": sorted(hle_features),
        "arc": sorted(arc_features),
        "hle_minus_arc": sorted(hle_features - arc_features),
        "arc_minus_hle": sorted(arc_features - hle_features),
        "intersection": sorted(hle_features & arc_features),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2) + "\n", encoding="utf-8"
    )
    write_feature_stats(
        args.output_dir / "feature_stats.csv",
        hle_mean=hle_mean,
        hle_std=hle_std,
        hle_prevalence=hle_prevalence,
        arc_mean=arc_mean,
        arc_std=arc_std,
        arc_prevalence=arc_prevalence,
        top_k=min(args.top_k, hle_mean.numel()),
    )
    torch.save(
        {
            "hle_mean": hle_mean.float(),
            "hle_std": hle_std.float(),
            "hle_prevalence": hle_prevalence.float(),
            "arc_mean": arc_mean.float(),
            "arc_std": arc_std.float(),
            "arc_prevalence": arc_prevalence.float(),
        },
        args.output_dir / "feature_statistics.pt",
    )
    summary = {
        "sae_release": loaded.release,
        "sae_id": loaded.sae_id,
        "layer": args.layer,
        "token_position": -1,
        "activation_threshold": args.activation_threshold,
        "min_mean": args.min_mean,
        "min_prevalence": args.min_prevalence,
        "hle_questions": hle_moments.count,
        "arc_questions": arc_moments.count,
        "missing_hle_files": hle_missing,
        "missing_arc_files": arc_missing,
        "set_sizes": {name: len(values) for name, values in feature_sets.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logging.info("Saved Experiment 1 outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
