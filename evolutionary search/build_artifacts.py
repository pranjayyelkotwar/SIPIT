#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from grounding import (
    PerplexityRegressionWeights,
    save_perplexity_regression_weights,
)
from llamascope import DEFAULT_RELEASE, DEFAULT_WIDTH, load_llamascope_sae


def normalize_dataset_name(name: str) -> str:
    lowered = name.lower()
    canonical = lowered.replace("-", "_").replace(" ", "_")
    if canonical.startswith("hle_"):
        return canonical
    if "arc" in lowered:
        return "arc_easy"
    if "hle" in lowered:
        return "hle"
    if "mmlu" in lowered:
        return "mmlu"
    return canonical


def resolve_activation_path(
    activation_dir: Path, layer: int, index: int, recorded: str | None = None
) -> Path:
    candidates = []
    if recorded:
        recorded_path = Path(recorded)
        candidates.append(recorded_path)
        candidates.append(activation_dir / recorded_path)
        candidates.append(activation_dir / f"layer_{layer}" / recorded_path.name)
    candidates.append(
        activation_dir / f"layer_{layer}" / f"activations_l{layer}_idx{index}.pt"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_hidden_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict) and torch.is_tensor(payload.get("hidden_states")):
        return payload["hidden_states"]
    raise TypeError(f"{path} is neither a tensor nor a hidden_states bundle.")


def flatten_states(states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 1:
        return states.unsqueeze(0)
    return states.reshape(-1, states.shape[-1])


def selected_state(states: torch.Tensor, token_pos: int) -> torch.Tensor:
    states = flatten_states(states)
    position = states.shape[0] - 1 if token_pos < 0 else token_pos
    if not 0 <= position < states.shape[0]:
        raise IndexError(f"Token position {token_pos} outside {states.shape[0]} states.")
    return states[position]


def metadata_records(
    metadata_file: Path, activation_dir: Path, layer: int
) -> Iterator[tuple[int, str, Path]]:
    with metadata_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_layer = record.get("capture_layer", layer)
            if int(record_layer) != layer:
                continue
            index = int(record.get("capture_dataset_idx", line_number))
            dataset = normalize_dataset_name(record.get("source_dataset", "unknown"))
            path = resolve_activation_path(
                activation_dir, layer, index, record.get("activation_path")
            )
            yield index, dataset, path


def encode_chunks(states: torch.Tensor, sae, chunk_size: int) -> Iterator[torch.Tensor]:
    states = flatten_states(states)
    for chunk in states.split(max(1, chunk_size)):
        with torch.no_grad():
            yield sae.encode(chunk.to(device=sae.device, dtype=sae.dtype))


def build_average_latents(args: argparse.Namespace, sae) -> None:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    missing = 0
    for _, dataset, path in metadata_records(
        args.metadata_file, args.activation_dir, args.layer
    ):
        if not path.exists():
            missing += 1
            continue
        for features in encode_chunks(load_hidden_tensor(path), sae, args.chunk_size):
            if dataset not in sums:
                sums[dataset] = torch.zeros(
                    features.shape[-1], device=features.device, dtype=features.dtype
                )
            sums[dataset] += features.sum(dim=0)
            counts[dataset] += features.shape[0]

    if not sums:
        raise ValueError("No activation files were encoded.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, total in sums.items():
        output = args.output_dir / f"avg_latents_{dataset}.pt"
        torch.save((total / counts[dataset]).cpu(), output)
        logging.info("Saved %s (%d tokens)", output, counts[dataset])
    if missing:
        logging.warning("Skipped %d missing activation files", missing)


def load_targets(path: Path, key: str, layer: int) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("capture_layer") is not None:
                if int(record["capture_layer"]) != layer:
                    continue
            index = record.get("capture_dataset_idx")
            target = record.get(key)
            if index is not None and target is not None:
                values[int(index)].append(float(target))
    return {index: sum(items) / len(items) for index, items in values.items()}


@dataclass
class RegressionConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    normalize_targets: bool
    use_bias: bool


def fit_regression(
    features: torch.Tensor,
    targets: torch.Tensor,
    config: RegressionConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, float, float, float, float]:
    mean = targets.mean().item()
    std = targets.std(unbiased=False).item()
    std = std if std > 0 else 1.0
    fitted_targets = (targets - mean) / std if config.normalize_targets else targets
    model = torch.nn.Linear(features.shape[-1], 1, bias=config.use_bias).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for epoch in range(config.epochs):
        permutation = torch.randperm(features.shape[0])
        epoch_loss = 0.0
        for start in range(0, features.shape[0], config.batch_size):
            indices = permutation[start : start + config.batch_size]
            prediction = model(features[indices].to(device)).squeeze(-1)
            expected = fitted_targets[indices].to(device)
            loss = (prediction - expected).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * indices.numel()
        logging.info(
            "Epoch %d/%d MSE=%.6g",
            epoch + 1,
            config.epochs,
            epoch_loss / features.shape[0],
        )
    with torch.no_grad():
        mse = (
            model(features.to(device)).squeeze(-1) - fitted_targets.to(device)
        ).square().mean().item()
    weights = model.weight.detach().cpu().squeeze(0)
    bias = (
        float(model.bias.detach().cpu().item()) if config.use_bias else 0.0
    )
    return weights, bias, mean, std, mse


def build_regression(args: argparse.Namespace, sae) -> None:
    targets = load_targets(args.targets_jsonl, args.target_key, args.layer)
    if not targets:
        raise ValueError("No matching targets were found in the JSONL file.")
    features = []
    expected = []
    for index, _, path in metadata_records(
        args.metadata_file, args.activation_dir, args.layer
    ):
        if index not in targets or not path.exists():
            continue
        state = selected_state(load_hidden_tensor(path), args.token_pos)
        with torch.no_grad():
            encoded = sae.encode(
                state.to(device=sae.device, dtype=sae.dtype).unsqueeze(0)
            )
        features.append(encoded.cpu().float().squeeze(0))
        expected.append(targets[index])
        if args.max_samples and len(features) >= args.max_samples:
            break
    if not features:
        raise ValueError("No activation/target pairs were found.")
    feature_tensor = torch.stack(features)
    target_tensor = torch.tensor(expected, dtype=torch.float32)
    weights, bias, mean, std, mse = fit_regression(
        feature_tensor,
        target_tensor,
        RegressionConfig(
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.normalize_targets,
            args.use_bias,
        ),
        sae.device,
    )
    artifact = PerplexityRegressionWeights(
        weights=weights,
        bias=bias,
        target_mean=mean,
        target_std=std if args.normalize_targets else 1.0,
        target_key=args.target_key,
        normalized=args.normalize_targets,
        layer=args.layer,
        token_pos=args.token_pos,
    )
    save_perplexity_regression_weights(artifact, args.output_weights)
    logging.info(
        "Saved %s (%d samples, %d features, training MSE %.6g)",
        args.output_weights,
        feature_tensor.shape[0],
        feature_tensor.shape[1],
        mse,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Build LlamaScope grounding artifacts for evolutionary search."
    )
    root.add_argument("--activation-dir", type=Path, required=True)
    root.add_argument("--metadata-file", type=Path, required=True)
    root.add_argument("--layer", type=int, required=True)
    root.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    root.add_argument("--release", default=DEFAULT_RELEASE)
    root.add_argument("--width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    root.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    commands = root.add_subparsers(dest="command", required=True)

    averages = commands.add_parser("averages")
    averages.add_argument("--output-dir", type=Path, required=True)
    averages.add_argument("--chunk-size", type=int, default=4096)

    regression = commands.add_parser("regression")
    regression.add_argument("--targets-jsonl", type=Path, required=True)
    regression.add_argument(
        "--target-key",
        choices=("perplexity", "mean_nll", "sum_nll"),
        default="perplexity",
    )
    regression.add_argument("--token-pos", type=int, default=-1)
    regression.add_argument("--max-samples", type=int)
    regression.add_argument("--epochs", type=int, default=100)
    regression.add_argument("--batch-size", type=int, default=32)
    regression.add_argument("--lr", type=float, default=1e-2)
    regression.add_argument("--weight-decay", type=float, default=0.0)
    regression.add_argument(
        "--normalize-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    regression.add_argument(
        "--use-bias", action=argparse.BooleanOptionalAction, default=True
    )
    regression.add_argument("--seed", type=int, default=0)
    regression.add_argument("--output-weights", type=Path, required=True)
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parser().parse_args()
    if args.command == "regression":
        torch.manual_seed(args.seed)
    loaded = load_llamascope_sae(
        layer=args.layer,
        device=args.device,
        release=args.release,
        width=args.width,
        dtype=args.dtype,
    )
    if args.command == "averages":
        build_average_latents(args, loaded.sae)
    else:
        build_regression(args, loaded.sae)


if __name__ == "__main__":
    main()
