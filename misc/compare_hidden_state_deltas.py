#!/usr/bin/env python3
"""Write hidden-state delta metrics for several artifacts to a CSV file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_hidden_states(path: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location='cpu')
    if torch.is_tensor(payload):
        return payload.detach().float().cpu(), {}
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get('hidden_states')):
        raise TypeError(f'{path} must be a tensor or contain a hidden_states tensor.')
    return payload['hidden_states'].detach().float().cpu(), payload


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f'Hidden-state shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}'
        )

    delta = candidate - reference
    ref_flat = reference.flatten()
    candidate_flat = candidate.flatten()
    delta_flat = delta.flatten()
    mse = delta_flat.square().mean()
    signal_power = ref_flat.square().mean()
    per_token_rmse = delta.square().mean(dim=-1).sqrt().flatten()
    per_token_cosine = F.cosine_similarity(reference, candidate, dim=-1).flatten()

    return {
        'token_count': reference.shape[-2] if reference.ndim >= 2 else 1,
        'hidden_size': reference.shape[-1],
        'cosine_similarity': F.cosine_similarity(
            ref_flat, candidate_flat, dim=0
        ).clamp(-1, 1).item(),
        'mse': mse.item(),
        'rmse': mse.sqrt().item(),
        'mae': delta_flat.abs().mean().item(),
        'max_abs_error': delta_flat.abs().max().item(),
        'l2_error': torch.linalg.vector_norm(delta_flat).item(),
        'relative_l2_error': (
            torch.linalg.vector_norm(delta_flat) / torch.linalg.vector_norm(ref_flat)
        ).item(),
        'snr_db': (
            math.inf if mse.item() == 0
            else (10 * torch.log10(signal_power / mse)).item()
        ),
        'mean_token_rmse': per_token_rmse.mean().item(),
        'max_token_rmse': per_token_rmse.max().item(),
        'mean_token_cosine': per_token_cosine.clamp(-1, 1).mean().item(),
        'min_token_cosine': per_token_cosine.clamp(-1, 1).min().item(),
    }


def token_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, input_ids: torch.Tensor | None
) -> list[dict[str, float | int | str]]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f'Hidden-state shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}'
        )
    if reference.ndim != 2:
        raise ValueError(
            f'Token comparison expects [tokens, hidden_size], got {tuple(reference.shape)}'
        )

    rows = []
    for token_index, (original_token, candidate_token) in enumerate(
        zip(reference, candidate)
    ):
        delta = candidate_token - original_token
        mse = delta.square().mean()
        original_norm = torch.linalg.vector_norm(original_token)
        delta_norm = torch.linalg.vector_norm(delta)
        token_id: int | str = ''
        if input_ids is not None and token_index < input_ids.numel():
            token_id = int(input_ids.flatten()[token_index])
        rows.append({
            'token_index': token_index,
            'token_id': token_id,
            'cosine_similarity': F.cosine_similarity(
                original_token, candidate_token, dim=0
            ).clamp(-1, 1).item(),
            'mse': mse.item(),
            'rmse': mse.sqrt().item(),
            'mae': delta.abs().mean().item(),
            'max_abs_error': delta.abs().max().item(),
            'l2_error': delta_norm.item(),
            'relative_l2_error': (
                delta_norm / original_norm
            ).item() if original_norm.item() else math.inf,
            'original_norm': original_norm.item(),
            'candidate_norm': torch.linalg.vector_norm(candidate_token).item(),
            'delta_mean': delta.mean().item(),
            'delta_std': delta.std().item(),
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument(
        '--token-output', type=Path,
        help='Optional long-form CSV with one row per candidate and token.'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference, reference_payload = load_hidden_states(args.original)
    rows = []
    token_rows = []
    input_ids = reference_payload.get('input_ids')
    if not torch.is_tensor(input_ids):
        input_ids = None
    for path in args.candidate:
        candidate, payload = load_hidden_states(path)
        noise = payload.get('noise', {}) if isinstance(payload, dict) else {}
        row = {
            'candidate': str(path),
            'reference': str(args.original),
            'model_id': payload.get('model_id', ''),
            'layer_idx': payload.get('layer_idx', ''),
            'reference_model_id': reference_payload.get('model_id', ''),
            'reference_layer_idx': reference_payload.get('layer_idx', ''),
            'noise_kind': noise.get('kind', ''),
            'noise_level': noise.get('level', ''),
            'noise_std': noise.get('std', ''),
            **metrics(reference, candidate),
        }
        rows.append(row)
        for token_row in token_metrics(reference, candidate, input_ids):
            token_rows.append({
                'candidate': str(path),
                'noise_kind': noise.get('kind', ''),
                'noise_level': noise.get('level', ''),
                'noise_std': noise.get('std', ''),
                **token_row,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} comparisons to {args.output}')
    if args.token_output:
        args.token_output.parent.mkdir(parents=True, exist_ok=True)
        with args.token_output.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(token_rows[0]))
            writer.writeheader()
            writer.writerows(token_rows)
        print(f'Wrote {len(token_rows)} token comparisons to {args.token_output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
