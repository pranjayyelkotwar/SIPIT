#!/usr/bin/env python3
"""Create a reproducible noise sweep from a capture-hidden-state tensor.

Noise levels are fractions of the hidden tensor's RMS, so the sweep remains
comparable across models and layers. Each output is accepted directly by
``sipit --command invert-hidden-state``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def add_relative_gaussian_noise(
    hidden_states: torch.Tensor, level: float, generator: torch.Generator
) -> tuple[torch.Tensor, float]:
    if level < 0:
        raise ValueError('Noise levels must be non-negative.')
    source = hidden_states.detach().float().cpu()
    signal_rms = torch.sqrt(torch.mean(source.square())).item()
    noise_std = level * signal_rms
    noise = torch.randn(source.shape, generator=generator, dtype=source.dtype)
    return source + noise * noise_std, noise_std


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location='cpu')
    if torch.is_tensor(payload):
        return {'hidden_states': payload, 'source_file': str(path)}
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get('hidden_states')):
        raise TypeError(f'{path} must be a tensor or contain a hidden_states tensor.')
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='A capture-hidden-state .pt artifact.')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument(
        '--levels', type=float, nargs='+', default=[0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    )
    parser.add_argument('--seed', type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_payload(args.input)
    hidden_states = payload['hidden_states']
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    rows = []

    for index, level in enumerate(args.levels):
        noisy, noise_std = add_relative_gaussian_noise(hidden_states, level, generator)
        output_payload = dict(payload)
        output_payload['hidden_states'] = noisy.to(hidden_states.dtype)
        output_payload['noise'] = {
            'kind': 'relative_gaussian', 'level': level,
            'std': noise_std, 'seed': args.seed,
        }
        filename = f'noise_{index:02d}_level_{level:g}.pt'
        torch.save(output_payload, args.output / filename)
        rows.append({'file': filename, 'level': level, 'noise_std': noise_std})

    with (args.output / 'noise_sweep.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'level', 'noise_std'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} noisy tensors and noise_sweep.csv to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
