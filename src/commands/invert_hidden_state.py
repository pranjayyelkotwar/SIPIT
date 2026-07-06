from __future__ import annotations

import json
from pathlib import Path

import torch

from src.algorithm import BruteForce, HardPrompts, SIPIT, InversionAlgorithm
from src.utils.model import setup
from src.utils.parser import HiddenStateInversionParser
from src.utils.utils import set_seed


def build_parser():
    return HiddenStateInversionParser()


def _load_tensor_payload(input_path: Path) -> tuple[torch.Tensor, dict]:
    if input_path.is_dir():
        manifest_path = input_path / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Missing manifest.json in {input_path}')

        with manifest_path.open('r', encoding='utf-8') as f:
            manifest = json.load(f)

        artifacts = manifest.get('artifacts', [])
        if not artifacts:
            raise ValueError(f'No tensor artifacts recorded in {manifest_path}')

        first_artifact = input_path / str(artifacts[0]['file'])
        payload = torch.load(first_artifact, map_location='cpu')
        if isinstance(payload, dict) and 'hidden_states' in payload:
            return payload['hidden_states'], payload
        if torch.is_tensor(payload):
            return payload, {'file': str(first_artifact)}
        raise TypeError(f'Unsupported tensor payload in {first_artifact}')

    payload = torch.load(input_path, map_location='cpu')
    if isinstance(payload, dict) and 'hidden_states' in payload:
        return payload['hidden_states'], payload
    if torch.is_tensor(payload):
        return payload, {'file': str(input_path)}
    raise TypeError(f'Unsupported tensor payload in {input_path}')


def _select_algorithm(args) -> type[InversionAlgorithm]:
    inversion_method_dict: dict[str, type[InversionAlgorithm]] = {
        'SIPIT': SIPIT,
        'BruteForce': BruteForce,
        'HardPrompts': HardPrompts,
    }
    return inversion_method_dict[args.method]


def run(args) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    model, tokenizer, _device, layer_idx = setup(
        model_id=args.model_id,
        precision=args.precision,
        layer_idx=args.layer_idx,
        print_stats=True,
    )

    hidden_states, payload = _load_tensor_payload(input_path)
    if hidden_states.dim() == 1:
        hidden_states = hidden_states.unsqueeze(0)

    algorithm_cls = _select_algorithm(args)
    inversion_algorithm = algorithm_cls(
        log_dir=args.log_file_path,
        log_name=args.log_file_name,
        special_start_token_id=args.special_start_token,
        use_scheduler=args.scheduler,
        projection_iters_base=args.projection_iters_base,
        vocab_scale_factor=args.vocab_scale_factor,
    )

    inversion_time, discovered_ids, timesteps, times = inversion_algorithm.find_prompt(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        target_hidden_states=hidden_states,
        step_size=args.step_size,
    )

    if any(x is None for x in (inversion_time, discovered_ids, timesteps, times)):
        print('Inversion Failed!')
        return 1

    prompt_token_ids = discovered_ids
    if args.special_start_token is not None and len(prompt_token_ids) > 0:
        prompt_token_ids = prompt_token_ids[1:]

    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
    output_path.write_text(prompt_text, encoding='utf-8')

    print(f'Recovered prompt written to {output_path}')
    if isinstance(payload, dict) and 'prompt' in payload:
        print(f'Original prompt: {payload["prompt"]}')
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())