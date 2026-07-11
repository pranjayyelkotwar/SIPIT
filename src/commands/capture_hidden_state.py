from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.datasets.dataset import DatasetCollection
from src.utils.model import hidden_states_from_input_ids, hidden_states_from_prompt, setup
from src.utils.parser import HiddenStateCaptureParser
from src.utils.utils import set_seed


def build_parser():
    return HiddenStateCaptureParser()


def _write_bundle_manifest(output_dir: Path, manifest: dict) -> None:
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _save_tensor_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _capture_prompt(
    output_dir: Path,
    prompt: str,
    model,
    tokenizer,
    layer_idx: int,
    model_id: str,
) -> list[dict]:
    hidden_states = hidden_states_from_prompt(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        require_grad=False,
        add_special_tokens=True,
    )
    if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
        hidden_states = hidden_states.squeeze(0)
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_attention_mask=False,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'][0].detach().cpu()  # type: ignore[index]
    artifact_path = output_dir / 'tensors' / 'prompt.pt'

    _save_tensor_artifact(
        artifact_path,
        {
            'kind': 'prompt',
            'model_id': model_id,
            'layer_idx': layer_idx,
            'prompt': prompt,
            'input_ids': input_ids,
            'hidden_states': hidden_states.detach().cpu(),
        },
    )

    return [
        {
            'kind': 'prompt',
            'file': str(artifact_path.relative_to(output_dir)),
            'token_length': int(input_ids.numel()),
            'hidden_state_shape': list(hidden_states.shape),
        }
    ]


def _capture_dataset(
    output_dir: Path,
    dataset_dir: Path,
    model,
    tokenizer,
    device: str,
    layer_idx: int,
    model_id: str,
) -> list[dict]:
    datasets = DatasetCollection.load(dataset_dir)
    artifacts: list[dict] = []

    for dataset_idx, dataset in enumerate(tqdm(datasets, desc='Capturing Hidden States')):
        dataset_name = datasets.dataset_names[dataset_idx]

        for sample_idx, token_ids in enumerate(dataset):
            token_ids_cpu = token_ids.detach().cpu()
            token_ids = token_ids_cpu.to(device)
            hidden_states = hidden_states_from_input_ids(
                input_ids=token_ids,
                model=model,
                layer_idx=layer_idx,
                require_grad=False,
            )
            artifact_path = output_dir / 'tensors' / f'{dataset_idx:04d}_{sample_idx:06d}.pt'

            _save_tensor_artifact(
                artifact_path,
                {
                    'kind': 'dataset-sample',
                    'model_id': model_id,
                    'layer_idx': layer_idx,
                    'dataset_name': dataset_name,
                    'sample_index': sample_idx,
                    'input_ids': token_ids_cpu,
                    'hidden_states': hidden_states.detach().cpu(),
                },
            )

            artifacts.append(
                {
                    'kind': 'dataset-sample',
                    'dataset_name': dataset_name,
                    'sample_index': sample_idx,
                    'file': str(artifact_path.relative_to(output_dir)),
                    'token_length': int(token_ids_cpu.numel()),
                    'hidden_state_shape': list(hidden_states.shape),
                    'decoded_text': tokenizer.decode(token_ids_cpu),
                }
            )

    return artifacts


def run(args) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer, _device, layer_idx = setup(
        model_id=args.model_id,
        precision=args.precision,
        layer_idx=args.layer_idx,
        print_stats=True,
    )

    if args.prompt is not None:
        artifacts = _capture_prompt(
            output_dir=output_dir,
            prompt=args.prompt,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            model_id=args.model_id,
        )
        capture_mode = 'prompt'
    else:
        artifacts = _capture_dataset(
            output_dir=output_dir,
            dataset_dir=Path(args.input),
            model=model,
            tokenizer=tokenizer,
            device=_device,
            layer_idx=layer_idx,
            model_id=args.model_id,
        )
        capture_mode = 'dataset'

    manifest = {
        'schema_version': 1,
        'command': 'capture-hidden-state',
        'capture_mode': capture_mode,
        'model_id': args.model_id,
        'precision': args.precision,
        'requested_layer_idx': args.layer_idx,
        'layer_idx': layer_idx,
        'seed': args.seed,
        'input': args.prompt if args.prompt is not None else str(Path(args.input)),
        'artifacts': artifacts,
    }
    _write_bundle_manifest(output_dir, manifest)

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())