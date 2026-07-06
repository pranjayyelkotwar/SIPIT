from pathlib import Path

import torch

from src.commands import capture_hidden_state, invert_hidden_state
from src.datasets.dataset import DatasetCollection, TokenizedDataset


PROMPT = 'Hello world'


def _load_tensor_artifact(path: Path) -> dict:
    return torch.load(path, map_location='cpu')


class TestHiddenStateCaptureSingle:
    def test_creates_bundle(self, tmp_data_dir, gpt2):
        parser = capture_hidden_state.build_parser()
        output_dir = tmp_data_dir / 'single-bundle'
        args = parser.parse_args([
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '-1',
            '--output', str(output_dir),
            '--prompt', PROMPT,
        ])

        ret = capture_hidden_state.run(args)
        assert ret == 0

        manifest = output_dir / 'manifest.json'
        tensor_file = output_dir / 'tensors' / 'prompt.pt'
        assert manifest.exists()
        assert tensor_file.exists()

        payload = _load_tensor_artifact(tensor_file)
        assert payload['kind'] == 'prompt'
        assert payload['hidden_states'].ndim == 2
        assert payload['input_ids'].ndim == 1

    def test_layer_selection_changes_output(self, tmp_data_dir, gpt2):
        parser = capture_hidden_state.build_parser()
        output_a = tmp_data_dir / 'layer-a'
        output_b = tmp_data_dir / 'layer-b'

        args_a = parser.parse_args([
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '0',
            '--output', str(output_a),
            '--prompt', PROMPT,
        ])
        args_b = parser.parse_args([
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '-1',
            '--output', str(output_b),
            '--prompt', PROMPT,
        ])

        capture_hidden_state.run(args_a)
        capture_hidden_state.run(args_b)

        tensor_a = _load_tensor_artifact(output_a / 'tensors' / 'prompt.pt')['hidden_states']
        tensor_b = _load_tensor_artifact(output_b / 'tensors' / 'prompt.pt')['hidden_states']

        assert tensor_a.shape == tensor_b.shape
        assert not torch.allclose(tensor_a, tensor_b)


class TestHiddenStateCaptureDataset:
    def test_creates_dataset_bundle(self, tmp_data_dir, gpt2):
        _, tokenizer, _, _ = gpt2

        base_tokens = torch.tensor(
            tokenizer(PROMPT, add_special_tokens=False)['input_ids'],
            dtype=torch.long,
        )
        token_ids = [base_tokens.clone(), base_tokens.clone()]
        dataset = TokenizedDataset(
            dataset_name='toy',
            prompt_tokens=max(len(t) for t in token_ids),
            token_ids=token_ids,
            start_ids=[0, 0],
            sample_ids=[0, 1],
        )
        dataset_dir = tmp_data_dir / 'dataset'
        DatasetCollection(group_name='toy-group', datasets=[dataset], dataset_names=['toy']).save(dataset_dir)

        parser = capture_hidden_state.build_parser()
        output_dir = tmp_data_dir / 'dataset-bundle'
        args = parser.parse_args([
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '-1',
            '--output', str(output_dir),
            '--input', str(dataset_dir),
        ])

        ret = capture_hidden_state.run(args)
        assert ret == 0

        manifest = output_dir / 'manifest.json'
        assert manifest.exists()
        tensor_files = sorted((output_dir / 'tensors').glob('*.pt'))
        assert len(tensor_files) == 2

        payload = _load_tensor_artifact(tensor_files[0])
        assert payload['kind'] == 'dataset-sample'
        assert payload['hidden_states'].ndim == 2


class TestHiddenStateInversion:
    def test_round_trip_prompt_tensor(self, tmp_data_dir, gpt2):
        capture_parser = capture_hidden_state.build_parser()
        capture_dir = tmp_data_dir / 'capture'
        prompt = 'Hello world'
        capture_args = capture_parser.parse_args([
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '-1',
            '--output', str(capture_dir),
            '--prompt', prompt,
        ])
        capture_hidden_state.run(capture_args)

        tensor_path = capture_dir / 'tensors' / 'prompt.pt'
        output_path = tmp_data_dir / 'recovered.txt'

        parser = invert_hidden_state.build_parser()
        args = parser.parse_args([
            '--method', 'SIPIT',
            '--model-id', 'openai-community/gpt2',
            '--layer-idx', '-1',
            '--output', str(output_path),
            '--input', str(tensor_path),
        ])
        ret = invert_hidden_state.run(args)

        assert ret == 0
        assert output_path.exists()
        assert output_path.read_text(encoding='utf-8') == prompt