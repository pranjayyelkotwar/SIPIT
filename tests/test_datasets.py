import json

from src.commands import create_dataset_random, create_dataset_collection, inversion_dataset
from src.datasets.dataset import DatasetCollection
SEED = 1234
SEED_STR = str(SEED)

TEST_COLLECTION_CONFIG = {
    "wikipedia": {
        "args": ["wikimedia/wikipedia", "20231101.en"],
        "num_samples": 1,
        "text_column": "text"
    },
    "colossal_clean_crawled_corpus": {
        "args": ["allenai/c4", "en"],
        "num_samples": 1,
        "text_column": "text"
    },
    "arxiv_pile": {
        "args": ["timaeus/pile-arxiv"],
        "num_samples": 1,
        "text_column": "text"
    },
    "github_python": {
        "args": ["angie-chen55/python-github-code"],
        "num_samples": 1,
        "text_column": "code"
    }
}


class TestCreateRandomDataset:
    def test_creates_manifest(self, tmp_data_dir):
        parser = create_dataset_random.build_parser()
        args = parser.parse_args([
            '--dataset-name', 'Test-Random',
            '--model-names', 'openai-community/gpt2',
            '--prompts', '3',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        ret = create_dataset_random.run(args)
        assert ret == 0

        manifest = tmp_data_dir / 'Test-Random' / 'manifest.json'
        assert manifest.exists()

    def test_dataset_loadable(self, tmp_data_dir):
        parser = create_dataset_random.build_parser()
        args = parser.parse_args([
            '--dataset-name', 'Test-Random',
            '--model-names', 'openai-community/gpt2',
            '--prompts', '3',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        create_dataset_random.run(args)

        ds = DatasetCollection.load(tmp_data_dir / 'Test-Random')
        assert len(ds.datasets) > 0
        assert len(ds.datasets[0]) == 3


class TestCreateCollectionDataset:
    def test_creates_manifest(self, tmp_data_dir):
        config_path = tmp_data_dir / 'test_datasets.json'
        with open(config_path, 'w') as f:
            json.dump(TEST_COLLECTION_CONFIG, f)

        parser = create_dataset_collection.build_parser()
        args = parser.parse_args([
            '--dataset-config', str(config_path),
            '--dataset-name', 'Test-Collection',
            '--model-names', 'openai-community/gpt2',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        ret = create_dataset_collection.run(args)
        assert ret == 0

        manifest = tmp_data_dir / 'Test-Collection' / 'gpt2' / 'manifest.json'
        assert manifest.exists()

    def test_dataset_loadable(self, tmp_data_dir):
        config_path = tmp_data_dir / 'test_datasets.json'
        with open(config_path, 'w') as f:
            json.dump(TEST_COLLECTION_CONFIG, f)

        parser = create_dataset_collection.build_parser()
        args = parser.parse_args([
            '--dataset-config', str(config_path),
            '--dataset-name', 'Test-Collection',
            '--model-names', 'openai-community/gpt2',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        create_dataset_collection.run(args)

        ds = DatasetCollection.load(tmp_data_dir / 'Test-Collection' / 'gpt2')
        assert len(ds.datasets) == 4
        for dataset in ds.datasets:
            assert len(dataset) == 1


class TestInvertDataset:
    def test_random_produces_csv(self, tmp_data_dir, gpt2):
        # Create a small random dataset
        parser = create_dataset_random.build_parser()
        args = parser.parse_args([
            '--dataset-name', 'Test-Random',
            '--model-names', 'openai-community/gpt2',
            '--prompts', '2',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        create_dataset_random.run(args)

        # Run inversion on it
        csv_path = tmp_data_dir / 'results_random.csv'
        parser = inversion_dataset.build_parser()
        args = parser.parse_args([
            '--method', 'SIPIT',
            '-i', str(tmp_data_dir / 'Test-Random'),
            '-o', str(csv_path),
            '--model-id', 'openai-community/gpt2',
            '--seed', SEED_STR,
            '-n', '2',
        ])
        ret = inversion_dataset.run(args)
        assert ret == 0
        assert csv_path.exists()

    def test_collection_produces_csv(self, tmp_data_dir, gpt2):
        # Create a collection dataset
        config_path = tmp_data_dir / 'test_datasets.json'
        with open(config_path, 'w') as f:
            json.dump(TEST_COLLECTION_CONFIG, f)

        parser = create_dataset_collection.build_parser()
        args = parser.parse_args([
            '--dataset-config', str(config_path),
            '--dataset-name', 'Test-Collection',
            '--model-names', 'openai-community/gpt2',
            '--tokens', '2',
            '--seed', SEED_STR,
            '-o', str(tmp_data_dir),
        ])
        create_dataset_collection.run(args)

        # Run inversion on it
        csv_path = tmp_data_dir / 'results_collection.csv'
        parser = inversion_dataset.build_parser()
        args = parser.parse_args([
            '--method', 'SIPIT',
            '-i', str(tmp_data_dir / 'Test-Collection' / 'gpt2'),
            '-o', str(csv_path),
            '--model-id', 'openai-community/gpt2',
            '--seed', SEED_STR,
            '-n', '4',
        ])
        ret = inversion_dataset.run(args)
        assert ret == 0
        assert csv_path.exists()
