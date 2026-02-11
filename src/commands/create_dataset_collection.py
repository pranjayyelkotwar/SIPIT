import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar

_ = disable_progress_bar()

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.datasets.dataset import TokenizedDataset, DatasetCollection
from src.utils.parser import DatasetCollectionParser
from src.utils.utils import set_seed


def download_dataset(
    output_dir: Path,
    dataset_dict: dict[str, Any],
    min_words: int = 0,
    overwrite: bool = False,
    output_text_column: str = 'text',
) -> None:
    if output_dir.exists() and not overwrite:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    args = dataset_dict['args']
    num_samples: int = dataset_dict['num_samples']
    text_column: str = dataset_dict['text_column']

    stream = load_dataset(*args, split='train', streaming=True, trust_remote_code=False)

    filtered_samples: list[dict[str, Any]] = []

    bar = tqdm(stream, desc=f"Filtering {output_dir.name}", leave=False, total=num_samples)
    for sample in bar:
        text = sample.get(text_column, "")

        if isinstance(text, str) and len(text.strip().split()) >= min_words:
            filtered_samples.append(sample)

        bar.set_postfix({'collected': len(filtered_samples)})

        if len(filtered_samples) >= num_samples:
            break

    if len(filtered_samples) < num_samples:
        raise ValueError(
            f"Only found {len(filtered_samples)} samples with at least {min_words} words "
            f"(needed {num_samples}) for {output_dir.name}."
        )

    dataset = Dataset.from_list(filtered_samples)

    if text_column != output_text_column:
        dataset = dataset.rename_column(text_column, output_text_column)

    dataset.save_to_disk(output_dir)


def random_subsequence(input_ids: torch.LongTensor, length: int) -> tuple[torch.LongTensor, int]:
    """
    Return a contiguous subsequence of `input_ids` with exactly `length` tokens,
    along with the chosen start index.

    Assumes input_ids.size(0) >= length.
    """
    start_idx = random.randint(0, input_ids.size(0) - length)
    return input_ids[start_idx:start_idx + length], start_idx # type: ignore


def tokenize_dataset_samples(
    tokenizer: PreTrainedTokenizer,
    saved_dataset: Dataset,
    text_column: str,
    prompt_tokens: int,
) -> tuple[list[torch.LongTensor], list[int], list[int]]:
    """
    Tokenize each sample in `saved_dataset[text_column]` and sample a random
    contiguous subsequence of length `prompt_tokens`.

    Returns:
        token_ids: list of (prompt_tokens,) LongTensors
        start_ids: chosen start indices (one per sample)
        sample_ids: original sample indices
    """
    token_ids: list[torch.LongTensor] = []
    start_ids: list[int] = []
    sample_ids: list[int] = []

    for sample_idx, sample in enumerate(saved_dataset[text_column]):
        enc = tokenizer(
            sample,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors='pt',
            truncation=True,
            max_length=min(getattr(tokenizer, 'model_max_length', 2048), 2048),  # type: ignore
        )
        input_ids: torch.LongTensor = enc['input_ids'][0] # type: ignore

        # You stated you know it's always >= prompt_tokens
        subseq, start_idx = random_subsequence(input_ids, prompt_tokens)

        token_ids.append(subseq)
        start_ids.append(start_idx)
        sample_ids.append(sample_idx)

    return token_ids, start_ids, sample_ids


def build_tokenized_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    dataset_dir: Path,
    text_column: str,
    prompt_tokens: int,
) -> TokenizedDataset:
    """
    Load a HF dataset from disk and convert it into your TokenizedDataset format.
    """
    saved_dataset: Dataset = load_from_disk(dataset_dir) # type: ignore

    token_ids, start_ids, sample_ids = tokenize_dataset_samples(
        tokenizer=tokenizer,
        saved_dataset=saved_dataset,
        text_column=text_column,
        prompt_tokens=prompt_tokens,
    )

    return TokenizedDataset(
        dataset_name=dataset_name,
        prompt_tokens=prompt_tokens,
        token_ids=token_ids,
        start_ids=start_ids,
        sample_ids=sample_ids,
    )


def build_parser():
    return DatasetCollectionParser()


def run(args) -> int:
    ROOT = Path(args.output)
    ROOT.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.dataset_config)

    if not config_path.exists():
        raise FileNotFoundError(f'Dataset config not found: {config_path}')

    with open(config_path, 'r') as f:
        dataset_dicts: dict[str, dict[str, Any]] = json.load(f)

    set_seed(args.seed)
    for name, dataset_dict in tqdm(dataset_dicts.items(), desc='Downloading Datasets'):
        dataset_dir = ROOT / name
        download_dataset(
            output_dir=dataset_dir,
            dataset_dict=dataset_dict,
            min_words=args.tokens,
            overwrite=args.overwrite,
            output_text_column=args.output_text_column
        )

    tokenizers = {
        name: AutoTokenizer.from_pretrained(name)
        for name in tqdm(args.model_names, desc='Loading Tokenizers')
    }

    set_seed(args.seed)
    for tokenizer_name, tokenizer in tqdm(tokenizers.items(), desc='Creating Datasets'):
        datasets: dict[str, TokenizedDataset] = {}

        for dataset_name in dataset_dicts.keys():
            dataset_dir = ROOT / dataset_name

            datasets[dataset_name] = build_tokenized_dataset(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                text_column=args.output_text_column,
                prompt_tokens=args.tokens,
            )

        dataset = DatasetCollection(
            group_name=tokenizer_name.split('/')[-1],
            datasets=list(datasets.values()),
            dataset_names=list(datasets.keys()),
        )

        dataset.save(
            ROOT / args.dataset_name / dataset.group_name,
            overwrite=True
        )

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
