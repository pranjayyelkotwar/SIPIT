from pathlib import Path

from datasets.utils.logging import disable_progress_bar

_ = disable_progress_bar()

import random
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.datasets.dataset import DatasetCollection, RandomTokenizedDataset
from src.utils.parser import DatasetRandomParser
from src.utils.utils import set_seed


def build_parser():
    return DatasetRandomParser()


def run(args) -> int:
    DATA_DIR = Path(args.output)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tokenizers = {
        name: AutoTokenizer.from_pretrained(name)
        for name in tqdm(args.model_names, desc='Loading Tokenizers')
    }

    token_maps: dict[str, dict[str, int]] = defaultdict(dict)

    for model, tokenizer in tqdm(tokenizers.items(), desc='Processing Tokenizers'):
        vocab: dict[str, int] = tokenizer.get_vocab()

        for token, tok_id in vocab.items():
            token_maps[token][model] = tok_id

    # Keep only tokens that appear in ALL models
    overlapping_tokens = {tok: ids for tok, ids in token_maps.items() if len(ids) == len(args.model_names)}

    if len(overlapping_tokens) == 0:
        raise RuntimeError("No overlapping tokens across all models.")

    common_token_strs = sorted(list(overlapping_tokens.keys()))

    set_seed(args.seed)
    sampled_prompt_token_strs = [
        random.choices(common_token_strs, k=args.tokens)
        for _ in range(args.prompts)
    ]

    datasets = {}

    for model_name in args.model_names:
        token_ids_list = []

        for token_str_seq in sampled_prompt_token_strs:
            token_id_seq = [
                overlapping_tokens[tok_str][model_name]
                for tok_str in token_str_seq
            ]
            token_ids_list.append(torch.tensor(token_id_seq, dtype=torch.long))

        datasets[model_name.split('/')[-1]] = RandomTokenizedDataset(
            token_ids=token_ids_list,
            prompt_tokens=args.tokens,
        )

    dataset = DatasetCollection(
        group_name=args.dataset_name,
        datasets=list(datasets.values()),
        dataset_names=list(datasets.keys())
    )

    dataset.save(
        DATA_DIR / dataset.group_name,
        overwrite=True
    )

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
