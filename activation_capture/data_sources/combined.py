from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import ConcatDataset, Dataset

from .questions import (
    ARCEasyDataset,
    HLEDataset,
    HLEMathShortAnswerDataset,
    HLENonMathShortAnswerDataset,
    MMLUDataset,
)


REGISTRY = {
    "arc_easy": ARCEasyDataset,
    "mmlu": MMLUDataset,
    "hle": HLEDataset,
    "hle_math_short_answer": HLEMathShortAnswerDataset,
    "hle_non_math_short_answer": HLENonMathShortAnswerDataset,
}


class CombinedQuestionDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self.concatenated = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.concatenated)

    def __getitem__(self, idx: int):
        tokens, _local_idx, length, metadata = self.concatenated[idx]
        return tokens, idx, length, {**metadata, "combined_dataset_idx": idx}


def build_combined_question_dataset(
    *,
    dataset_names: Sequence[str],
    tokenizer,
    max_token_length: int,
    add_bos_token: bool = True,
    include_choices: bool = True,
    num_samples: dict[str, int] | None = None,
    shuffle: bool = False,
    seed: int = 42,
    prompt_suffix: str | None = None,
) -> CombinedQuestionDataset:
    unknown = set(dataset_names) - set(REGISTRY)
    if unknown:
        raise ValueError(f"Unknown QA datasets: {sorted(unknown)}")
    datasets = [
        REGISTRY[name](
            tokenizer,
            max_token_length,
            add_bos_token=add_bos_token,
            include_choices=include_choices,
            num_samples=(num_samples or {}).get(name),
            shuffle=shuffle,
            seed=seed,
            prompt_suffix=prompt_suffix,
        )
        for name in dataset_names
    ]
    return CombinedQuestionDataset(datasets)
