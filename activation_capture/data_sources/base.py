from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


class BaseTextDataset(Dataset, ABC):
    dataset_name: str
    default_split: str

    def __init__(
        self,
        tokenizer,
        max_token_length: int,
        *,
        split: str | None = None,
        add_bos_token: bool = True,
        include_choices: bool = True,
        num_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.add_bos_token = add_bos_token
        self.include_choices = include_choices
        self.split = split or self.default_split
        logging.info("Loading %s (%s)", self.dataset_name, self.split)
        dataset = self.load_hf_dataset(self.split).filter(self.keep_example)
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        self.dataset = dataset

    @abstractmethod
    def load_hf_dataset(self, split: str) -> HFDataset:
        raise NotImplementedError

    @abstractmethod
    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def keep_example(self, record: dict[str, Any]) -> bool:
        image_keys = (
            "image", "images", "figure", "figures", "image_path", "image_url"
        )
        return not any(self._contains_payload(record.get(key)) for key in image_keys)

    @staticmethod
    def _contains_payload(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, bytes):
            return bool(value)
        if isinstance(value, dict):
            return any(BaseTextDataset._contains_payload(x) for x in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(BaseTextDataset._contains_payload(x) for x in value)
        return True

    @staticmethod
    def format_choices(choices: list[str] | None) -> str:
        if not choices:
            return ""
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(
            f"{labels[index]}. {choice}" for index, choice in enumerate(choices)
        )

    def build_prompt(self, question: str, choices: list[str] | None = None) -> str:
        lines = ["Answer the following question.", "", f"Question: {question.strip()}"]
        if self.include_choices and choices:
            lines.extend(["Choices:", self.format_choices(choices)])
        return "\n".join(lines)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=self.add_bos_token,
            truncation=True,
            max_length=self.max_token_length,
        )
        return list(tokens)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        metadata = self.normalize_record(self.dataset[idx], idx)
        tokens = self.encode(metadata["prompt_text"])
        metadata = {**metadata, "token_count": len(tokens)}
        return tokens, idx, len(tokens), metadata
