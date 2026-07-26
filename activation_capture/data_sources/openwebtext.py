from __future__ import annotations

from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from .base import BaseTextDataset


class OpenWebTextSentencesDataset(BaseTextDataset):
    dataset_name = "paulpauls/openwebtext-sentences"
    default_split = "train"

    def __init__(self, *args, shuffle: bool = False, seed: int = 42, **kwargs):
        self.shuffle = shuffle
        self.seed = seed
        super().__init__(*args, **kwargs)

    def load_hf_dataset(self, split: str) -> HFDataset:
        dataset = load_dataset(self.dataset_name, split=split)
        return dataset.shuffle(seed=self.seed) if self.shuffle else dataset

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        text = record["text"]
        return {
            "prompt_text": text,
            "question_text": None,
            "difficulty_label": None,
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": str(idx),
            "subject": None,
            "question_type": "text",
            "choices": None,
            "gold_answer": None,
            "has_image": False,
        }
