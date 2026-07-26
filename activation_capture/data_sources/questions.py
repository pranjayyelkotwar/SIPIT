from __future__ import annotations

from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from .base import BaseTextDataset


class ARCEasyDataset(BaseTextDataset):
    dataset_name = "allenai/ai2_arc:ARC-Easy"
    default_split = "train"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        pairs = zip(
            record["choices"]["label"], record["choices"]["text"], strict=True
        )
        choices = [text for _, text in sorted(pairs, key=lambda item: item[0])]
        return {
            "prompt_text": self.build_prompt(record["question"], choices),
            "question_text": record["question"],
            "difficulty_label": "easy",
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": record.get("id", str(idx)),
            "subject": "science",
            "question_type": "multiple_choice",
            "choices": choices,
            "gold_answer": record.get("answerKey"),
            "has_image": False,
        }


class MMLUDataset(BaseTextDataset):
    dataset_name = "cais/mmlu"
    default_split = "test"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("cais/mmlu", "all", split=split)

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        answer = record.get("answer")
        if isinstance(answer, int):
            answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer]
        choices = record.get("choices")
        return {
            "prompt_text": self.build_prompt(record["question"], choices),
            "question_text": record["question"],
            "difficulty_label": "medium",
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": str(idx),
            "subject": record.get("subject", "unknown"),
            "question_type": "multiple_choice",
            "choices": choices,
            "gold_answer": answer,
            "has_image": False,
        }


class HLEDataset(BaseTextDataset):
    dataset_name = "cais/hle"
    default_split = "test"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("cais/hle", split=split)

    def keep_example(self, record: dict[str, Any]) -> bool:
        return record.get("image", "") == ""

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        choices = (
            record.get("choices")
            or record.get("options")
            or record.get("multiple_choice_targets")
        )
        answer = (
            record.get("answer")
            or record.get("answer_key")
            or record.get("correct_answer")
            or record.get("solution")
        )
        source_id = (
            record.get("question_id")
            or record.get("id")
            or record.get("uid")
            or str(idx)
        )
        subject = (
            record.get("subject")
            or record.get("category")
            or record.get("raw_subject")
            or record.get("field")
            or "unknown"
        )
        return {
            "prompt_text": self.build_prompt(record["question"], choices),
            "question_text": record["question"],
            "difficulty_label": "hard",
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": source_id,
            "subject": subject,
            "question_type": record.get("answer_type")
            or ("multiple_choice" if choices else "short_answer"),
            "choices": choices,
            "gold_answer": answer,
            "has_image": False,
        }
