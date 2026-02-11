from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, NamedTuple, Optional, TypedDict, Union

import numpy as np
import torch
from tqdm import tqdm

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from src.utils.utils import set_seed


DatasetLike = Union["TokenizedDataset", "RandomTokenizedDataset"]


class TokenizedDataset(NamedTuple):
    dataset_name: str
    prompt_tokens: int # fixed length for every 1-D tensor
    token_ids: list[torch.LongTensor]
    start_ids: list[int]
    sample_ids: list[int]

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[torch.LongTensor]: # type: ignore
        return iter(self.token_ids)

    def to_json(self, output_path: Path):
        """
        Serialize to JSON.
        Validates:
          - token_ids/start_ids/sample_ids have the same number of samples
          - prompt_tokens is a positive int
          - every token_ids[i] is 1-D and has length == prompt_tokens
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not (len(self.token_ids) == len(self.start_ids) == len(self.sample_ids)):
            raise ValueError("token_ids, start_ids, and sample_ids must have the same length.")

        L = int(self.prompt_tokens)
        if L <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")

        lengths = []
        for i, t in enumerate(self.token_ids):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"token_ids[{i}] must be a torch.Tensor.")
            if t.ndim != 1:
                raise ValueError(f"token_ids[{i}] must be 1-D; got shape {tuple(t.shape)}.")
            n = int(t.numel())
            lengths.append(n)
            if n != L:
                raise ValueError(
                    f"token_ids[{i}] length {n} != prompt_tokens {L} (all sequences must match)."
                )

        payload = {
            "dataset_name": self.dataset_name,
            "prompt_tokens": L,
            "token_ids": [t.detach().cpu().tolist() for t in self.token_ids],
            "start_ids": [int(x) for x in self.start_ids],
            "sample_ids": [int(x) for x in self.sample_ids],
        }

        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return p

    @classmethod
    def from_json(cls, input_path: Path | str) -> "TokenizedDataset":
        """
        Load from JSON (no inference):
          - requires 'prompt_tokens' (int)
          - reconstructs token_ids as torch.LongTensor
          - checks every tensor is 1-D and length == prompt_tokens
          - checks list lengths match across fields
        """
        p = Path(input_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Required keys only — no back-compat/inference
        for key in ("dataset_name", "prompt_tokens", "token_ids", "start_ids", "sample_ids"):
            if key not in data:
                raise KeyError(f"Missing '{key}' in {p}.")

        dataset_name = str(data["dataset_name"])
        prompt_tokens = int(data["prompt_tokens"])
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")

        token_ids_raw = data["token_ids"]
        if not isinstance(token_ids_raw, list):
            raise TypeError("`token_ids` must be a list of sequences.")

        token_ids: list[torch.LongTensor] = []
        lengths: list[int] = []
        for i, seq in enumerate(token_ids_raw):
            try:
                t: torch.LongTensor = torch.tensor(list(seq), dtype=torch.long) # type: ignore
            except Exception as ex:
                raise TypeError(f"`token_ids[{i}]` must be a sequence of ints.") from ex
            if t.ndim != 1:
                raise ValueError(f"`token_ids[{i}]` must be 1-D; got shape {tuple(t.shape)}.")
            n = int(t.numel())
            lengths.append(n)
            if n != prompt_tokens:
                raise ValueError(
                    f"token_ids[{i}] length {n} != prompt_tokens {prompt_tokens}."
                )
            token_ids.append(t)

        start_ids = [int(x) for x in data["start_ids"]]
        sample_ids = [int(x) for x in data["sample_ids"]]

        if not (len(token_ids) == len(start_ids) == len(sample_ids)):
            raise ValueError("token_ids, start_ids, and sample_ids must have the same length.")

        return cls(
            dataset_name=dataset_name,
            prompt_tokens=prompt_tokens,
            token_ids=token_ids,
            start_ids=start_ids,
            sample_ids=sample_ids,
        )


class RandomTokenizedDataset(NamedTuple):
    prompt_tokens: int
    token_ids: list[torch.LongTensor]

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[torch.LongTensor]: # type: ignore
        return iter(self.token_ids)

    def to_json(self, output_path: Path):
        """
        Serialize to JSON.
        Validates:
          - prompt_tokens is a positive int
          - every token_ids[i] is a 1-D tensor of length == prompt_tokens
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        L = int(self.prompt_tokens)
        if L <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")

        for i, t in enumerate(self.token_ids):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"token_ids[{i}] must be a torch.Tensor.")
            if t.ndim != 1:
                raise ValueError(f"token_ids[{i}] must be 1-D; got shape {tuple(t.shape)}.")
            n = int(t.numel())
            if n != L:
                raise ValueError(
                    f"token_ids[{i}] length {n} != prompt_tokens {L} (all sequences must match)."
                )

        payload = {
            "prompt_tokens": L,
            "token_ids": [t.detach().cpu().tolist() for t in self.token_ids],
        }

        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return p

    @classmethod
    def from_json(cls, input_path: Path | str) -> "RandomTokenizedDataset":
        """
        Load from JSON (no inference):
          - requires 'prompt_tokens' (int) and 'token_ids' (list of sequences)
          - reconstructs token_ids as torch.LongTensor
          - checks every tensor is 1-D and length == prompt_tokens
        """
        p = Path(input_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ("prompt_tokens", "token_ids"):
            if key not in data:
                raise KeyError(f"Missing '{key}' in {p}.")

        prompt_tokens = int(data["prompt_tokens"])
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be a positive integer.")

        token_ids_raw = data["token_ids"]
        if not isinstance(token_ids_raw, list):
            raise TypeError("`token_ids` must be a list of sequences.")

        token_ids: list[torch.LongTensor] = []
        for i, seq in enumerate(token_ids_raw):
            try:
                t: torch.LongTensor = torch.tensor(list(seq), dtype=torch.long) # type: ignore
            except Exception as ex:
                raise TypeError(f"`token_ids[{i}]` must be a sequence of ints.") from ex
            if t.ndim != 1:
                raise ValueError(f"`token_ids[{i}]` must be 1-D; got shape {tuple(t.shape)}.")
            n = int(t.numel())
            if n != prompt_tokens:
                raise ValueError(
                    f"`token_ids[{i}]` length {n} != prompt_tokens {prompt_tokens}."
                )
            token_ids.append(t)

        return cls(prompt_tokens=prompt_tokens, token_ids=token_ids)


class _ManifestItem(TypedDict, total=True):
    kind: Literal["tokenized", "random"]
    file: str
    prompt_tokens: int
    num_samples: int


class _Manifest(TypedDict, total=True):
    schema_version: int
    group_name: str
    created_at: str
    dataset_names: list[str]
    items: list[_ManifestItem]


def _ensure_json_ext(name: str) -> str:
    name = str(name)
    return name if name.endswith(".json") else f"{name}.json"


def _auto_filename(index: int, ds: DatasetLike) -> str:
    """
    Generate filenames that match the screenshot pattern exactly:
      - TokenizedDataset with ds.dataset_name -> '{idx:04d}_{name}_tokenized.json'
      - RandomTokenizedDataset -> '{idx:04d}_random-{idx:04d}_random.json'
    No slugifying: names are used verbatim.
    """
    if ds.__class__.__name__ == "TokenizedDataset":
        base = getattr(ds, "dataset_name", None) or "dataset"
        return f"{index:04d}_{base}_tokenized.json"
    elif ds.__class__.__name__ == "RandomTokenizedDataset":
        return f"{index:04d}_random-{index:04d}_random.json"
    else:
        raise TypeError(f"Unsupported dataset type: {type(ds)}")


class DatasetCollection:
    """
    A collection of datasets that persists/loads using a manifest.
    `dataset_names` holds the exact **filenames** (including '.json').

    Directory layout:
      <dir>/
        manifest.json
        <dataset_names[0]>
        <dataset_names[1]>
        ...
    """

    def __init__(
        self,
        group_name: str,
        datasets: Optional[list[DatasetLike]] = None,
        dataset_names: Optional[list[str]] = None,   # exact filenames
    ):
        self.group_name = str(group_name)
        self.datasets: list[DatasetLike] = list(datasets or [])

        if dataset_names is None:
            self.dataset_names: list[str] = [
                _auto_filename(i, ds) for i, ds in enumerate(self.datasets)
            ]
        else:
            if len(dataset_names) != len(self.datasets):
                raise ValueError("dataset_names must have the same length as datasets.")
            # normalize to include .json
            self.dataset_names = [_ensure_json_ext(n) for n in dataset_names]

        self._validate_invariants()

    # -------------- Python protocol --------------
    def __len__(self) -> int:
        return len(self.datasets)

    def __iter__(self) -> Iterator[DatasetLike]:
        return iter(self.datasets)

    # -------------- Mutations --------------
    def append(self, ds: DatasetLike, name: Optional[str] = None) -> None:
        """
        Append a dataset with an optional filename (include .json or not).
        If name is omitted, we generate a filename matching the screenshot pattern.
        """
        idx = len(self.datasets)
        filename = _ensure_json_ext(name) if name else _auto_filename(idx, ds)
        self.datasets.append(ds)
        self.dataset_names.append(filename)
        self._validate_invariants()

    # -------------- Persistence --------------
    def save(self, dir_path: Path | str, overwrite: bool = False) -> Path:
        """
        Save all datasets to files named exactly as `dataset_names`.
        Writes schema v3 manifest that includes the same names.
        """
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)

        manifest_path = d / "manifest.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists. Pass overwrite=True to replace it."
            )

        # write datasets
        items: list[_ManifestItem] = []
        for name, ds in zip(self.dataset_names, self.datasets):
            path = d / name
            kind: Literal["tokenized", "random"]
            if ds.__class__.__name__ == "TokenizedDataset":
                kind = "tokenized"
            elif ds.__class__.__name__ == "RandomTokenizedDataset":
                kind = "random"
            else:
                raise TypeError(f"Unsupported dataset type: {type(ds)}")

            ds.to_json(path)  # runs each dataset's own validation

            items.append(
                _ManifestItem(
                    kind=kind,
                    file=name,
                    prompt_tokens=int(getattr(ds, "prompt_tokens")),
                    num_samples=int(len(ds)),
                )
            )

        manifest: _Manifest = _Manifest(
            schema_version=3,
            group_name=self.group_name,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            dataset_names=list(self.dataset_names),   # exact filenames
            items=items,
        )

        with (d / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return d

    @classmethod
    def load(cls, dir_path: Path | str) -> "DatasetCollection":
        """
        Load a collection from directory with manifest.json.

        - schema_version == 3 (preferred): uses top-level dataset_names directly.
        - schema_version == 2 or 1: falls back to using item['file'] as the name.
        """
        d = Path(dir_path)
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {d}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        schema_version = int(manifest.get("schema_version", 1))
        group_name = str(manifest.get("group_name", "dataset_collection"))
        items = manifest.get("items")
        if not isinstance(items, list):
            raise ValueError("Invalid manifest: 'items' must be a list.")

        # names to use
        if schema_version >= 3 and "dataset_names" in manifest:
            dataset_names = [str(n) for n in manifest["dataset_names"]]
        else:
            # v1/v2 back-compat: use the filenames listed in items
            dataset_names = [str(it["file"]) for it in items]

        # load datasets
        datasets: list[DatasetLike] = []
        for it in items:
            file_rel = str(it["file"])
            kind = str(it["kind"])
            file_path = d / file_rel

            if kind == "tokenized":
                ds = TokenizedDataset.from_json(file_path)
            elif kind == "random":
                ds = RandomTokenizedDataset.from_json(file_path)
            else:
                raise ValueError(f"Unknown dataset kind in manifest: {kind!r}")

            datasets.append(ds)

        return cls(group_name=group_name, datasets=datasets, dataset_names=dataset_names)

    # -------------- Utilities --------------
    def summary(self) -> list[dict]:
        """
        Quick summary, including the exact filename used for each dataset.
        """
        out: list[dict] = []
        for name, ds in zip(self.dataset_names, self.datasets):
            out.append(
                {
                    "filename": name,
                    "kind": "TokenizedDataset" if ds.__class__.__name__ == "TokenizedDataset" else "RandomTokenizedDataset",
                    "prompt_tokens": int(getattr(ds, "prompt_tokens")),
                    "num_samples": int(len(ds)),
                }
            )
        return out

    def _validate_invariants(self) -> None:
        if len(self.datasets) != len(self.dataset_names):
            raise ValueError("datasets and dataset_names must have the same length.")
        # must be non-empty, unique, and end with .json
        seen = set()
        for i, nm in enumerate(self.dataset_names):
            if not isinstance(nm, str) or not nm.strip():
                raise ValueError(f"dataset_names[{i}] must be a non-empty string.")
            if not nm.endswith(".json"):
                raise ValueError(f"dataset_names[{i}] must end with .json (got {nm!r}).")
            if nm in seen:
                raise ValueError(f"Duplicate filename in dataset_names: {nm!r}")
            seen.add(nm)



def extract_tokenized_prompts(
    dataset_path: Path,
    text_column: str,

    tokenizer: AutoTokenizer,

    prompt_tokens: int,
    max_prompts: int,
    batch_size: int,

    seed: int = 8,
):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    ds: Dataset = load_from_disk(str(dataset_path)) # type: ignore

    # we’ll take at most this many *eligible* samples
    total_to_take = min(max_prompts, len(ds))
    taken_so_far = 0

    # Basic guardrails
    if prompt_tokens <= 0:
        raise ValueError('prompt_tokens must be > 0')
    if batch_size <= 0:
        raise ValueError('batch_size must be > 0')

    # Truncation budget
    max_len = min(getattr(tokenizer, 'model_max_length', 1000) or 1000, 1000)

    num_batches = math.ceil(total_to_take / batch_size)

    # Build per-sample random contiguous windows of length prompt_tokens
    token_ids: list[torch.LongTensor] = []
    start_ids: list[int] = []
    sample_ids: list[int] = []

    for b in tqdm(range(num_batches), desc='Processing batches'):
        if taken_so_far >= total_to_take:
            break

        start_idx_ds = b * batch_size
        end_idx_ds = min((b + 1) * batch_size, len(ds))

        # Cap by remaining prompts we still want
        remaining = total_to_take - taken_so_far
        if (end_idx_ds - start_idx_ds) > remaining:
            end_idx_ds = start_idx_ds + remaining

        batch = ds.select(range(start_idx_ds, end_idx_ds))
        texts: list[str] = list(batch[text_column])

        # Tokenize without special tokens; truncate to max_len
        enc = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
        ) # type: ignore
        input_ids_list: list[list[int]] = enc['input_ids']

        for local_i, ids in enumerate(input_ids_list):
            L = len(ids)
            
            # skip too-short sample
            if L < prompt_tokens:
                continue

            # choose start uniformly among valid windows
            s = int(rng.integers(0, L - prompt_tokens + 1))
            window = ids[s : s + prompt_tokens]

            token_ids.append(torch.tensor(window, dtype=torch.long)) # type: ignore
            start_ids.append(s)
            sample_ids.append(start_idx_ds + local_i) # index w.r.t. full dataset

        if not token_ids:
            # no eligible samples in this batch; continue
            continue

        # Cap within this batch if we already hit the global limit
        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]
            start_ids = start_ids[:remaining]
            sample_ids = sample_ids[:remaining]

    return TokenizedDataset(
        dataset_path.stem, 
        prompt_tokens, 
        token_ids, 
        start_ids, 
        sample_ids
    )


def random_tokenized_prompts(
    tokenizer: AutoTokenizer,

    prompt_tokens: int,
    max_prompts: int,

    seed: int = 8,
):
    set_seed(seed)

    if prompt_tokens <= 0:
        raise ValueError('`prompt_tokens` must be > 0')

    vocab = tokenizer.get_vocab() # type: ignore # dict: token -> id
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    special = set(tokenizer.all_special_tokens) # type: ignore

    valid_tokens = [
        idx for idx, tok in id_to_token.items()
        if tok not in special and not tok.startswith("##")
    ]

    token_ids: list[torch.LongTensor] = [
        torch.tensor(
            random.choices(valid_tokens, k=prompt_tokens), 
            dtype=torch.long
        ) 
        for _ in range(max_prompts)
    ] # type: ignore

    return RandomTokenizedDataset(prompt_tokens, token_ids)

