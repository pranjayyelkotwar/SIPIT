#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from capture import ActivationWriter, CaptureBatch, ResidualPostCapture, merge_metadata
from data_sources import OpenWebTextSentencesDataset, build_combined_question_dataset


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("At least one capture layer is required.")
    return values


def parse_limits(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    result = {}
    for item in raw.split(","):
        name, value = item.split(":", 1)
        result[name.strip()] = int(value)
    return result


def collate(tokenizer):
    def collate_batch(batch):
        max_length = max(length for _, _, length, _ in batch)
        pad_id = tokenizer.pad_token_id
        ids, masks = [], []
        for tokens, _, length, _ in batch:
            padding = max_length - length
            ids.append(tokens + [pad_id] * padding)
            masks.append([1] * length + [0] * padding)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            [index for _, index, _, _ in batch],
            [length for _, _, length, _ in batch],
            [metadata for _, _, _, metadata in batch],
        )

    return collate_batch


def distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Llama residual-post activations for LlamaScope."
    )
    parser.add_argument(
        "--model", default="meta-llama/Meta-Llama-3.1-8B"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("activation_outs"))
    parser.add_argument(
        "--layers",
        default="22",
        help=(
            "Comma-separated Hugging Face hidden-state indices to capture "
            "(1..num_hidden_layers; index 0 is the embedding output)."
        ),
    )
    parser.add_argument(
        "--dataset-source", choices=("qa", "openwebtext"), default="qa"
    )
    parser.add_argument("--qa-datasets", default="arc_easy,mmlu,hle")
    parser.add_argument("--qa-num-samples")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--max-token-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--add-bos-token", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include-choices", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prompt-suffix",
        help=(
            "Optional common final prompt line, for example 'Answer:'. The "
            "suffix is preserved when long prompt bodies are truncated."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [rank={rank}] [%(levelname)s] %(message)s",
    )
    layers = parse_int_list(args.layers)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset_source == "qa":
        dataset = build_combined_question_dataset(
            dataset_names=[
                name.strip() for name in args.qa_datasets.split(",") if name.strip()
            ],
            tokenizer=tokenizer,
            max_token_length=args.max_token_length,
            add_bos_token=args.add_bos_token,
            include_choices=args.include_choices,
            num_samples=parse_limits(args.qa_num_samples),
            shuffle=args.shuffle,
            seed=args.seed,
            prompt_suffix=args.prompt_suffix,
        )
    else:
        dataset = OpenWebTextSentencesDataset(
            tokenizer,
            args.max_token_length,
            add_bos_token=args.add_bos_token,
            num_samples=args.num_samples,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    # Non-overlapping deterministic shards avoid DistributedSampler padding and
    # preserve each dataset's original global index in filenames.
    shard = Subset(dataset, range(rank, len(dataset), world_size))
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate(tokenizer),
    )

    logging.info("Loading %s on %s", args.model, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    writer = ActivationWriter(args.output_dir, rank)
    writer.start()

    with ResidualPostCapture(model, layers) as capture:
        for batch_number, (ids, mask, indices, lengths, metadata) in enumerate(loader):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            capture.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            activations = {
                layer: [
                    states[row, :length].to(dtype=dtype).cpu()
                    for row, length in enumerate(lengths)
                ]
                for layer, states in capture.values.items()
            }
            writer.submit(CaptureBatch(activations, indices, metadata))
            if (batch_number + 1) % 10 == 0:
                logging.info("Captured %d/%d batches", batch_number + 1, len(loader))
    writer.close()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_metadata(args.output_dir, world_size)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Capture complete")


if __name__ == "__main__":
    main()
