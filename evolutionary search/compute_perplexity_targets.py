#!/usr/bin/env python3
"""Compute the JSONL targets consumed by ``build_artifacts.py regression``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute causal-LM perplexity targets from activation metadata."
    )
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        args.metadata_file.open("r", encoding="utf-8") as source,
        args.output_jsonl.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = record.get("prompt_text")
            if not prompt:
                continue
            capture_layer = record.get("capture_layer")
            if args.layer is not None and capture_layer is not None:
                if int(capture_layer) != args.layer:
                    continue

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            if input_ids.shape[1] < 2:
                continue

            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits[:, :-1].float()
                labels = input_ids[:, 1:]
                token_nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                )
                sum_nll = token_nll.sum().item()
                mean_nll = token_nll.mean().item()

            output = {
                "capture_dataset_idx": int(
                    record.get("capture_dataset_idx", line_number)
                ),
                "capture_layer": (
                    int(capture_layer) if capture_layer is not None else args.layer
                ),
                "source_dataset": record.get("source_dataset"),
                "token_count": int(input_ids.shape[1]),
                "sum_nll": sum_nll,
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
            }
            destination.write(json.dumps(output) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"Computed {written} targets")
            if args.max_records is not None and written >= args.max_records:
                break
    print(f"Saved {written} targets to {args.output_jsonl}")


if __name__ == "__main__":
    main()
