#!/usr/bin/env python3
"""Compare clean generation with a persistent SAE residual-stream intervention."""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_RELEASE = "llama_scope_lxr_32x"
DEFAULT_LAYER = 22
DEFAULT_WIDTH = "32x"
DEFAULT_PROMPTS = Path(__file__).with_name("sae_reconstruction_prompts.json")


def sae_id_for_layer(layer: int, width: str) -> str:
    return f"l{layer}r_{width}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean and SAE-round-tripped generation on curated prompts."
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="JSON prompt list (default: bundled five-prompt sample).",
    )
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sae-release", default=DEFAULT_RELEASE)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--sae-width", choices=("8x", "32x"), default=DEFAULT_WIDTH)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map", default="auto", help="Transformers device_map value."
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-format", choices=("auto", "plain", "chat"), default="auto"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("sae_reconstruction_results")
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def normalise_prompt_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "source_id": str(record.get("id") or f"prompt-{index + 1}"),
        "prompt": str(record["prompt"]),
        "category": record.get("category") or "general",
    }


def load_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1.")
    raw = json.loads(args.prompts_file.read_text(encoding="utf-8"))
    raw_records = raw["prompts"] if isinstance(raw, dict) else raw
    records = [normalise_prompt_record(record, i) for i, record in enumerate(raw_records)]
    if args.shuffle:
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(records), generator=generator).tolist()
        records = [records[index] for index in order]
    records = records[: args.num_prompts]
    if len(records) < args.num_prompts:
        raise ValueError(
            f"Requested {args.num_prompts} prompts, found {len(records)}."
        )
    return records


def encode_prompt(tokenizer, prompt: dict[str, Any], prompt_format: str) -> torch.Tensor:
    content = prompt["prompt"]
    use_chat = prompt_format == "chat" or (
        prompt_format == "auto" and bool(getattr(tokenizer, "chat_template", None))
    )
    if use_chat:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        ids = tokenizer(
            f"{content}\n\nResponse:", add_special_tokens=True, return_tensors="pt"
        )["input_ids"]
    return ids.long()


def _cfg_value(sae, cfg_dict: Any, *names: str) -> Any:
    for source in (getattr(sae, "cfg", None), cfg_dict):
        for name in names:
            if isinstance(source, dict) and name in source:
                return source[name]
            if source is not None and hasattr(source, name):
                return getattr(source, name)
    return None


def load_sae(release: str, sae_id: str, device: torch.device, dtype: torch.dtype):
    if hasattr(SAE, "from_pretrained_with_cfg_and_sparsity"):
        sae, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release, sae_id=sae_id, device=str(device)
        )
    else:
        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
        if isinstance(loaded, tuple):
            sae, cfg_dict = loaded[0], loaded[1]
        else:
            sae, cfg_dict = loaded, None
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    sae.requires_grad_(False)
    return sae, cfg_dict


def residual_block(model, hf_hidden_state_index: int):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("This experiment currently supports Llama-like HF causal LMs.")
    layers = model.model.layers
    if not 1 <= hf_hidden_state_index <= len(layers):
        raise ValueError(
            f"--layer must be a Hugging Face hidden-state index in [1, {len(layers)}]."
        )
    return layers[hf_hidden_state_index - 1]


def _metric_block(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    x, y = original.float(), reconstructed.float()
    difference = y - x
    x_flat, y_flat = x.reshape(-1), y.reshape(-1)
    squared_error = torch.sum(difference.square())
    centered_energy = torch.sum((x - x.mean()).square())
    norm = torch.linalg.vector_norm(x_flat)
    return {
        "cosine": float(F.cosine_similarity(x_flat, y_flat, dim=0).item()),
        "mean_token_cosine": float(F.cosine_similarity(x, y, dim=-1).mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(difference.reshape(-1)) / norm.clamp_min(1e-12)).item()
        ),
        "explained_variance": float(
            (1.0 - squared_error / centered_energy.clamp_min(1e-12)).item()
        ),
    }


def reconstruction_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor, features: torch.Tensor
) -> dict[str, float]:
    metrics = {
        f"all_{key}": value
        for key, value in _metric_block(original, reconstructed).items()
    }
    if original.shape[-2] > 1:
        non_bos = _metric_block(original[..., 1:, :], reconstructed[..., 1:, :])
        metrics.update({f"non_bos_{key}": value for key, value in non_bos.items()})
    active = (features > 0).sum(dim=-1).float()
    metrics["mean_l0"] = float(active.mean().item())
    metrics["latent_density"] = float((active.mean() / features.shape[-1]).item())
    return metrics


class SAERoundTripHook(AbstractContextManager):
    """Replace residual-post with SAE.decode(SAE.encode(residual-post))."""

    def __init__(self, block, sae, *, record_first_call: bool = False) -> None:
        self.block = block
        self.sae = sae
        self.record_first_call = record_first_call
        self.first_call_metrics: dict[str, float] | None = None
        self.calls = 0
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sae_input = hidden.to(device=self.sae.device, dtype=self.sae.dtype)
        features = self.sae.encode(sae_input)
        reconstructed = self.sae.decode(features).to(
            device=hidden.device, dtype=hidden.dtype
        )
        if self.record_first_call and self.first_call_metrics is None:
            self.first_call_metrics = reconstruction_metrics(
                hidden.detach(), reconstructed.detach(), features.detach()
            )
        self.calls += 1
        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        return reconstructed

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args):
        if self.handle is not None:
            self.handle.remove()
        return False


def generate(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )
    return sequences[:, input_ids.shape[-1] :]


def output_similarity(clean: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, Any]:
    clean_ids = clean.flatten().tolist()
    reconstructed_ids = reconstructed.flatten().tolist()
    common_prefix = 0
    for clean_id, reconstructed_id in zip(clean_ids, reconstructed_ids):
        if clean_id != reconstructed_id:
            break
        common_prefix += 1
    max_length = max(len(clean_ids), len(reconstructed_ids), 1)
    positional_matches = sum(
        clean_id == reconstructed_id
        for clean_id, reconstructed_id in zip(clean_ids, reconstructed_ids)
    )
    return {
        "exact_token_match": clean_ids == reconstructed_ids,
        "common_prefix_tokens": common_prefix,
        "position_match_rate": positional_matches / max_length,
        "clean_token_count": len(clean_ids),
        "sae_token_count": len(reconstructed_ids),
    }


def compare_teacher_forced_logits(
    clean_logits: torch.Tensor,
    reconstructed_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | bool]:
    clean, reconstructed = clean_logits.float(), reconstructed_logits.float()
    clean_log_probs = F.log_softmax(clean, dim=-1)
    reconstructed_log_probs = F.log_softmax(reconstructed, dim=-1)
    kl_per_token = torch.sum(
        clean_log_probs.exp() * (clean_log_probs - reconstructed_log_probs), dim=-1
    )
    clean_top1 = clean.argmax(dim=-1)
    reconstructed_top1 = reconstructed.argmax(dim=-1)
    flat_targets = targets.reshape(-1)
    return {
        "mean_kl_clean_to_sae": float(kl_per_token.mean().item()),
        "first_token_kl_clean_to_sae": float(kl_per_token.reshape(-1)[0].item()),
        "top1_agreement_rate": float((clean_top1 == reconstructed_top1).float().mean().item()),
        "first_token_top1_agrees": bool(
            clean_top1.reshape(-1)[0].item() == reconstructed_top1.reshape(-1)[0].item()
        ),
        "clean_nll_on_clean_continuation": float(
            F.cross_entropy(clean.reshape(-1, clean.shape[-1]), flat_targets).item()
        ),
        "sae_nll_on_clean_continuation": float(
            F.cross_entropy(
                reconstructed.reshape(-1, reconstructed.shape[-1]), flat_targets
            ).item()
        ),
    }


def teacher_forced_comparison(model, block, sae, prompt_ids, clean_tokens):
    full = torch.cat([prompt_ids, clean_tokens], dim=-1)
    model_input = full[:, :-1]
    start = prompt_ids.shape[-1] - 1
    stop = start + clean_tokens.shape[-1]
    with torch.inference_mode():
        clean_logits = model(
            input_ids=model_input,
            attention_mask=torch.ones_like(model_input),
            use_cache=False,
        ).logits[:, start:stop, :]
        with SAERoundTripHook(block, sae, record_first_call=True) as hook:
            reconstructed_logits = model(
                input_ids=model_input,
                attention_mask=torch.ones_like(model_input),
                use_cache=False,
            ).logits[:, start:stop, :]
        comparison = compare_teacher_forced_logits(
            clean_logits, reconstructed_logits, clean_tokens
        )
    comparison["sae_nll_increase"] = (
        comparison["sae_nll_on_clean_continuation"]
        - comparison["clean_nll_on_clean_continuation"]
    )
    return comparison, hook.first_call_metrics


def mean_key(results: list[dict[str, Any]], section: str, key: str) -> float:
    values = [item[section][key] for item in results]
    return float(sum(values) / len(values))


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_prompts": len(results),
        "generation_exact_match_rate": sum(
            item["generation_similarity"]["exact_token_match"] for item in results
        ) / len(results),
        "mean_generation_position_match_rate": mean_key(
            results, "generation_similarity", "position_match_rate"
        ),
        "mean_common_prefix_tokens": mean_key(
            results, "generation_similarity", "common_prefix_tokens"
        ),
        "first_token_agreement_rate": sum(
            item["teacher_forced"]["first_token_top1_agrees"] for item in results
        ) / len(results),
        "teacher_forced_top1_agreement_rate": mean_key(
            results, "teacher_forced", "top1_agreement_rate"
        ),
        "mean_teacher_forced_kl": mean_key(
            results, "teacher_forced", "mean_kl_clean_to_sae"
        ),
        "mean_sae_nll_increase": mean_key(
            results, "teacher_forced", "sae_nll_increase"
        ),
        "mean_prompt_non_bos_cosine": mean_key(
            results, "prompt_reconstruction", "non_bos_cosine"
        ),
        "mean_prompt_non_bos_relative_l2": mean_key(
            results, "prompt_reconstruction", "non_bos_relative_l2"
        ),
        "mean_prompt_non_bos_explained_variance": mean_key(
            results, "prompt_reconstruction", "non_bos_explained_variance"
        ),
        "mean_prompt_l0": mean_key(results, "prompt_reconstruction", "mean_l0"),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = []
    for item in payload["results"]:
        rows.append(
            {
                "source_id": item["source_id"],
                "category": item["category"],
                "clean_output": item["clean_output"],
                "sae_output": item["sae_output"],
                **item["generation_similarity"],
                **{f"prompt_{k}": v for k, v in item["prompt_reconstruction"].items()},
                **{f"tf_{k}": v for k, v in item["teacher_forced"].items()},
            }
        )
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = payload["summary"]
    lines = [
        "# SAE reconstruction experiment",
        "",
        f"- Prompts: {summary['num_prompts']}",
        f"- Exact generated-output match: {summary['generation_exact_match_rate']:.1%}",
        f"- First-token agreement: {summary['first_token_agreement_rate']:.1%}",
        f"- Teacher-forced top-1 agreement: {summary['teacher_forced_top1_agreement_rate']:.1%}",
        f"- Mean teacher-forced KL(clean || SAE): {summary['mean_teacher_forced_kl']:.6f}",
        f"- Mean clean-continuation NLL increase: {summary['mean_sae_nll_increase']:.6f}",
        f"- Mean non-BOS activation cosine: {summary['mean_prompt_non_bos_cosine']:.6f}",
        "- Mean non-BOS activation relative L2 error: "
        f"{summary['mean_prompt_non_bos_relative_l2']:.6f}",
        "- Mean non-BOS explained variance: "
        f"{summary['mean_prompt_non_bos_explained_variance']:.6f}",
        f"- Mean active SAE latents/token (L0): {summary['mean_prompt_l0']:.2f}",
        "",
        "## Outputs",
        "",
    ]
    for index, item in enumerate(payload["results"], 1):
        lines.extend(
            [
                f"### {index}. {item['source_id']}",
                "",
                f"**Prompt:** {item['prompt']}",
                "",
                f"**Clean:** {item['clean_output']}",
                "",
                f"**SAE:** {item['sae_output']}",
                "",
                "**Fidelity:** "
                f"exact={item['generation_similarity']['exact_token_match']}, "
                f"prefix={item['generation_similarity']['common_prefix_tokens']} tokens, "
                f"non-BOS cosine={item['prompt_reconstruction']['non_bos_cosine']:.4f}, "
                f"teacher-forced KL={item['teacher_forced']['mean_kl_clean_to_sae']:.4f}",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    torch.manual_seed(args.seed)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    prompts = load_prompts(args)

    print(f"Loading model {args.model} ({args.dtype}, device_map={args.device_map})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    block = residual_block(model, args.layer)
    block_parameter = next(block.parameters())
    sae_id = sae_id_for_layer(args.layer, args.sae_width)
    print(f"Loading SAE {args.sae_release}/{sae_id} on {block_parameter.device}")
    sae, sae_cfg = load_sae(
        args.sae_release, sae_id, block_parameter.device, block_parameter.dtype
    )
    d_in = _cfg_value(sae, sae_cfg, "d_in", "d_model", "input_dim")
    hidden_size = int(model.config.hidden_size)
    if d_in is not None and int(d_in) != hidden_size:
        raise ValueError(f"SAE d_in={d_in} does not match model hidden size {hidden_size}.")

    input_device = model.get_input_embeddings().weight.device
    results = []
    started = time.time()
    for index, prompt in enumerate(prompts, 1):
        print(f"[{index}/{len(prompts)}] {prompt['source_id']}")
        prompt_ids = encode_prompt(tokenizer, prompt, args.prompt_format).to(input_device)
        with torch.inference_mode():
            clean_tokens = generate(model, prompt_ids, args.max_new_tokens)
            with SAERoundTripHook(block, sae, record_first_call=True) as generation_hook:
                sae_tokens = generate(model, prompt_ids, args.max_new_tokens)
        teacher_forced, teacher_reconstruction = teacher_forced_comparison(
            model, block, sae, prompt_ids, clean_tokens
        )
        prompt_reconstruction = generation_hook.first_call_metrics or teacher_reconstruction
        if prompt_reconstruction is None:
            raise RuntimeError("SAE hook did not observe a model forward pass.")
        results.append(
            {
                **prompt,
                "prompt_token_count": int(prompt_ids.shape[-1]),
                "formatted_prompt": tokenizer.decode(
                    prompt_ids[0], skip_special_tokens=False
                ),
                "clean_output": tokenizer.decode(clean_tokens[0], skip_special_tokens=True),
                "sae_output": tokenizer.decode(sae_tokens[0], skip_special_tokens=True),
                "generation_similarity": output_similarity(clean_tokens, sae_tokens),
                "prompt_reconstruction": prompt_reconstruction,
                "teacher_forced": teacher_forced,
                "sae_generation_hook_calls": generation_hook.calls,
            }
        )

    payload = {
        "config": {
            "model": args.model,
            "sae_release": args.sae_release,
            "sae_id": sae_id,
            "hf_hidden_state_index": args.layer,
            "decoder_block_index": args.layer - 1,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "prompt_format": args.prompt_format,
            "prompt_source": str(args.prompts_file),
            "seed": args.seed,
            "elapsed_seconds": time.time() - started,
            "sae_hook_name": _cfg_value(
                sae, sae_cfg, "hook_name", "hook_point", "hook_point_in"
            ),
            "sae_training_model": _cfg_value(
                sae, sae_cfg, "model_name", "model_name_or_path"
            ),
        },
        "summary": build_summary(results),
        "results": results,
    }
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
