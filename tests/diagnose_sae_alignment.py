#!/usr/bin/env python3

import argparse

import torch
import torch.nn.functional as F

from src.utils.model import setup


def _load_hidden(path: str) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "hidden_states" not in obj:
        raise TypeError(f"{path} must be a dict payload containing hidden_states.")
    return obj["hidden_states"].detach().float().cpu(), obj


def _metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    flat_x = x.flatten()
    flat_y = y.flatten()
    diff = flat_y - flat_x
    cosine = F.cosine_similarity(flat_x, flat_y, dim=0).item()
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    relative_l2 = (
        torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(flat_x)
    ).item()
    return cosine, rmse, relative_l2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--reconstructed", required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--precision", type=int, default=16, choices=[4, 8, 16, 32])
    parser.add_argument("--center-layer", type=int, default=19)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    recon_hidden, _recon_payload = _load_hidden(args.reconstructed)
    _orig_hidden, orig_payload = _load_hidden(args.original)

    input_ids = orig_payload.get("input_ids")
    if input_ids is None:
        raise KeyError("Original payload must contain input_ids.")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    model, _tokenizer, _device, _layer_idx = setup(
        model_id=args.model_id,
        precision=args.precision,
        layer_idx=args.center_layer,
        print_stats=True,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(model.device),
            output_hidden_states=True,
            use_cache=False,
        )

    total_hidden = len(outputs.hidden_states)
    start = max(0, args.center_layer - args.window)
    end = min(total_hidden - 1, args.center_layer + args.window)

    print(f"reconstructed shape: {tuple(recon_hidden.shape)}")
    print(f"HF hidden-state indices: 0..{total_hidden - 1}")
    print()
    print("idx  all_cos     all_rmse    all_rel_l2   skip0_cos   skip0_rmse  skip0_rel_l2")

    for idx in range(start, end + 1):
        candidate = outputs.hidden_states[idx].detach().float().cpu()
        if candidate.dim() == 3 and candidate.size(0) == 1:
            candidate = candidate.squeeze(0)
        if candidate.shape != recon_hidden.shape:
            print(f"{idx:>3}  shape mismatch {tuple(candidate.shape)}")
            continue

        all_cos, all_rmse, all_rel_l2 = _metrics(candidate, recon_hidden)
        if candidate.size(0) > 1:
            skip_cos, skip_rmse, skip_rel_l2 = _metrics(candidate[1:], recon_hidden[1:])
        else:
            skip_cos, skip_rmse, skip_rel_l2 = float("nan"), float("nan"), float("nan")

        print(
            f"{idx:>3}  "
            f"{all_cos:>10.6f} {all_rmse:>11.6g} {all_rel_l2:>12.6g} "
            f"{skip_cos:>11.6f} {skip_rmse:>11.6g} {skip_rel_l2:>13.6g}"
        )


if __name__ == "__main__":
    main()
