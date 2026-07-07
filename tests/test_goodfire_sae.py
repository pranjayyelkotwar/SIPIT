#!/usr/bin/env python3

import argparse

import torch
from huggingface_hub import hf_hub_download


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, d_in: int, expansion_factor: int):
        super().__init__()

        d_hidden = d_in * expansion_factor

        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return torch.relu(self.encoder_linear(x))

    def decode(self, f):
        return self.decoder_linear(f)


def load_sae(device="cpu"):
    path = hf_hub_download(
        repo_id="Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
        filename="Llama-3.1-8B-Instruct-SAE-l19.pth",
    )

    sae = SparseAutoEncoder(
        d_in=4096,
        expansion_factor=16,
    ).to(device)

    state = torch.load(path, map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.eval()

    return sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input hidden-state bundle (.pt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output SAE feature bundle (.pt)",
    )

    args = parser.parse_args()

    print(f"Loading {args.input}")

    bundle = torch.load(args.input, map_location="cpu")

    hidden_states = bundle["hidden_states"].float()

    print("Hidden states:", tuple(hidden_states.shape))

    sae = load_sae()

    with torch.no_grad():
        features = sae.encode(hidden_states)

    print("SAE features:", tuple(features.shape))

    output = dict(bundle)
    output["sae_features"] = features.cpu()

    torch.save(output, args.output)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()