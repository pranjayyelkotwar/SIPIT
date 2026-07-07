#!/usr/bin/env python3

import argparse

import torch
from huggingface_hub import hf_hub_download


D_MODEL = 4096
EXPANSION_FACTOR = 16
SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_FILE = "Llama-3.1-8B-Instruct-SAE-l19.pth"


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, d_in: int, expansion_factor: int):
        super().__init__()

        d_hidden = d_in * expansion_factor

        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return torch.relu(self.encoder_linear(x))

    def decode(self, z):
        return self.decoder_linear(z)


def load_sae(device: str):
    path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=SAE_FILE,
    )

    sae = SparseAutoEncoder(
        d_in=D_MODEL,
        expansion_factor=EXPANSION_FACTOR,
    ).to(device)

    state = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    sae.load_state_dict(state)
    sae.eval()

    return sae


def encode_bundle(bundle, sae):
    if "hidden_states" not in bundle:
        raise KeyError("Input bundle does not contain 'hidden_states'.")

    hidden = bundle["hidden_states"].float()

    with torch.no_grad():
        features = sae.encode(hidden)

    output = dict(bundle)
    output["sae_features"] = features.cpu()

    return output


def decode_bundle(bundle, sae):
    if "sae_features" not in bundle:
        raise KeyError("Input bundle does not contain 'sae_features'.")

    features = bundle["sae_features"].float()

    with torch.no_grad():
        reconstructed = sae.decode(features)

    output = dict(bundle)

    # Replace ONLY the hidden states.
    output["hidden_states"] = reconstructed.cpu()

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input bundle (.pt)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output bundle (.pt)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--encode",
        action="store_true",
        help="Encode hidden_states into sae_features.",
    )

    mode.add_argument(
        "--decode",
        action="store_true",
        help="Decode sae_features back into hidden_states.",
    )

    args = parser.parse_args()

    print(f"Loading bundle: {args.input}")
    bundle = torch.load(args.input, map_location="cpu")

    sae = load_sae(args.device)

    if args.encode:
        print("Encoding hidden states...")
        output = encode_bundle(bundle, sae)

        print(
            f"{tuple(bundle['hidden_states'].shape)} -> "
            f"{tuple(output['sae_features'].shape)}"
        )

    else:
        print("Decoding SAE features...")
        output = decode_bundle(bundle, sae)

        print(
            f"{tuple(bundle['sae_features'].shape)} -> "
            f"{tuple(output['hidden_states'].shape)}"
        )

    torch.save(output, args.output)
    print(f"Saved bundle to {args.output}")


if __name__ == "__main__":
    main()