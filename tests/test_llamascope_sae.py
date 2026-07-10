#!/usr/bin/env python3

import argparse

import torch
from sae_lens import SAE


DEFAULT_RELEASE = "llama_scope_lxr_32x"
DEFAULT_LAYER = 19
DEFAULT_WIDTH = "32x"


def sae_id_for_layer(layer: int, width: str) -> str:
    if layer < 0 or layer > 31:
        raise ValueError("Llama Scope Llama-3.1-8B residual SAEs have layers 0-31.")
    return f"l{layer}r_{width}"


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_sae(
    device: str,
    release: str = DEFAULT_RELEASE,
    layer: int = DEFAULT_LAYER,
    width: str = DEFAULT_WIDTH,
    dtype: str = "float32",
):
    sae_id = sae_id_for_layer(layer, width)

    if hasattr(SAE, "from_pretrained_with_cfg_and_sparsity"):
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release,
            sae_id=sae_id,
            device=device,
        )
    else:
        loaded = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        if isinstance(loaded, tuple):
            sae, cfg_dict, sparsity = loaded
        else:
            sae, cfg_dict, sparsity = loaded, None, None

    requested_dtype = resolve_dtype(dtype)
    if requested_dtype is not None:
        sae = sae.to(requested_dtype)

    sae.eval()

    return sae, cfg_dict, sparsity


def _tensor_for_sae(tensor: torch.Tensor, sae: SAE) -> torch.Tensor:
    return tensor.to(device=sae.device, dtype=sae.dtype)


def _cfg_value(cfg, cfg_dict, *names):
    for source in (cfg, cfg_dict):
        if source is None:
            continue
        for name in names:
            if isinstance(source, dict) and name in source:
                return source[name]
            if hasattr(source, name):
                return getattr(source, name)
    return "unknown"


def encode_bundle(bundle, sae):
    if "hidden_states" not in bundle:
        raise KeyError("Input bundle does not contain 'hidden_states'.")

    hidden = _tensor_for_sae(bundle["hidden_states"], sae)

    with torch.no_grad():
        features = sae.encode(hidden)

    output = dict(bundle)
    output["sae_features"] = features.cpu()

    return output


def decode_bundle(bundle, sae):
    if "sae_features" not in bundle:
        raise KeyError("Input bundle does not contain 'sae_features'.")

    features = _tensor_for_sae(bundle["sae_features"], sae)

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

    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="SAELens release name.",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Llama layer to load, 0-31.",
    )

    parser.add_argument(
        "--width",
        default=DEFAULT_WIDTH,
        choices=("32x", "8x"),
        help="Llama Scope expansion width.",
    )

    dtype_group = parser.add_mutually_exclusive_group()

    dtype_group.add_argument(
        "--float32",
        action="store_const",
        dest="dtype",
        const="float32",
        default="float32",
        help="Convert the SAE to float32 at runtime. This is the default.",
    )

    dtype_group.add_argument(
        "--bfloat16",
        action="store_const",
        dest="dtype",
        const="bfloat16",
        help="Run the SAE in bfloat16.",
    )

    dtype_group.add_argument(
        "--native-dtype",
        action="store_const",
        dest="dtype",
        const="auto",
        help="Keep the dtype chosen by the SAELens loader.",
    )

    dtype_group.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        help="Runtime SAE dtype. Kept for compatibility; default is float32.",
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

    sae, cfg_dict, sparsity = load_sae(
        device=args.device,
        release=args.release,
        layer=args.layer,
        width=args.width,
        dtype=args.dtype,
    )

    print(
        "Loaded SAE "
        f"release={args.release} sae_id={sae_id_for_layer(args.layer, args.width)} "
        f"hook={_cfg_value(sae.cfg, cfg_dict, 'hook_name', 'hook_point', 'hook_point_in')} "
        f"d_in={_cfg_value(sae.cfg, cfg_dict, 'd_in', 'd_model', 'input_dim')} "
        f"d_sae={_cfg_value(sae.cfg, cfg_dict, 'd_sae', 'num_latents', 'dict_size')} "
        f"dtype={sae.dtype} sparsity_present={sparsity is not None}"
    )

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
