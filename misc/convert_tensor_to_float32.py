import argparse
from pathlib import Path

import torch


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_float32{input_path.suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to a .pt file containing a torch.Tensor")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .pt path. Defaults to '<input_stem>_float32.pt'.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    output_path = Path(args.output) if args.output else default_output_path(input_path)

    tensor = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        print(f"Expected torch.Tensor, got {type(tensor)}")
        return

    converted = tensor.to(torch.float32)
    torch.save(converted, output_path)

    print(f"Saved float32 tensor: {output_path}")
    print(f"Shape: {tuple(converted.shape)}")
    print(f"Dtype: {converted.dtype}")


if __name__ == "__main__":
    main()
