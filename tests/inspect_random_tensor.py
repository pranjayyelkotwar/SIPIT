import torch
from pathlib import Path
from pprint import pprint

PT_FILE = "hidden-state-bundle/tensors/prompt.pt"


def inspect(obj, name="root", depth=0, max_depth=5):
    indent = "    " * depth

    if depth > max_depth:
        print(f"{indent}{name}: <max depth reached>")
        return

    if isinstance(obj, torch.Tensor):
        print(f"{indent}{name}:")
        print(f"{indent}  Type   : Tensor")
        print(f"{indent}  Shape  : {tuple(obj.shape)}")
        print(f"{indent}  Dtype  : {obj.dtype}")
        print(f"{indent}  Device : {obj.device}")
        print(f"{indent}  Min    : {obj.min().item() if obj.numel() else 'N/A'}")
        print(f"{indent}  Max    : {obj.max().item() if obj.numel() else 'N/A'}")
        print(f"{indent}  Mean   : {obj.float().mean().item() if obj.numel() else 'N/A'}")
        return

    elif isinstance(obj, dict):
        print(f"{indent}{name}: dict ({len(obj)} keys)")
        for k, v in obj.items():
            inspect(v, f"[{repr(k)}]", depth + 1, max_depth)

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__} (len={len(obj)})")
        for i, v in enumerate(obj[:10]):  # first 10 elements
            inspect(v, f"[{i}]", depth + 1, max_depth)
        if len(obj) > 10:
            print(f"{indent}    ... ({len(obj)-10} more elements)")

    else:
        print(f"{indent}{name}: {type(obj).__name__}")
        try:
            print(f"{indent}  Value: {repr(obj)}")
        except Exception:
            pass


def main():
    path = Path(PT_FILE)

    print(f"Loading: {path.resolve()}")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("\nTop-level type:", type(obj))
    print("-" * 80)

    inspect(obj)


if __name__ == "__main__":
    main()