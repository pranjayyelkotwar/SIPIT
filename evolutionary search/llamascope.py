from __future__ import annotations

from dataclasses import dataclass

import torch
from sae_lens import SAE


DEFAULT_RELEASE = "llama_scope_lxr_32x"
DEFAULT_WIDTH = "32x"


def sae_id_for_layer(layer: int, width: str = DEFAULT_WIDTH) -> str:
    if not 0 <= layer <= 31:
        raise ValueError("Llama Scope Llama-3.1-8B residual SAEs use layers 0-31.")
    return f"l{layer}r_{width}"


def resolve_dtype(dtype: str) -> torch.dtype | None:
    values = {
        "auto": None,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported SAE dtype: {dtype}") from exc


@dataclass
class LoadedLlamaScopeSAE:
    sae: SAE
    config: object | None
    sparsity: object | None
    release: str
    sae_id: str


def load_llamascope_sae(
    *,
    layer: int,
    device: str = "cpu",
    release: str = DEFAULT_RELEASE,
    width: str = DEFAULT_WIDTH,
    dtype: str = "float32",
) -> LoadedLlamaScopeSAE:
    """Load LlamaScope through SAELens, matching tests/test_llamascope_sae.py."""
    sae_id = sae_id_for_layer(layer, width)
    if hasattr(SAE, "from_pretrained_with_cfg_and_sparsity"):
        sae, config, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release,
            sae_id=sae_id,
            device=device,
        )
    else:
        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        if isinstance(loaded, tuple):
            sae, config, sparsity = loaded
        else:
            sae, config, sparsity = loaded, None, None

    requested_dtype = resolve_dtype(dtype)
    if requested_dtype is not None:
        sae = sae.to(requested_dtype)
    sae.eval()
    return LoadedLlamaScopeSAE(sae, config, sparsity, release, sae_id)


def tensor_for_sae(tensor: torch.Tensor, sae: SAE) -> torch.Tensor:
    return tensor.to(device=sae.device, dtype=sae.dtype)

