from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CaptureBatch:
    activations: dict[int, list[torch.Tensor]]
    indices: list[int]
    metadata: list[dict]


class ResidualPostCapture:
    """Capture outputs of Hugging Face Llama decoder blocks."""

    def __init__(self, model, layers: list[int]) -> None:
        decoder_layers = model.model.layers
        invalid = [layer for layer in layers if not 0 <= layer < len(decoder_layers)]
        if invalid:
            raise ValueError(
                f"Invalid layers {invalid}; model has {len(decoder_layers)} blocks."
            )
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            decoder_layers[layer].register_forward_hook(self._hook(layer))
            for layer in layers
        ]

    def _hook(self, layer: int):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.values[layer] = hidden.detach()

        return save

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ActivationWriter:
    def __init__(self, output_dir: Path, rank: int, max_queue: int = 8) -> None:
        self.output_dir = output_dir
        self.part_path = output_dir / f"metadata.part_rank{rank}.jsonl"
        self.queue: queue.Queue[CaptureBatch | None] = queue.Queue(maxsize=max_queue)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, batch: CaptureBatch) -> None:
        if self.error is not None:
            raise RuntimeError("Activation writer failed") from self.error
        self.queue.put(batch)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Activation writer failed") from self.error

    def _run(self) -> None:
        try:
            with self.part_path.open("w", encoding="utf-8") as metadata_file:
                while True:
                    batch = self.queue.get()
                    if batch is None:
                        return
                    for layer, samples in batch.activations.items():
                        layer_dir = self.output_dir / f"layer_{layer}"
                        layer_dir.mkdir(parents=True, exist_ok=True)
                        for sample, index, metadata in zip(
                            samples, batch.indices, batch.metadata, strict=True
                        ):
                            path = layer_dir / f"activations_l{layer}_idx{index}.pt"
                            torch.save(sample, path)
                            record = {
                                **metadata,
                                "activation_path": str(path),
                                "capture_dataset_idx": index,
                                "capture_layer": layer,
                            }
                            metadata_file.write(json.dumps(record) + "\n")
        except BaseException as exc:
            self.error = exc


def merge_metadata(output_dir: Path, world_size: int) -> Path:
    records = []
    for rank in range(world_size):
        part = output_dir / f"metadata.part_rank{rank}.jsonl"
        if not part.exists():
            continue
        with part.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(
        key=lambda item: (item["capture_dataset_idx"], item["capture_layer"])
    )
    destination = output_dir / "metadata_rank0.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    logging.info("Merged %d metadata records into %s", len(records), destination)
    return destination

