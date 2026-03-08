"""Pure PyTorch data loader for 1TB+ datasets. Zero JAX dependency.

Drop-in replacement for OpenPI's data_loader.py for PyTorch training.
Uses WebDataset shard streaming with multi-level shuffle.

Usage:
    from piutil.data.torch_loader import TorchScalableDataLoader

    loader = TorchScalableDataLoader(
        shard_pattern="/data/shards/{00000..01000}.tar",
        batch_size=32,
        num_workers=4,
    )
    for batch in loader:
        images = batch["observation"]["image"].to(device)
        actions = batch["actions"].to(device)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# tree_map: drop-in replacement for jax.tree.map on nested dicts/tensors
# ---------------------------------------------------------------------------


def tree_map(fn: Callable, tree: Any, *rest: Any) -> Any:
    """Apply fn to every leaf in a nested dict/list/tuple structure.

    Drop-in replacement for jax.tree.map — works on dicts, lists, tuples.
    Leaves are anything that is not a dict/list/tuple.

    Usage:
        # Move batch to device (replaces jax.tree.map)
        batch = tree_map(lambda x: x.to(device), batch)

        # Convert numpy to torch
        batch = tree_map(torch.as_tensor, batch)
    """
    if isinstance(tree, dict):
        keys = tree.keys()
        children = [tree[k] for k in keys]
        rest_children = [[r[k] for k in keys] for r in rest]
        return {k: tree_map(fn, c, *[rc[i] for rc in rest_children]) for i, (k, c) in enumerate(zip(keys, children))}
    elif isinstance(tree, (list, tuple)):
        rest_children = [list(r) for r in rest]
        mapped = [tree_map(fn, c, *[rc[i] for rc in rest_children]) for i, c in enumerate(tree)]
        return type(tree)(mapped)
    else:
        if rest:
            return fn(tree, *[r for r in rest])
        return fn(tree)


# ---------------------------------------------------------------------------
# TorchScalableDataLoader
# ---------------------------------------------------------------------------


class TorchScalableDataLoader:
    """WebDataset-based data loader for PyTorch. No JAX dependency.

    Features:
    - Shard-level parallelism via WebDataset
    - Multi-level shuffle (shard + buffer) for 1TB+ datasets
    - DDP-aware: automatic shard splitting across ranks
    - Returns torch tensors directly (no numpy->jax->torch conversion)
    - pin_memory for faster GPU transfer
    - Standard torch.utils.data.DataLoader under the hood
    """

    def __init__(
        self,
        shard_pattern: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        shuffle_buffer_size: int = 500_000,
        num_workers: int = 4,
        image_decode: str = "rgb8",
        decoder_fn: Callable | None = None,
        transform_fn: Callable[[dict], dict] | None = None,
        num_batches: int | None = None,
        pin_memory: bool = True,
        prefetch: int = 2,
        drop_last: bool = True,
        seed: int = 0,
        device: torch.device | str | None = None,
    ):
        """Create a scalable PyTorch data loader.

        Args:
            shard_pattern: WebDataset shard URL pattern.
                Examples: "/data/shards/{00000..01000}.tar"
                          "pipe:gsutil cat gs://bucket/shard-{0000..0999}.tar"
            batch_size: Global batch size. Auto-divided by DDP world size.
            shuffle: Whether to shuffle data.
            shuffle_buffer_size: Sample-level shuffle buffer size.
                500K gives good randomness for 1TB+ datasets.
            num_workers: Parallel I/O workers for DataLoader.
            image_decode: Image decode mode. "rgb8" for uint8 numpy, "pil" for PIL.
            decoder_fn: Custom decoder. Overrides image_decode if provided.
            transform_fn: Per-sample transform applied after decode.
            num_batches: Max batches to yield. None = infinite.
            pin_memory: Pin memory for faster CPU->GPU transfer.
            prefetch: Prefetch factor for DataLoader.
            drop_last: Drop incomplete final batch.
            seed: Random seed for shuffling.
            device: If set, tensors are moved to this device automatically.
        """
        import webdataset as wds

        self._num_batches = num_batches
        self._device = torch.device(device) if isinstance(device, str) else device

        # Compute local batch size for DDP
        local_batch_size = self._compute_local_batch_size(batch_size)

        # Build WebDataset pipeline
        pipeline = wds.WebDataset(
            shard_pattern,
            shardshuffle=1000 if shuffle else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            seed=seed,
        )

        if shuffle:
            pipeline = pipeline.shuffle(shuffle_buffer_size, initial=shuffle_buffer_size // 4)

        # Decode
        if decoder_fn is not None:
            pipeline = pipeline.map(decoder_fn)
        else:
            pipeline = pipeline.decode(image_decode)

        # Restructure flat WebDataset keys -> nested OpenPI dict
        pipeline = pipeline.map(_restructure_sample)

        # Per-sample transform
        if transform_fn is not None:
            pipeline = pipeline.map(transform_fn)

        # Batch + collate to torch tensors
        pipeline = pipeline.batched(
            local_batch_size,
            collation_fn=_torch_collate,
            partial=not drop_last,
        )

        # Wrap in PyTorch DataLoader for parallel workers
        if num_workers > 0:
            self._loader = wds.WebLoader(
                pipeline,
                num_workers=num_workers,
                batch_size=None,
                prefetch_factor=prefetch,
                persistent_workers=True,
                pin_memory=pin_memory and torch.cuda.is_available(),
            )
        else:
            self._loader = pipeline

        logger.info(
            f"TorchScalableDataLoader: pattern={shard_pattern}, "
            f"batch_size={batch_size}, local={local_batch_size}, "
            f"shuffle={shuffle}, buffer={shuffle_buffer_size}, "
            f"workers={num_workers}, device={self._device}"
        )

    @staticmethod
    def _compute_local_batch_size(batch_size: int) -> int:
        if dist.is_initialized():
            return batch_size // dist.get_world_size()
        return batch_size

    def __iter__(self) -> Iterator[dict]:
        num_items = 0
        while True:
            for batch in self._loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                num_items += 1

                if self._device is not None:
                    batch = tree_map(
                        lambda x: x.to(self._device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
                        batch,
                    )

                yield batch

            # Loop forever if num_batches is None (same as OpenPI)
            if self._num_batches is None:
                continue
            else:
                return


# ---------------------------------------------------------------------------
# OpenPI integration
# ---------------------------------------------------------------------------


class _TorchDataLoaderImpl:
    """Wraps TorchScalableDataLoader to match OpenPI's DataLoader protocol.

    Yields (observation_dict, actions) tuples.
    observation_dict is a plain dict of torch tensors (no JAX Observation class).
    """

    def __init__(self, data_config: Any, loader: TorchScalableDataLoader):
        self._data_config = data_config
        self._loader = loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._loader:
            actions = batch.pop("actions")
            yield batch, actions


def create_torch_data_loader(
    config,
    *,
    shard_pattern: str | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    shuffle_buffer_size: int = 500_000,
    num_workers: int = 4,
    prefetch: int = 2,
    pin_memory: bool = True,
    device: torch.device | str | None = None,
):
    """Drop-in replacement for OpenPI's create_data_loader(framework="pytorch").

    No JAX dependency. Returns plain dict batches with torch tensors.

    Usage in train_pytorch.py:
        # Before:
        from openpi.training.data_loader import create_data_loader
        loader = create_data_loader(config, framework="pytorch", shuffle=True)
        for observation, actions in loader:
            observation = jax.tree.map(lambda x: x.to(device), observation)

        # After:
        from piutil.data.torch_loader import create_torch_data_loader, tree_map
        loader = create_torch_data_loader(config, shard_pattern=pattern, device=device)
        for observation, actions in loader:
            # Already on device if device was specified, otherwise:
            observation = tree_map(lambda x: x.to(device), observation)

    Args:
        config: OpenPI TrainConfig.
        shard_pattern: WebDataset shard URL pattern.
        shuffle: Whether to shuffle.
        num_batches: Max batches.
        skip_norm_stats: Skip normalization.
        shuffle_buffer_size: Shuffle buffer size.
        num_workers: Parallel workers.
        prefetch: Prefetch count.
        pin_memory: Pin memory for GPU transfer.
        device: Auto-move tensors to this device.
    """
    import openpi.transforms as _transforms

    data_config = config.data.create(config.assets_dirs, config.model)

    # Resolve shard pattern
    if shard_pattern is None:
        shard_pattern = getattr(data_config, "shard_pattern", None) or getattr(config, "shard_pattern", None)
    if shard_pattern is None:
        raise ValueError(
            "shard_pattern is required. Convert data first with "
            "piutil.data.convert_rlds_to_shards() or convert_lerobot_to_shards()."
        )

    # Build transform pipeline (same as OpenPI, but outputs torch tensors)
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Run `scripts/compute_norm_stats.py --config-name=<your-config>` first."
            )
        norm_stats = data_config.norm_stats

    transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    loader = TorchScalableDataLoader(
        shard_pattern=shard_pattern,
        batch_size=config.batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        num_batches=num_batches,
        transform_fn=transforms,
        prefetch=prefetch,
        pin_memory=pin_memory,
        device=device,
    )

    return _TorchDataLoaderImpl(data_config, loader)


# ---------------------------------------------------------------------------
# Collation & restructuring (torch-native, no JAX)
# ---------------------------------------------------------------------------


_TOP_LEVEL_FIELDS = {"actions", "prompt"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_ARRAY_EXTS = {".npy", ".pyd"}


def _restructure_sample(sample: dict) -> dict:
    """Convert flat WebDataset keys to nested OpenPI-compatible dict.

    All fields that aren't "actions" or "prompt" are placed under "observation".
    Image fields (detected by extension) are stored as uint8 numpy arrays.
    Numeric fields are stored as float32 numpy arrays.
    """
    result = {}
    observation = {}

    for key, value in sample.items():
        if key.startswith("__"):
            continue

        # Strip file extensions to get the field name
        field = key
        is_image = False
        for ext in (*_IMAGE_EXTS, *_ARRAY_EXTS, ".txt", ".json"):
            if field.endswith(ext):
                is_image = ext in _IMAGE_EXTS
                field = field[: -len(ext)]
                break

        # Route: top-level or observation
        if field == "actions":
            result["actions"] = np.asarray(value, dtype=np.float32) if not isinstance(value, np.ndarray) else value
        elif field == "prompt":
            result["prompt"] = value if isinstance(value, str) else value.decode("utf-8") if isinstance(value, bytes) else str(value)
        else:
            # Everything else goes into observation
            observation[field] = np.asarray(value)

    if observation:
        result["observation"] = observation

    return result


def _torch_collate(samples: list[dict]) -> dict:
    """Collate list of sample dicts into a batched dict of torch tensors."""
    if not samples:
        return {}

    keys = samples[0].keys()
    result = {}
    for key in keys:
        values = [s[key] for s in samples]
        if isinstance(values[0], dict):
            result[key] = _torch_collate(values)
        elif isinstance(values[0], np.ndarray):
            stacked = np.stack(values, axis=0)
            if stacked.dtype.kind in ("f", "i", "u"):
                result[key] = torch.from_numpy(stacked)
            else:
                result[key] = stacked
        elif isinstance(values[0], str):
            result[key] = values  # keep strings as list
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return result
