"""Scalable data loader for 1TB+ datasets.

Drop-in replacement for OpenPI's TorchDataLoader / RLDSDataLoader.
Uses WebDataset for shard-based streaming instead of LeRobot random access
or TensorFlow RLDS pipelines.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_jax():
    try:
        import jax
        return jax
    except ImportError:
        return None


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class ScalableDataLoader:
    """WebDataset-based data loader optimized for 1TB+ datasets.

    Replaces both TorchDataLoader and RLDSDataLoader from OpenPI.
    Yields dict batches in the same format OpenPI expects.

    Key improvements:
    - Shard-level parallelism: reads multiple tar shards concurrently
    - Multi-level shuffle: shard shuffle + sample buffer shuffle
    - Constant memory: streams from disk, never loads full dataset
    - No TF dependency: pure PyTorch/numpy pipeline
    - Supports multi-node: automatic shard splitting across nodes
    """

    def __init__(
        self,
        shard_pattern: str,
        batch_size: int,
        *,
        # Shuffle
        shuffle: bool = True,
        shuffle_buffer_size: int = 500_000,
        # Parallelism
        num_workers: int = 4,
        # Decode
        image_decode: str = "rgb8",
        decoder_fn: Callable | None = None,
        # Output
        framework: Literal["jax", "pytorch"] = "jax",
        sharding: Any | None = None,
        # Iteration
        num_batches: int | None = None,
        # Transform
        transform_fn: Callable[[dict], dict] | None = None,
        # Advanced
        cache_dir: str | None = None,
        prefetch: int = 2,
        drop_last: bool = True,
        seed: int = 0,
    ):
        """Create a scalable data loader.

        Args:
            shard_pattern: WebDataset shard URL pattern.
                Examples: "/data/shards/{00000..01000}.tar"
                          "pipe:gsutil cat gs://bucket/shard-{0000..0999}.tar"
            batch_size: Global batch size. Will be divided by device count.
            shuffle: Whether to shuffle data.
            shuffle_buffer_size: Size of the sample-level shuffle buffer.
                Larger = better randomness but more memory. 500K is good for 1TB+.
            num_workers: Number of parallel I/O workers.
            image_decode: Image decode mode. "rgb8" for uint8 numpy, "pil" for PIL.
            decoder_fn: Custom decoder function. If provided, overrides image_decode.
            framework: Output framework - "jax" or "pytorch".
            sharding: JAX sharding spec. If None, uses default data-parallel sharding.
            num_batches: Max batches to yield. None = infinite iteration.
            transform_fn: Optional per-sample transform applied after decoding.
            cache_dir: If set, cache decoded samples to this directory.
            prefetch: Number of batches to prefetch.
            drop_last: Drop the last incomplete batch.
            seed: Random seed for shuffling.
        """
        import webdataset as wds

        self._shard_pattern = shard_pattern
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._num_workers = num_workers
        self._image_decode = image_decode
        self._decoder_fn = decoder_fn
        self._framework = framework
        self._num_batches = num_batches
        self._transform_fn = transform_fn
        self._cache_dir = cache_dir
        self._prefetch = prefetch
        self._drop_last = drop_last
        self._seed = seed

        # Compute local batch size
        self._local_batch_size = self._compute_local_batch_size(batch_size, framework)

        # Set up JAX sharding
        self._sharding = sharding
        if framework == "jax" and sharding is None:
            jax = _try_import_jax()
            if jax is not None:
                self._sharding = jax.sharding.NamedSharding(
                    jax.sharding.Mesh(jax.devices(), ("B",)),
                    jax.sharding.PartitionSpec("B"),
                )

        # Build the WebDataset pipeline
        self._dataset = self._build_pipeline(wds)

        logger.info(
            f"ScalableDataLoader: pattern={shard_pattern}, "
            f"batch_size={batch_size}, local_batch_size={self._local_batch_size}, "
            f"shuffle={shuffle}, buffer={shuffle_buffer_size}, workers={num_workers}"
        )

    @staticmethod
    def _compute_local_batch_size(batch_size: int, framework: str) -> int:
        if framework == "pytorch":
            torch = _try_import_torch()
            if torch is not None and torch.distributed.is_initialized():
                return batch_size // torch.distributed.get_world_size()
            return batch_size
        else:
            jax = _try_import_jax()
            if jax is not None:
                return batch_size // jax.process_count()
            return batch_size

    def _build_pipeline(self, wds):
        """Build the WebDataset streaming pipeline."""
        # Shard-level: open tar files with shard shuffling
        pipeline = wds.WebDataset(
            self._shard_pattern,
            shardshuffle=1000 if self._shuffle else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            seed=self._seed,
        )

        # Sample-level shuffle buffer
        if self._shuffle:
            pipeline = pipeline.shuffle(self._shuffle_buffer_size, initial=self._shuffle_buffer_size // 4)

        # Decode
        if self._decoder_fn is not None:
            pipeline = pipeline.map(self._decoder_fn)
        else:
            pipeline = pipeline.decode(self._image_decode)

        # Restructure: convert flat WebDataset keys to nested OpenPI dict format
        pipeline = pipeline.map(_restructure_sample)

        # Per-sample transform
        if self._transform_fn is not None:
            pipeline = pipeline.map(self._transform_fn)

        # Batching
        pipeline = pipeline.batched(
            self._local_batch_size,
            collation_fn=_numpy_collate,
            partial=not self._drop_last,
        )

        return pipeline

    def __iter__(self) -> Iterator[dict]:
        import webdataset as wds

        torch = _try_import_torch()
        jax = _try_import_jax()

        if self._num_workers > 0 and torch is not None:
            # Use PyTorch DataLoader for parallel workers
            loader = wds.WebLoader(
                self._dataset,
                num_workers=self._num_workers,
                batch_size=None,  # batching handled by WebDataset
                prefetch_factor=self._prefetch if self._num_workers > 0 else None,
                persistent_workers=self._num_workers > 0,
                pin_memory=(self._framework == "pytorch"),
            )
        else:
            loader = self._dataset

        num_items = 0
        while True:
            for batch in loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                num_items += 1

                if self._framework == "jax" and jax is not None and self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x)
                        if isinstance(x, np.ndarray) and x.dtype.kind in ("f", "i", "u")
                        else x,
                        batch,
                    )
                elif self._framework == "pytorch" and torch is not None:
                    yield _torch_tensorize(batch, torch)
                else:
                    yield batch

            # If num_batches is None, loop forever (like OpenPI)
            if self._num_batches is None:
                continue
            else:
                return


def _restructure_sample(sample: dict) -> dict:
    """Convert flat WebDataset keys to nested OpenPI-compatible dict.

    All fields except "actions" and "prompt" go under "observation".
    """
    result = {}
    observation = {}

    for key, value in sample.items():
        if key.startswith("__"):
            continue

        # Strip file extensions to get the field name
        field = key
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".npy", ".pyd", ".txt", ".json"):
            if field.endswith(ext):
                field = field[: -len(ext)]
                break

        if field == "actions":
            result["actions"] = np.asarray(value, dtype=np.float32) if not isinstance(value, np.ndarray) else value
        elif field == "prompt":
            result["prompt"] = value if isinstance(value, str) else value.decode("utf-8") if isinstance(value, bytes) else str(value)
        else:
            observation[field] = np.asarray(value)

    if observation:
        result["observation"] = observation

    return result


def _numpy_collate(samples: list[dict]) -> dict:
    """Collate samples into batched numpy arrays. Same semantics as OpenPI's _collate_fn."""
    if not samples:
        return {}

    keys = samples[0].keys()
    result = {}
    for key in keys:
        values = [s[key] for s in samples]
        if isinstance(values[0], dict):
            result[key] = _numpy_collate(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = np.stack(values, axis=0)
        elif isinstance(values[0], str):
            result[key] = np.array(values)
        elif isinstance(values[0], (int, float)):
            result[key] = np.array(values)
        else:
            result[key] = values
    return result


def _torch_tensorize(batch: dict, torch) -> dict:
    """Recursively convert numpy arrays to torch tensors."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = _torch_tensorize(value, torch)
        elif isinstance(value, np.ndarray) and value.dtype.kind in ("f", "i", "u"):
            result[key] = torch.as_tensor(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Integration with OpenPI
# ---------------------------------------------------------------------------


class _DataLoaderImpl:
    """Wraps ScalableDataLoader to match OpenPI's DataLoader protocol.

    Yields (Observation, Actions) tuples just like OpenPI's DataLoaderImpl.
    """

    def __init__(self, data_config: Any, loader: ScalableDataLoader):
        self._data_config = data_config
        self._loader = loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        # Import here to avoid hard dependency on OpenPI
        try:
            import openpi.models.model as _model
        except ImportError:
            raise ImportError(
                "OpenPI is required for create_data_loader(). "
                "Use ScalableDataLoader directly for standalone usage."
            )

        for batch in self._loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


def create_data_loader(
    config,
    *,
    shard_pattern: str | None = None,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    # Scalable loader specific
    shuffle_buffer_size: int = 500_000,
    num_workers: int = 4,
    prefetch: int = 2,
):
    """Drop-in replacement for OpenPI's create_data_loader().

    Usage in OpenPI's train.py:
        # Before:
        from openpi.training.data_loader import create_data_loader
        # After:
        from piutil.data import create_data_loader

    Args:
        config: OpenPI TrainConfig.
        shard_pattern: WebDataset shard URL pattern. Required.
            If not provided, falls back to config.data.shard_pattern (if exists).
        sharding: JAX sharding.
        shuffle: Whether to shuffle.
        num_batches: Max batches.
        skip_norm_stats: Whether to skip normalization.
        framework: "jax" or "pytorch".
        shuffle_buffer_size: Sample-level shuffle buffer size.
        num_workers: Parallel I/O workers.
        prefetch: Prefetch count.
    """
    import openpi.training.config as _config
    import openpi.transforms as _transforms

    data_config = config.data.create(config.assets_dirs, config.model)

    # Resolve shard pattern
    if shard_pattern is None:
        shard_pattern = getattr(data_config, "shard_pattern", None) or getattr(config, "shard_pattern", None)
    if shard_pattern is None:
        raise ValueError(
            "shard_pattern is required. Provide it as an argument or add it to your config. "
            "Use piutil.data.convert_rlds_to_shards() or convert_lerobot_to_shards() "
            "to convert your existing data to WebDataset format first."
        )

    # Build transform pipeline (same as OpenPI)
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    loader = ScalableDataLoader(
        shard_pattern=shard_pattern,
        batch_size=config.batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        framework=framework,
        sharding=sharding,
        num_batches=num_batches,
        transform_fn=transforms,
        prefetch=prefetch,
    )

    return _DataLoaderImpl(data_config, loader)


def create_scalable_data_loader(
    shard_pattern: str,
    batch_size: int,
    *,
    shuffle: bool = True,
    shuffle_buffer_size: int = 500_000,
    num_workers: int = 4,
    framework: Literal["jax", "pytorch"] = "jax",
    sharding=None,
    num_batches: int | None = None,
    transform_fn: Callable[[dict], dict] | None = None,
    image_decode: str = "rgb8",
    prefetch: int = 2,
    seed: int = 0,
) -> ScalableDataLoader:
    """Create a ScalableDataLoader without OpenPI dependency.

    For standalone usage or custom integrations.

    Args:
        shard_pattern: WebDataset shard URL pattern.
        batch_size: Global batch size.
        shuffle: Whether to shuffle.
        shuffle_buffer_size: Shuffle buffer size (default 500K).
        num_workers: Parallel workers.
        framework: "jax" or "pytorch".
        sharding: JAX sharding spec.
        num_batches: Max batches per epoch.
        transform_fn: Per-sample transform.
        image_decode: Image decode mode ("rgb8", "pil").
        prefetch: Prefetch count.
        seed: Random seed.
    """
    return ScalableDataLoader(
        shard_pattern=shard_pattern,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        framework=framework,
        sharding=sharding,
        num_batches=num_batches,
        transform_fn=transform_fn,
        image_decode=image_decode,
        prefetch=prefetch,
        seed=seed,
    )
