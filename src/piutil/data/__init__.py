"""Scalable data loading for 1TB+ datasets.

Drop-in replacement for OpenPI's data_loader.py with:
- WebDataset shard-based streaming (constant memory usage)
- Multi-level shuffling (shard + buffer) for proper randomness
- Parallel I/O with configurable workers
- Optional GPU image decoding via NVIDIA DALI

Two variants:
- loader.py: JAX-compatible (matches OpenPI's default JAX pipeline)
- torch_loader.py: Pure PyTorch, zero JAX dependency

Usage:
    # 1. Convert existing data to WebDataset shards (one-time)
    from piutil.data import ShardWriter
    writer = ShardWriter("/data/shards", max_samples_per_shard=1000)
    for sample in existing_dataset:
        writer.write_sample(sample)
    writer.close()

    # 2a. JAX version (drop-in for OpenPI's train.py)
    from piutil.data import create_data_loader
    loader = create_data_loader(config, sharding=sharding, shuffle=True)

    # 2b. PyTorch version (drop-in for train_pytorch.py, no JAX)
    from piutil.data.torch_loader import create_torch_data_loader, tree_map
    loader = create_torch_data_loader(config, shard_pattern=pattern, device="cuda")
"""

from piutil.data.loader import (
    ScalableDataLoader,
    create_data_loader,
    create_scalable_data_loader,
)
from piutil.data.shard_writer import (
    ShardWriter,
    convert_iterator_to_shards,
    convert_lerobot_to_shards,
    convert_rlds_to_shards,
)
from piutil.data.torch_loader import (
    TorchScalableDataLoader,
    create_torch_data_loader,
    tree_map,
)

__all__ = [
    # JAX-compatible
    "ScalableDataLoader",
    "create_data_loader",
    "create_scalable_data_loader",
    # PyTorch-native (no JAX)
    "TorchScalableDataLoader",
    "create_torch_data_loader",
    "tree_map",
    # Shard conversion
    "ShardWriter",
    "convert_iterator_to_shards",
    "convert_lerobot_to_shards",
    "convert_rlds_to_shards",
]
