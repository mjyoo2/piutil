"""End-to-end demo: LeRobot dataset -> WebDataset shards -> ScalableDataLoader.

Downloads a small LeRobot dataset, converts to shards, then loads with
the PyTorch-native data loader (and optionally JAX loader).

Usage:
    python examples/data_loader_demo.py
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Download a small LeRobot dataset and inspect it
# ---------------------------------------------------------------------------

def step1_inspect_lerobot():
    """Download and inspect the ALOHA sim dataset from LeRobot."""
    print("=" * 60)
    print("STEP 1: Inspect LeRobot dataset")
    print("=" * 60)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = "lerobot/aloha_sim_transfer_cube_human"
    print(f"Loading: {repo_id}")

    meta = LeRobotDatasetMetadata(repo_id)
    print(f"  FPS: {meta.fps}")

    # Load without delta_timestamps (single-step actions)
    dataset = LeRobotDataset(repo_id)
    print(f"  Total samples: {len(dataset)}")

    # Inspect one sample
    sample = dataset[0]
    print(f"  Sample keys: {sorted(sample.keys())}")
    for key in sorted(sample.keys()):
        value = sample[key]
        if hasattr(value, "shape"):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"    {key}: {type(value).__name__} = {repr(value)[:80]}")

    return repo_id, meta, dataset


# ---------------------------------------------------------------------------
# Step 2: Convert to WebDataset shards
# ---------------------------------------------------------------------------

def step2_convert_to_shards(dataset, meta, output_dir):
    """Convert LeRobot dataset to WebDataset shards."""
    print("\n" + "=" * 60)
    print("STEP 2: Convert to WebDataset shards")
    print("=" * 60)

    from piutil.data.shard_writer import ShardWriter

    shard_dir = Path(output_dir) / "shards"
    writer = ShardWriter(shard_dir, max_samples_per_shard=200, jpeg_quality=90)

    start = time.perf_counter()
    n_samples = min(len(dataset), 500)  # limit for demo speed

    for i in range(n_samples):
        sample = dataset[i]

        # Build the sample dict in OpenPI-like format:
        #   actions: float32 array
        #   observation: {image: uint8 HWC, state: float32, ...}
        #   prompt: str
        obs = {}

        for key, value in sample.items():
            val = np.asarray(value)

            # Skip non-observation fields
            if key in ("action", "actions", "episode_index", "frame_index",
                       "timestamp", "next.done", "index", "task_index", "task"):
                continue

            # Images: CHW float -> HWC uint8
            if val.ndim == 3 and val.shape[0] in (1, 3, 4):
                img = np.transpose(val, (1, 2, 0))  # CHW -> HWC
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)

                # Map to standard names: first image -> "image"
                if "image" not in obs:
                    obs["image"] = img
                else:
                    clean_key = key.replace("observation.images.", "").replace("observation.", "")
                    obs[clean_key] = img

            # State vectors
            elif val.ndim <= 1 and val.dtype.kind == "f":
                clean_key = key.replace("observation.", "")
                obs[clean_key] = val.astype(np.float32)

        # Action key: LeRobot uses "action" (singular)
        action_key = "action" if "action" in sample else "actions"
        action_val = np.asarray(sample[action_key], dtype=np.float32)

        shard_sample = {
            "actions": action_val,
            "observation": obs,
        }

        # Add prompt from task
        if "task" in sample:
            shard_sample["prompt"] = str(sample["task"])

        writer.write_sample(shard_sample)

        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{n_samples} samples...")

    elapsed = time.perf_counter() - start
    shard_pattern = writer.close()

    print(f"  Done: {n_samples} samples in {elapsed:.1f}s")

    # Show shard files
    shard_files = sorted(shard_dir.glob("*.tar"))
    total_bytes = sum(f.stat().st_size for f in shard_files)
    print(f"  Shards: {len(shard_files)} files, {total_bytes / 1024 / 1024:.1f} MB total")

    return shard_dir


# ---------------------------------------------------------------------------
# Step 3: Load with TorchScalableDataLoader
# ---------------------------------------------------------------------------

def step3_torch_loader(shard_dir):
    """Load shards with the pure PyTorch data loader."""
    print("\n" + "=" * 60)
    print("STEP 3: TorchScalableDataLoader (no JAX)")
    print("=" * 60)

    import torch
    from piutil.data.torch_loader import TorchScalableDataLoader, tree_map

    shard_files = sorted(shard_dir.glob("*.tar"))
    if not shard_files:
        print("  ERROR: No shard files found!")
        return False

    shard_urls = [str(f) for f in shard_files]

    loader = TorchScalableDataLoader(
        shard_pattern=shard_urls,
        batch_size=4,
        shuffle=True,
        shuffle_buffer_size=100,
        num_workers=0,  # keep it simple for demo
        num_batches=5,
        pin_memory=False,
    )

    print("  Loading 5 batches...")
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        print(f"\n  Batch {i}:")
        _print_batch(batch, indent=4)

        # Count tensor types
        tensor_count = [0]
        non_tensor_count = [0]

        def count_types(x):
            if isinstance(x, torch.Tensor):
                tensor_count[0] += 1
            elif isinstance(x, np.ndarray):
                non_tensor_count[0] += 1
            return x

        tree_map(count_types, batch)
        print(f"    -> torch.Tensors: {tensor_count[0]}, np.arrays: {non_tensor_count[0]}")

    elapsed = time.perf_counter() - start
    print(f"\n  5 batches in {elapsed:.2f}s ({elapsed / 5 * 1000:.0f}ms/batch)")

    # Test tree_map device transfer simulation
    print("\n  Testing tree_map (simulating .to(device))...")
    batch_copy = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, batch)
    print("  tree_map OK")

    # Test iteration pattern matching OpenPI's train_pytorch.py:
    #   for observation, actions in loader:
    print("\n  Testing OpenPI iteration pattern...")
    loader2 = TorchScalableDataLoader(
        shard_pattern=shard_urls,
        batch_size=4,
        shuffle=False,
        shuffle_buffer_size=10,
        num_workers=0,
        num_batches=2,
        pin_memory=False,
    )
    for batch in loader2:
        actions = batch["actions"]
        observation = {k: v for k, v in batch.items() if k != "actions"}
        print(f"    actions: {actions.shape}")
        print(f"    observation keys: {list(observation.keys())}")
        break
    print("  OpenPI pattern OK")

    return True


# ---------------------------------------------------------------------------
# Step 4: Load with ScalableDataLoader (JAX-compatible, optional)
# ---------------------------------------------------------------------------

def step4_jax_loader(shard_dir):
    """Load shards with the JAX-compatible data loader."""
    print("\n" + "=" * 60)
    print("STEP 4: ScalableDataLoader (JAX-compatible)")
    print("=" * 60)

    try:
        import jax
        print(f"  JAX available: {jax.__version__}")
    except ImportError:
        print("  JAX not installed, skipping JAX loader test.")
        return True

    from piutil.data.loader import ScalableDataLoader

    shard_urls = [str(f) for f in sorted(shard_dir.glob("*.tar"))]

    loader = ScalableDataLoader(
        shard_pattern=shard_urls,
        batch_size=4,
        shuffle=True,
        shuffle_buffer_size=100,
        num_workers=0,
        num_batches=3,
        framework="jax",
    )

    print("  Loading 3 batches...")
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        print(f"\n  Batch {i}:")
        _print_batch(batch, indent=4)

    elapsed = time.perf_counter() - start
    print(f"\n  3 batches in {elapsed:.2f}s")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_batch(batch, indent=2):
    """Pretty-print a nested batch dict."""
    prefix = " " * indent
    for key, value in batch.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_batch(value, indent + 2)
        elif hasattr(value, "shape"):
            print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            sample = value[:2]
            print(f"{prefix}{key}: list[{len(value)}] = {sample}...")
        else:
            print(f"{prefix}{key}: {type(value).__name__} = {repr(value)[:60]}")


def main():
    with tempfile.TemporaryDirectory(prefix="piutil_demo_") as tmpdir:
        print(f"Working directory: {tmpdir}\n")

        # Step 1: Download & inspect
        repo_id, meta, dataset = step1_inspect_lerobot()

        # Step 2: Convert to shards
        shard_dir = step2_convert_to_shards(dataset, meta, tmpdir)

        # Step 3: PyTorch loader (no JAX)
        ok = step3_torch_loader(shard_dir)
        if not ok:
            print("\nFAILED: PyTorch loader")
            sys.exit(1)

        # Step 4: JAX loader (optional)
        step4_jax_loader(shard_dir)

        print("\n" + "=" * 60)
        print("ALL STEPS PASSED")
        print("=" * 60)


if __name__ == "__main__":
    main()
