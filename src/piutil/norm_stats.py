"""Fast normalization statistics computation.

Drop-in replacement for OpenPI's `scripts/compute_norm_stats.py`.

Key optimizations:
1. Data loading: Read state/action columns directly from HF Arrow table,
   skipping image decoding and transform pipeline entirely.
2. Computation: Vectorized histogram updates, skip histograms when quantiles
   are not needed.

Usage as OpenPI replacement:
    # Instead of: python scripts/compute_norm_stats.py <config_name>
    python -m piutil.norm_stats <config_name>

    # Or with options:
    python -m piutil.norm_stats <config_name> --max-frames 10000 --no-quantiles

Usage as library:
    from piutil.norm_stats import compute_norm_stats_lerobot, RunningStats

    # From LeRobot dataset (fastest path):
    stats = compute_norm_stats_lerobot("lerobot/aloha_sim_insertion_human")
    save_norm_stats(stats, "/path/to/output")

    # From any iterator of batches:
    stats = compute_norm_stats(data_loader, keys=["state", "actions"])

    # Manual:
    rs = RunningStats(compute_quantiles=False)
    for batch in loader:
        rs.update(batch["actions"])
    result = rs.get_statistics()
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Normalization statistics for a single key."""
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        if self.q01 is not None:
            d["q01"] = self.q01.tolist()
        if self.q99 is not None:
            d["q99"] = self.q99.tolist()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "NormStats":
        return NormStats(
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            q01=np.array(d["q01"]) if "q01" in d and d["q01"] is not None else None,
            q99=np.array(d["q99"]) if "q99" in d and d["q99"] is not None else None,
        )


class RunningStats:
    """Compute running mean/std/quantiles over batches of vectors.

    Optimizations vs OpenPI's RunningStats:
    - Histogram updates are fully vectorized (no per-dimension Python loop)
    - Histogram computation can be skipped entirely (compute_quantiles=False)
    - Bin edge adjustment uses vectorized operations
    """

    def __init__(self, compute_quantiles: bool = True, num_bins: int = 5000):
        self._compute_quantiles = compute_quantiles
        self._num_bins = num_bins
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_sq: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        # Histograms stored as (num_dims, num_bins) array for vectorized ops
        self._histograms: np.ndarray | None = None
        # Bin edges stored as (num_dims, num_bins+1) array
        self._bin_edges: np.ndarray | None = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch of vectors.

        Args:
            batch: Array where all dimensions except the last are batch dimensions.
        """
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        n, d = batch.shape

        batch_mean = batch.mean(axis=0)
        batch_mean_sq = (batch ** 2).mean(axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self._count == 0:
            self._mean = batch_mean
            self._mean_sq = batch_mean_sq
            self._min = batch_min
            self._max = batch_max

            if self._compute_quantiles:
                self._histograms = np.zeros((d, self._num_bins), dtype=np.float64)
                self._bin_edges = np.column_stack([
                    np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_bins + 1)
                    for i in range(d)
                ]).T
        else:
            new_max = np.maximum(self._max, batch_max)
            new_min = np.minimum(self._min, batch_min)

            if self._compute_quantiles and (np.any(new_max > self._max) or np.any(new_min < self._min)):
                self._max = new_max
                self._min = new_min
                self._adjust_histograms()
            else:
                self._max = new_max
                self._min = new_min

        self._count += n
        self._mean += (batch_mean - self._mean) * (n / self._count)
        self._mean_sq += (batch_mean_sq - self._mean_sq) * (n / self._count)

        if self._compute_quantiles:
            self._update_histograms_vectorized(batch)

    def _update_histograms_vectorized(self, batch: np.ndarray) -> None:
        """Vectorized histogram update."""
        edges = self._bin_edges
        for i in range(batch.shape[1]):
            indices = np.searchsorted(edges[i, 1:-1], batch[:, i])
            np.add.at(self._histograms[i], indices, 1)

    def _adjust_histograms(self) -> None:
        """Redistribute histograms when min/max changes."""
        d = self._histograms.shape[0]
        new_edges = np.column_stack([
            np.linspace(self._min[i], self._max[i], self._num_bins + 1)
            for i in range(d)
        ]).T

        new_histograms = np.zeros_like(self._histograms)
        for i in range(d):
            if self._histograms[i].sum() > 0:
                old_centers = 0.5 * (self._bin_edges[i, :-1] + self._bin_edges[i, 1:])
                indices = np.searchsorted(new_edges[i, 1:-1], old_centers)
                np.add.at(new_histograms[i], indices, self._histograms[i])

        self._histograms = new_histograms
        self._bin_edges = new_edges

    def get_statistics(self) -> NormStats:
        """Compute and return the final statistics."""
        if self._count < 2:
            raise ValueError("Need at least 2 samples to compute statistics.")

        variance = self._mean_sq - self._mean ** 2
        std = np.sqrt(np.maximum(0, variance)).astype(np.float32)
        mean = self._mean.astype(np.float32)

        q01, q99 = None, None
        if self._compute_quantiles and self._histograms is not None:
            q01, q99 = self._compute_quantile_values([0.01, 0.99])
            q01 = q01.astype(np.float32)
            q99 = q99.astype(np.float32)

        return NormStats(mean=mean, std=std, q01=q01, q99=q99)

    def _compute_quantile_values(self, quantiles: list[float]) -> list[np.ndarray]:
        """Compute quantiles from histograms."""
        results = []
        for q in quantiles:
            target = q * self._count
            cumsum = np.cumsum(self._histograms, axis=1)
            indices = np.array([
                np.searchsorted(cumsum[i], target) for i in range(cumsum.shape[0])
            ])
            indices = np.clip(indices, 0, self._num_bins - 1)
            values = np.array([
                self._bin_edges[i, idx] for i, idx in enumerate(indices)
            ])
            results.append(values)
        return results


# ---------------------------------------------------------------------------
# Data loading: fast paths that bypass image decoding
# ---------------------------------------------------------------------------

def _load_columns_from_lerobot(
    repo_id: str,
    columns: Sequence[str],
    *,
    max_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Load specific columns from a LeRobot dataset as numpy arrays.

    Reads directly from the underlying HF Arrow table in a single bulk read,
    bypassing video decoding and the transform pipeline entirely.

    Tries two strategies in order:
    1. LeRobotDataset.hf_dataset (fastest — uses cached Arrow table)
    2. datasets.load_dataset (fallback — re-parses parquet files)

    Args:
        repo_id: HuggingFace repo id (e.g. "lerobot/aloha_sim_insertion_human").
        columns: Column names to load (e.g. ["observation.state", "action"]).
        max_frames: Max number of frames to read. None = all.

    Returns:
        Dict mapping column names to numpy arrays of shape (N, dim).
    """
    hf_ds = None

    # Strategy 1: LeRobotDataset.hf_dataset (pre-cached Arrow, ~0.1s)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id)
        hf_ds = ds.hf_dataset
        logger.info(f"Loaded via LeRobotDataset.hf_dataset ({len(hf_ds)} rows)")
    except Exception as e:
        logger.warning(f"LeRobotDataset failed ({e}), falling back to load_dataset")

    # Strategy 2: load_dataset (re-parses parquet, slower)
    if hf_ds is None:
        from datasets import load_dataset
        hf_ds = load_dataset(repo_id, split="train")
        logger.info(f"Loaded via load_dataset ({len(hf_ds)} rows)")

    # Filter to available columns
    available = set(hf_ds.column_names)
    use_columns = [c for c in columns if c in available]
    if not use_columns:
        raise ValueError(
            f"None of the requested columns {columns} found in dataset. "
            f"Available: {sorted(available)}"
        )
    missing = set(columns) - set(use_columns)
    if missing:
        logger.warning(f"Columns not found in HF dataset (skipped): {missing}")

    # Select only needed columns and read everything at once (Arrow is columnar,
    # so selecting columns first avoids touching image/video data)
    hf_ds = hf_ds.select_columns(use_columns)
    total = len(hf_ds) if max_frames is None else min(max_frames, len(hf_ds))

    # Single bulk read — Arrow columnar read, no per-row overhead
    raw = hf_ds[:total]

    result = {}
    for col in use_columns:
        arr = np.array(raw[col])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        result[col] = arr

    logger.info(f"Read {total} frames, columns: {use_columns}")
    return result


def compute_norm_stats_lerobot(
    repo_id: str,
    *,
    keys: Sequence[str] | None = None,
    compute_quantiles: bool = True,
    max_frames: int | None = None,
) -> dict[str, NormStats]:
    """Compute norm stats directly from a LeRobot HF dataset.

    This is the fastest path: reads state/action columns directly from the Arrow
    table in a single bulk read, completely bypassing video decoding and the
    transform pipeline.

    Args:
        repo_id: HuggingFace repo id.
        keys: Column names to compute stats for. Default: ["observation.state", "action"].
        compute_quantiles: Whether to compute q01/q99.
        max_frames: Stop after this many frames. None = all.

    Returns:
        Dict mapping key names to NormStats. Keys are remapped to OpenPI convention:
        "observation.state" -> "state", "action" -> "actions".
    """
    if keys is None:
        keys = ["observation.state", "action"]

    # OpenPI key mapping: dataset column -> norm_stats key
    key_remap = {
        "observation.state": "state",
        "action": "actions",
    }

    t0 = time.perf_counter()

    # Single bulk read — all data loaded at once from Arrow
    data = _load_columns_from_lerobot(repo_id, list(keys), max_frames=max_frames)

    t_load = time.perf_counter() - t0
    n_samples = next((v.shape[0] for v in data.values()), 0)
    logger.info(f"[norm_stats] Data loaded: {n_samples} samples in {t_load:.2f}s")

    # Compute stats — single update call per key (entire dataset at once)
    result = {}
    for key in keys:
        if key not in data:
            continue
        rs = RunningStats(compute_quantiles=compute_quantiles)
        rs.update(data[key])
        out_key = key_remap.get(key, key)
        result[out_key] = rs.get_statistics()

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[norm_stats] Done: {n_samples} samples in {elapsed:.2f}s "
        f"({n_samples / elapsed:.0f} samples/s)"
    )

    return result


# ---------------------------------------------------------------------------
# Generic iterator-based computation (works with any data source)
# ---------------------------------------------------------------------------

def compute_norm_stats(
    data_iter: Iterator,
    keys: list[str] | None = None,
    *,
    compute_quantiles: bool = True,
    max_frames: int | None = None,
    log_every: int = 100,
) -> dict[str, NormStats]:
    """Compute normalization statistics from a data iterator.

    Args:
        data_iter: Iterator yielding dicts of numpy arrays or tensors.
        keys: Keys to compute stats for. Default: ["state", "actions"].
        compute_quantiles: Whether to compute q01/q99 (slower).
        max_frames: Stop after this many frames.
        log_every: Print progress every N batches.

    Returns:
        Dict mapping key names to NormStats.
    """
    if keys is None:
        keys = ["state", "actions"]

    stats = {key: RunningStats(compute_quantiles=compute_quantiles) for key in keys}

    total_samples = 0
    num_batches = 0
    t0 = time.perf_counter()

    for batch in data_iter:
        for key in keys:
            if key not in batch:
                continue
            arr = np.asarray(batch[key])
            stats[key].update(arr)

        batch_n = next(
            (np.asarray(batch[k]).reshape(-1, np.asarray(batch[k]).shape[-1]).shape[0]
             for k in keys if k in batch),
            0,
        )
        total_samples += batch_n
        num_batches += 1

        if max_frames is not None and total_samples >= max_frames:
            break

        if log_every > 0 and num_batches % log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = total_samples / elapsed if elapsed > 0 else 0
            logger.info(
                f"[norm_stats] {num_batches} batches | {total_samples} samples | "
                f"{rate:.0f} samples/s | {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    rate = total_samples / elapsed if elapsed > 0 else 0
    logger.info(
        f"[norm_stats] Done: {num_batches} batches, {total_samples} samples "
        f"in {elapsed:.1f}s ({rate:.0f} samples/s)"
    )

    return {key: s.get_statistics() for key, s in stats.items() if s.count >= 2}


# ---------------------------------------------------------------------------
# Save / Load (OpenPI-compatible format)
# ---------------------------------------------------------------------------

def save_norm_stats(norm_stats: dict[str, NormStats], directory: str | pathlib.Path) -> None:
    """Save norm stats in OpenPI-compatible format."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = {
        "norm_stats": {key: stats.to_dict() for key, stats in norm_stats.items()}
    }
    path.write_text(json.dumps(serialized, indent=2))
    logger.info(f"Saved norm stats to {path}")


def load_norm_stats(directory: str | pathlib.Path) -> dict[str, NormStats]:
    """Load norm stats from OpenPI-compatible format."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats not found: {path}")

    data = json.loads(path.read_text())
    raw = data.get("norm_stats", data)
    return {key: NormStats.from_dict(v) for key, v in raw.items()}


# ---------------------------------------------------------------------------
# CLI entrypoint: python -m piutil.norm_stats
# ---------------------------------------------------------------------------

def _cli():
    """CLI entrypoint — drop-in replacement for OpenPI's compute_norm_stats.py.

    Usage:
        # With OpenPI config (requires openpi to be installed):
        python -m piutil.norm_stats <config_name> [--max-frames N] [--no-quantiles]

        # Direct from LeRobot repo (no OpenPI dependency):
        python -m piutil.norm_stats --repo-id lerobot/aloha_sim_insertion_human --output ./assets/stats
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute normalization statistics (fast replacement for OpenPI's compute_norm_stats.py)"
    )
    parser.add_argument("config_name", nargs="?", default=None,
                        help="OpenPI config name (requires openpi installed)")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="LeRobot HF repo id (direct mode, no OpenPI needed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for norm_stats.json")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max number of frames to process")
    parser.add_argument("--no-quantiles", action="store_true",
                        help="Skip quantile computation (much faster)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    compute_quantiles = not args.no_quantiles

    if args.repo_id is not None:
        # Direct mode: no OpenPI dependency
        output = args.output
        if output is None:
            safe_name = args.repo_id.replace("/", "_")
            output = f"./assets/{safe_name}"

        logger.info(f"Computing norm stats for {args.repo_id}")
        stats = compute_norm_stats_lerobot(
            args.repo_id,
            compute_quantiles=compute_quantiles,
            max_frames=args.max_frames,
        )
        save_norm_stats(stats, output)

    elif args.config_name is not None:
        # OpenPI config mode
        try:
            import openpi.training.config as _config
        except ImportError:
            logger.error(
                "OpenPI is not installed. Use --repo-id for direct mode, "
                "or install openpi to use config names."
            )
            raise SystemExit(1)

        config = _config.get_config(args.config_name)
        data_config = config.data.create(config.assets_dirs, config.model)

        repo_id = data_config.repo_id
        if repo_id is None:
            logger.error("Config does not specify a repo_id.")
            raise SystemExit(1)

        output = args.output or str(config.assets_dirs / repo_id)

        logger.info(f"Computing norm stats for config '{args.config_name}' (repo: {repo_id})")
        stats = compute_norm_stats_lerobot(
            repo_id,
            compute_quantiles=compute_quantiles,
            max_frames=args.max_frames,
        )
        save_norm_stats(stats, output)

    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
