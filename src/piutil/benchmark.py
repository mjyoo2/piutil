"""Training benchmark: GPU profiling, throughput tracking, TensorBoard integration.

Usage:
    from piutil import Benchmark

    bench = Benchmark(log_dir="runs/exp1")

    for step, batch in enumerate(loader):
        bench.step_start()

        with bench.phase("data/to_device"):
            batch = batch.to(device)

        with bench.phase("forward"):
            loss = model(batch)

        with bench.phase("backward"):
            loss.backward()

        with bench.phase("optimizer"):
            optimizer.step()

        bench.step_end(step, samples=len(batch))

    bench.close()
    print(bench.summary())
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from piutil.timer import TimerStats, _check_cuda, _cuda_sync, _format_time


@dataclass
class GPUMemory:
    """Snapshot of GPU memory state (in bytes)."""
    allocated: int = 0
    reserved: int = 0
    peak_allocated: int = 0
    peak_reserved: int = 0

    @staticmethod
    def snapshot(device: int = 0) -> "GPUMemory":
        try:
            import torch
            if not torch.cuda.is_available():
                return GPUMemory()
            return GPUMemory(
                allocated=torch.cuda.memory_allocated(device),
                reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device),
            )
        except ImportError:
            return GPUMemory()

    def to_dict(self, unit: str = "MB") -> dict[str, float]:
        div = {"B": 1, "KB": 1024, "MB": 2**20, "GB": 2**30}[unit]
        return {
            f"gpu_mem/allocated_{unit}": self.allocated / div,
            f"gpu_mem/reserved_{unit}": self.reserved / div,
            f"gpu_mem/peak_allocated_{unit}": self.peak_allocated / div,
            f"gpu_mem/peak_reserved_{unit}": self.peak_reserved / div,
        }

    @staticmethod
    def reset_peak(device: int = 0):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
        except ImportError:
            pass


class ThroughputTracker:
    """Track steps/sec and samples/sec with a sliding window."""

    def __init__(self, window: int = 50):
        self.window = window
        self._step_times: list[float] = []
        self._step_samples: list[int] = []
        self._total_steps = 0
        self._total_samples = 0
        self._start_time: float | None = None

    def record(self, step_time: float, samples: int = 0):
        if self._start_time is None:
            self._start_time = time.perf_counter() - step_time
        self._step_times.append(step_time)
        self._step_samples.append(samples)
        self._total_steps += 1
        self._total_samples += samples
        if len(self._step_times) > self.window:
            self._step_times.pop(0)
            self._step_samples.pop(0)

    def to_dict(self) -> dict[str, float]:
        if not self._step_times:
            return {}
        window_time = sum(self._step_times)
        window_steps = len(self._step_times)
        d = {
            "throughput/steps_per_sec": window_steps / window_time if window_time > 0 else 0,
            "throughput/sec_per_step": window_time / window_steps if window_steps > 0 else 0,
        }
        window_samples = sum(self._step_samples)
        if window_samples > 0:
            d["throughput/samples_per_sec"] = window_samples / window_time if window_time > 0 else 0
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        if elapsed > 0:
            d["throughput/cumulative_steps_per_sec"] = self._total_steps / elapsed
        return d


class Benchmark:
    """Main training benchmark — the only class you need.

    Combines phase timing, GPU memory profiling, throughput tracking,
    TensorBoard logging, and JSONL export in one simple interface.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        log_every: int = 10,
        jsonl_path: str | Path | None = None,
        cuda_sync: bool | None = None,
        throughput_window: int = 50,
    ):
        """
        Args:
            log_dir: TensorBoard log directory. None = no TB logging.
            log_every: Print console summary every N steps.
            jsonl_path: Path to JSONL file. None = auto (log_dir/metrics.jsonl if log_dir set).
            cuda_sync: Force CUDA sync on/off. None = auto-detect.
            throughput_window: Sliding window size for throughput calculation.
        """
        self._cuda_sync = cuda_sync if cuda_sync is not None else _check_cuda()
        self._log_every = log_every
        self._throughput = ThroughputTracker(window=throughput_window)
        self._phases: dict[str, TimerStats] = {}
        self._step_start: float = 0.0
        self._writer = None
        self._jsonl_file = None
        self._current_step: int = 0

        # TensorBoard
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(log_dir))
            except ImportError:
                pass

        # JSONL
        if jsonl_path is not None:
            p = Path(jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(p, "a")
        elif log_dir is not None:
            p = Path(log_dir) / "metrics.jsonl"
            self._jsonl_file = open(p, "a")

    def step_start(self):
        """Call at the beginning of each training step."""
        if self._cuda_sync:
            _cuda_sync()
        self._step_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str):
        """Time a phase within a step. Use '/' for hierarchy."""
        if name not in self._phases:
            self._phases[name] = TimerStats(name=name)

        if self._cuda_sync:
            _cuda_sync()

        start = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda_sync:
                _cuda_sync()
            self._phases[name].times.append(time.perf_counter() - start)

    def step_end(self, step: int, *, samples: int = 0, extra: dict[str, float] | None = None):
        """Call at the end of each training step.

        Args:
            step: Current training step number.
            samples: Number of samples in this step's batch.
            extra: Additional metrics to log (e.g., {"loss": 0.5, "lr": 1e-4}).
        """
        if self._cuda_sync:
            _cuda_sync()

        step_time = time.perf_counter() - self._step_start
        self._current_step = step
        self._throughput.record(step_time, samples)

        # Collect all metrics
        metrics: dict[str, float] = {"step_time": step_time}

        # Phase timings (last recorded time)
        for name, stats in self._phases.items():
            if stats.times:
                metrics[f"time/{name}"] = stats.last

        # Throughput
        metrics.update(self._throughput.to_dict())

        # GPU memory
        if self._cuda_sync:
            metrics.update(GPUMemory.snapshot().to_dict())

        # User-provided extras
        if extra:
            metrics.update(extra)

        # Write to TensorBoard
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

        # Write to JSONL
        if self._jsonl_file is not None:
            record = {"step": step, **metrics}
            self._jsonl_file.write(json.dumps(record) + "\n")
            self._jsonl_file.flush()

        # Console output
        if self._log_every > 0 and step % self._log_every == 0:
            self._print_step(step, step_time, metrics, extra)

    def _print_step(self, step: int, step_time: float, metrics: dict, extra: dict | None):
        parts = [f"[step {step:>6d}]  {_format_time(step_time).strip()}/step"]

        tp = metrics.get("throughput/samples_per_sec")
        if tp and tp > 0:
            parts.append(f"{tp:.1f} samples/s")

        # Show phase breakdown
        phase_parts = []
        for name, stats in self._phases.items():
            if stats.times:
                pct = (stats.last / step_time * 100) if step_time > 0 else 0
                phase_parts.append(f"{name}={_format_time(stats.last).strip()}({pct:.0f}%)")
        if phase_parts:
            parts.append("  ".join(phase_parts))

        if extra:
            for k, v in extra.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4g}")
                else:
                    parts.append(f"{k}={v}")

        print("  ".join(parts))

    def summary(self) -> str:
        """Return a formatted summary of all recorded phases."""
        if not self._phases:
            return "No phases recorded."

        headers = ["Phase", "Count", "Total", "Avg", "Min", "Max", "Last"]
        rows = []
        for stats in sorted(self._phases.values(), key=lambda s: s.total, reverse=True):
            rows.append([
                stats.name,
                str(stats.count),
                _format_time(stats.total),
                _format_time(stats.avg),
                _format_time(stats.min),
                _format_time(stats.max),
                _format_time(stats.last),
            ])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

        lines = [sep, header_line, sep]
        for row in rows:
            line = "|" + "|".join(f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
            lines.append(line)
        lines.append(sep)

        # Add throughput info
        tp = self._throughput.to_dict()
        if tp:
            lines.append("")
            if "throughput/steps_per_sec" in tp:
                lines.append(f"Throughput: {tp['throughput/steps_per_sec']:.2f} steps/sec")
            if "throughput/samples_per_sec" in tp:
                lines.append(f"           {tp['throughput/samples_per_sec']:.1f} samples/sec")

        return "\n".join(lines)

    def close(self):
        """Flush and close all outputs."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
