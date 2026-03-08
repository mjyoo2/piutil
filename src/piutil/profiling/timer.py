"""Lightweight hierarchical timer for ML training profiling.

Usage:
    # Context manager
    with timer("data_loading"):
        batch = next(data_iter)

    # Decorator
    @timer("forward_pass")
    def forward(model, batch):
        return model(batch)

    # Nested timing
    with timer("train_step"):
        with timer("train_step/forward"):
            loss = model(batch)
        with timer("train_step/backward"):
            loss.backward()

    # Get stats
    print(summary())           # Pretty table
    get_timer("forward").avg   # Programmatic access

    # CUDA sync (auto-detected, or explicit)
    with timer("gpu_op", cuda_sync=True):
        output = model(x)
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

_cuda_available: bool | None = None


def _check_cuda() -> bool:
    global _cuda_available
    if _cuda_available is None:
        try:
            import torch
            _cuda_available = torch.cuda.is_available()
        except ImportError:
            _cuda_available = False
    return _cuda_available


def _cuda_sync():
    import torch
    torch.cuda.synchronize()


@dataclass
class TimerStats:
    """Accumulated statistics for a named timer."""
    name: str
    times: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0

    @property
    def last(self) -> float:
        return self.times[-1] if self.times else 0.0

    def reset(self):
        self.times.clear()


# Global registry
_timers: dict[str, TimerStats] = {}
_active_stack: list[str] = []


def get_timer(name: str) -> TimerStats:
    """Get timer stats by name. Creates if not exists."""
    if name not in _timers:
        _timers[name] = TimerStats(name=name)
    return _timers[name]


def reset_timers():
    """Reset all timer stats."""
    _timers.clear()
    _active_stack.clear()


@contextmanager
def timer(name: str, *, cuda_sync: bool | None = None):
    """Context manager / decorator for timing code blocks.

    Args:
        name: Timer name. Use '/' for hierarchy (e.g., "train_step/forward").
        cuda_sync: Whether to synchronize CUDA before measuring.
                   None = auto-detect (sync if CUDA available).
    """
    do_sync = cuda_sync if cuda_sync is not None else _check_cuda()

    if do_sync:
        _cuda_sync()

    _active_stack.append(name)
    start = time.perf_counter()

    try:
        yield get_timer(name)
    finally:
        if do_sync:
            _cuda_sync()

        elapsed = time.perf_counter() - start
        get_timer(name).times.append(elapsed)
        _active_stack.pop()


def timer_decorator(name: str | None = None, *, cuda_sync: bool | None = None) -> Callable:
    """Use timer as a decorator.

    @timer_decorator("forward")
    def forward(model, x):
        return model(x)
    """
    def decorator(fn: Callable) -> Callable:
        timer_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timer(timer_name, cuda_sync=cuda_sync):
                return fn(*args, **kwargs)

        return wrapper
    return decorator


# Allow `timer` to also work as a decorator when called with a function
_original_timer = timer


class _TimerDispatcher:
    """Makes `timer` work as both context manager and decorator."""

    def __call__(self, name_or_fn: str | Callable = "", *, cuda_sync: bool | None = None):
        if callable(name_or_fn):
            # Used as @timer without arguments on a function
            return timer_decorator()(name_or_fn)
        # Used as context manager: with timer("name"):
        # or as decorator factory: @timer("name")
        return _original_timer(name_or_fn, cuda_sync=cuda_sync)

    def __enter__(self):
        raise TypeError("Use timer('name') with a name argument, e.g., with timer('my_block'):")

    def __exit__(self, *args):
        pass


timer = _TimerDispatcher()


def _format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:>8.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:>8.2f}ms"
    if seconds < 60.0:
        return f"{seconds:>8.3f}s "
    minutes = seconds / 60
    return f"{minutes:>8.2f}m "


def summary(sort_by: str = "total") -> str:
    """Return a formatted summary table of all timers.

    Args:
        sort_by: Sort key - "total", "avg", "count", "name".
    """
    if not _timers:
        return "No timers recorded."

    headers = ["Name", "Count", "Total", "Avg", "Min", "Max", "Last"]
    rows = []

    for stats in _timers.values():
        rows.append([
            stats.name,
            str(stats.count),
            _format_time(stats.total),
            _format_time(stats.avg),
            _format_time(stats.min),
            _format_time(stats.max),
            _format_time(stats.last),
        ])

    sort_keys = {
        "total": lambda r: _timers[r[0]].total,
        "avg": lambda r: _timers[r[0]].avg,
        "count": lambda r: _timers[r[0]].count,
        "name": lambda r: r[0],
    }
    rows.sort(key=sort_keys.get(sort_by, sort_keys["total"]), reverse=(sort_by != "name"))

    # Calculate column widths
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

    return "\n".join(lines)


def to_dict() -> dict[str, dict[str, float]]:
    """Export all timer stats as a dict (useful for wandb logging)."""
    return {
        name: {
            "count": s.count,
            "total": s.total,
            "avg": s.avg,
            "min": s.min,
            "max": s.max,
            "last": s.last,
        }
        for name, s in _timers.items()
    }
