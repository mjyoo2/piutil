# piutil

Lightweight profiling utilities for ML training pipelines.

## Install

```bash
pip install piutil
pip install "piutil[all]"  # with torch + tensorboard
```

## Quick Start

### Simple Timer

```python
from piutil import timer, summary

with timer("forward"):
    loss = model(batch)

with timer("backward"):
    loss.backward()

print(summary())
```

### Training Benchmark

```python
from piutil import Benchmark

bench = Benchmark(log_dir="runs/exp1", log_every=10)

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

    bench.step_end(step, samples=len(batch), extra={"loss": loss.item()})

print(bench.summary())
bench.close()
```

## Features

- **Phase timing** — `with bench.phase("name"):` to wrap any code block
- **Hierarchical phases** — use `/` separator (e.g., `forward/transformer`)
- **CUDA sync** — auto-detected for accurate GPU timing
- **TensorBoard** — all metrics logged as scalars
- **JSONL export** — structured per-step metrics
- **Throughput** — steps/sec, samples/sec with sliding window
- **GPU memory** — allocated, reserved, peak tracking
- **Console output** — periodic summary with percentage breakdown
- **Zero dependencies** — torch and tensorboard are optional

## License

MIT
