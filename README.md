# piutil

Lightweight ML training utilities: profiling + scalable data loading.

## Install

```bash
pip install piutil                # core (profiling only)
pip install "piutil[all]"         # profiling + data loader + tensorboard
pip install "piutil[data]"        # data loader only (webdataset + numpy)
pip install "piutil[dali]"        # + GPU image decoding (NVIDIA DALI)
```

---

## Profiling

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

**Features:** phase timing, hierarchical phases (`/` separator), auto CUDA sync, TensorBoard/JSONL logging, throughput tracking, GPU memory profiling.

---

## Scalable Data Loader (1TB+)

Drop-in replacement for [OpenPI](https://github.com/Physical-Intelligence/openpi)'s data loader, optimized for datasets exceeding 1TB.

### Why?

OpenPI provides two data loading paths, both with limitations at scale:

| | OpenPI LeRobot | OpenPI RLDS | **piutil** |
|---|---|---|---|
| Format | LeRobot (random access) | TF RLDS | WebDataset tar shards |
| Shuffle | Full dataset in sampler | 250K buffer, single-level | Shard shuffle + 500K buffer (2-level) |
| Pipeline | PyTorch DataLoader | TF→NumPy→JAX (3 conversions) | **Direct to target framework** |
| Workers | Standard PyTorch workers | `num_workers=0` forced | Multi-worker + prefetch |
| Image decode | CPU (PIL) | CPU (`tf.io.decode`) | CPU (PIL) or **GPU (DALI)** |
| Memory | Loads metadata into RAM | `.with_ram_budget(1)` heuristic | Constant memory streaming |
| Scale limit | < 100GB practical | Works but inefficient > 1TB | **Designed for 1TB+** |

### Two Variants

| | `piutil.data.loader` | `piutil.data.torch_loader` |
|---|---|---|
| JAX dependency | Optional | **None** |
| Output type | numpy → JAX sharded arrays | `torch.Tensor` directly |
| DDP support | `jax.process_count()` | `torch.distributed` |
| GPU transfer | `jax.make_array_from_process_local_data` | `pin_memory` + `non_blocking` |
| Tree utilities | `jax.tree.map` | Built-in `tree_map` |
| Use with | OpenPI `train.py` (JAX) | OpenPI `train_pytorch.py` or any PyTorch training |

**Choose PyTorch variant** if you train with PyTorch. It removes JAX entirely — no extra GPU memory preallocation, no CUDA version conflicts, simpler debugging.

**Choose JAX variant** only if your training loop is JAX-based (e.g., OpenPI's default `train.py`).

### Step 1: Convert Data to WebDataset Shards (One-Time)

```python
from piutil.data import convert_rlds_to_shards, convert_lerobot_to_shards

# From RLDS/DROID
pattern = convert_rlds_to_shards(
    data_dir="/data/rlds",
    output_dir="/data/shards",
    datasets=[
        {"name": "droid", "version": "1.0.1", "filter_dict_path": "filter.json"},
    ],
)

# From LeRobot
pattern = convert_lerobot_to_shards(
    repo_id="lerobot/aloha_sim_insertion_human",
    output_dir="/data/shards",
)

# From any iterator
from piutil.data import convert_iterator_to_shards
pattern = convert_iterator_to_shards(my_iterator, "/data/shards")
```

### Step 2: Apply to `train_pytorch.py`

Below is a line-by-line walkthrough of every change needed in OpenPI's `scripts/train_pytorch.py`. No other files need to change.

#### 2-1. Imports (lines 34, 47)

```diff
- import jax
+ from piutil.data.torch_loader import create_torch_data_loader, tree_map
  import numpy as np
  ...
  import openpi.training.config as _config
- import openpi.training.data_loader as _data
```

`import jax` is removed entirely. `_data` is no longer needed — the loader comes from piutil.

#### 2-2. `build_datasets` function (line 125–128)

```diff
- def build_datasets(config: _config.TrainConfig):
-     data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
-     return data_loader, data_loader.data_config()
+ SHARD_PATTERN = "/data/shards/shard-{000000..000999}.tar"  # ← your shard path
+
+ def build_datasets(config: _config.TrainConfig, device=None):
+     data_loader = create_torch_data_loader(
+         config,
+         shard_pattern=SHARD_PATTERN,
+         shuffle=True,
+         num_workers=4,
+         shuffle_buffer_size=500_000,
+         pin_memory=True,
+         device=device,
+     )
+     return data_loader, data_loader.data_config()
```

#### 2-3. Calling `build_datasets` (line 359)

```diff
- loader, data_config = build_datasets(config)
+ loader, data_config = build_datasets(config, device=device)
```

#### 2-4. Wandb sample logging (lines 362–390)

The original code calls `_data.create_data_loader` and uses `observation.to_dict()` (the JAX `Observation` class). With piutil the loader yields plain dicts, so the wandb block simplifies:

```diff
  if is_main and config.wandb_enabled and not resuming:
-     sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
-     sample_batch = next(iter(sample_data_loader))
-     observation, actions = sample_batch
-     sample_batch = observation.to_dict()
-     sample_batch["actions"] = actions
+     sample_data_loader = create_torch_data_loader(
+         config, shard_pattern=SHARD_PATTERN, shuffle=False, num_workers=0,
+     )
+     sample_batch = next(iter(sample_data_loader))
+     observation, actions = sample_batch  # observation is a plain dict
+     # observation already has "image" dict and "state" tensor

      images_to_log = []
-     batch_size = next(iter(sample_batch["image"].values())).shape[0]
+     # observation["image"] is a dict of {camera_name: tensor}
+     # If your data has a single "image" key under "observation":
+     image_dict = observation.get("image", observation.get("observation", {}))
+     if not isinstance(image_dict, dict):
+         image_dict = {"cam0": image_dict}
+     batch_size = next(iter(image_dict.values())).shape[0]
      for i in range(min(5, batch_size)):
-         img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
+         img_concatenated = torch.cat(
+             [img[i] if img[i].ndim == 3 else img[i].unsqueeze(-1) for img in image_dict.values()],
+             dim=1,
+         )
          img_concatenated = img_concatenated.cpu().numpy()
          images_to_log.append(wandb.Image(img_concatenated))
      wandb.log({"camera_views": images_to_log}, step=0)
      ...
```

#### 2-5. Training loop — device transfer (lines 514–522)

This is the core change. The original uses `jax.tree.map` to move tensors to GPU:

```diff
  for observation, actions in loader:
      if global_step >= config.num_train_steps:
          break

-     observation = jax.tree.map(lambda x: x.to(device), observation)
-     actions = actions.to(torch.float32)
-     actions = actions.to(device)
+     # With device= in build_datasets, tensors are already on GPU.
+     # If you didn't pass device=, do it here:
+     # observation = tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, observation)
+     # actions = actions.to(device, dtype=torch.float32)
+     actions = actions.to(torch.float32)
```

If you passed `device=device` in `build_datasets`, all tensors arrive on GPU already. Otherwise use `tree_map` as shown.

#### 2-6. Forward pass — observation format (line 529)

The model's `forward(observation, actions)` calls `_preprocess_observation`, which expects an `Observation` object with `.images`, `.image_masks`, `.state`, `.tokenized_prompt`, etc. These are created by OpenPI's transform pipeline.

**Two options:**

**Option A: Use `create_torch_data_loader` with OpenPI's transforms (recommended)**

`create_torch_data_loader(config, ...)` automatically applies the same transform pipeline as OpenPI (repack → data transforms → normalize → model transforms). The output dict already contains `image`, `image_mask`, `state`, `tokenized_prompt`, etc.

Wrap the dict into an `Observation` before passing to the model:

```python
from openpi.models.model import Observation

for observation_dict, actions in loader:
    observation = Observation.from_dict(observation_dict)
    losses = model(observation, actions)
```

**Option B: Use `TorchScalableDataLoader` standalone (no OpenPI transforms)**

If you skip OpenPI's transform pipeline, you get raw data:
```python
{"actions": tensor, "observation": {"image": tensor, "state": tensor}, "prompt": str}
```
You must apply transforms yourself before feeding to the model.

#### Full diff summary

```
scripts/train_pytorch.py
  Line 34:  - import jax
            + from piutil.data.torch_loader import create_torch_data_loader, tree_map
  Line 47:  - import openpi.training.data_loader as _data
  Line 125: build_datasets() → use create_torch_data_loader
  Line 359: build_datasets(config) → build_datasets(config, device=device)
  Line 364: wandb sampling → use create_torch_data_loader
  Line 514: for observation, actions in loader: (unchanged)
  Line 520: - jax.tree.map(lambda x: x.to(device), observation)
            + tree_map(...) or pass device= to loader
  Line 529: model(Observation.from_dict(observation), actions) if needed
```

Total: **6 locations**, **0 new files**, **remove `import jax` entirely**.

### Step 2 (JAX): Apply to `train.py`

```python
# BEFORE (OpenPI)
from openpi.training.data_loader import create_data_loader

loader = create_data_loader(config, sharding=sharding, shuffle=True)

# AFTER (piutil)
from piutil.data import create_data_loader

loader = create_data_loader(
    config,
    shard_pattern="/data/shards/shard-{000000..001000}.tar",
    sharding=sharding,
    shuffle=True,
)
```

### Standalone Usage (No OpenPI)

```python
from piutil.data.torch_loader import TorchScalableDataLoader

loader = TorchScalableDataLoader(
    shard_pattern="/data/shards/{00000..01000}.tar",
    batch_size=32,
    num_workers=4,
    shuffle=True,
    shuffle_buffer_size=500_000,
    device="cuda",
)

for batch in loader:
    images = batch["observation"]["image"]      # torch.Tensor on GPU
    actions = batch["actions"]                  # torch.Tensor on GPU
```

### Optional: GPU Image Decoding (DALI)

```python
from piutil.data.decode import create_dali_decoder

decoder = create_dali_decoder(device_id=0, output_size=(224, 224))
loader = TorchScalableDataLoader(
    shard_pattern=pattern,
    batch_size=32,
    decoder_fn=decoder,
)
```

---

## Project Structure

```
src/piutil/
├── __init__.py              # Top-level re-exports (backward compatible)
├── profiling/               # Timer & benchmark
│   ├── timer.py
│   └── benchmark.py
└── data/                    # Scalable data loading
    ├── loader.py            # JAX-compatible variant
    ├── torch_loader.py      # Pure PyTorch variant (no JAX)
    ├── shard_writer.py      # Data conversion utilities
    └── decode.py            # GPU decode (DALI, optional)
```

## License

MIT
