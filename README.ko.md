# piutil

ML 학습 파이프라인을 위한 경량 유틸리티: 프로파일링 + 대규모 데이터 로딩.

## 설치

```bash
pip install piutil                # 코어 (프로파일링만)
pip install "piutil[all]"         # 프로파일링 + 데이터로더 + 텐서보드
pip install "piutil[data]"        # 데이터로더만 (webdataset + numpy)
pip install "piutil[dali]"        # + GPU 이미지 디코딩 (NVIDIA DALI)
```

---

## 프로파일링

### 타이머

```python
from piutil import timer, summary

with timer("forward"):
    loss = model(batch)

with timer("backward"):
    loss.backward()

print(summary())
```

### 학습 벤치마크

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

**기능:** 페이즈 타이밍, 계층적 페이즈(`/` 구분), CUDA 자동 동기화, TensorBoard/JSONL 로깅, 처리량 추적, GPU 메모리 프로파일링.

---

## 대규모 데이터 로더 (1TB+)

[OpenPI](https://github.com/Physical-Intelligence/openpi) 데이터 로더의 드롭인 대체. 1TB 이상 데이터셋에 최적화.

### 왜 필요한가?

OpenPI는 두 가지 데이터 로딩 경로를 제공하지만, 대규모에서 모두 한계가 있습니다:

| | OpenPI LeRobot | OpenPI RLDS | **piutil** |
|---|---|---|---|
| 포맷 | LeRobot (랜덤 액세스) | TF RLDS | WebDataset tar 샤드 |
| 셔플 | 전체 데이터셋 샘플러 | 250K 버퍼, 단일 레벨 | 샤드 셔플 + 500K 버퍼 (2단계) |
| 파이프라인 | PyTorch DataLoader | TF→NumPy→JAX (3단계 변환) | **타깃 프레임워크로 직접 전달** |
| 워커 | PyTorch 표준 워커 | `num_workers=0` 강제 | 다중 워커 + prefetch |
| 이미지 디코드 | CPU (PIL) | CPU (`tf.io.decode`) | CPU (PIL) 또는 **GPU (DALI)** |
| 메모리 | 메타데이터를 RAM에 로드 | `.with_ram_budget(1)` 휴리스틱 | 상수 메모리 스트리밍 |
| 스케일 한계 | 실질적으로 100GB 미만 | 1TB 이상에서 비효율적 | **1TB+ 설계** |

### 두 가지 버전

| | `piutil.data.loader` | `piutil.data.torch_loader` |
|---|---|---|
| JAX 의존성 | Optional | **없음** |
| 출력 타입 | numpy → JAX sharded array | `torch.Tensor` 직접 |
| DDP 지원 | `jax.process_count()` | `torch.distributed` |
| GPU 전송 | `jax.make_array_from_process_local_data` | `pin_memory` + `non_blocking` |
| Tree 유틸 | `jax.tree.map` | 내장 `tree_map` |
| 용도 | OpenPI `train.py` (JAX) | OpenPI `train_pytorch.py` 또는 일반 PyTorch 학습 |

**PyTorch로 학습한다면 PyTorch 버전을 선택하세요.** JAX를 완전히 제거합니다 — GPU 메모리 선점 없음, CUDA 버전 충돌 없음, 디버깅 단순화.

**JAX 버전**은 학습 루프가 JAX 기반일 때만 사용하세요 (예: OpenPI 기본 `train.py`).

### 1단계: 데이터를 WebDataset 샤드로 변환 (1회성)

```python
from piutil.data import convert_rlds_to_shards, convert_lerobot_to_shards

# RLDS/DROID에서 변환
pattern = convert_rlds_to_shards(
    data_dir="/data/rlds",
    output_dir="/data/shards",
    datasets=[
        {"name": "droid", "version": "1.0.1", "filter_dict_path": "filter.json"},
    ],
)

# LeRobot에서 변환
pattern = convert_lerobot_to_shards(
    repo_id="lerobot/aloha_sim_insertion_human",
    output_dir="/data/shards",
)

# 임의의 이터레이터에서 변환
from piutil.data import convert_iterator_to_shards
pattern = convert_iterator_to_shards(my_iterator, "/data/shards")
```

### 2단계: `train_pytorch.py`에 적용하기

아래는 OpenPI의 `scripts/train_pytorch.py`에 필요한 모든 변경 사항을 줄 단위로 설명합니다. 다른 파일은 수정 불필요.

#### 2-1. import 변경 (34번, 47번 줄)

```diff
- import jax
+ from piutil.data.torch_loader import create_torch_data_loader, tree_map
  import numpy as np
  ...
  import openpi.training.config as _config
- import openpi.training.data_loader as _data
```

`import jax`를 완전히 제거합니다. `_data`도 더 이상 필요 없습니다.

#### 2-2. `build_datasets` 함수 변경 (125~128번 줄)

```diff
- def build_datasets(config: _config.TrainConfig):
-     data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
-     return data_loader, data_loader.data_config()
+ SHARD_PATTERN = "/data/shards/shard-{000000..000999}.tar"  # ← 본인의 샤드 경로
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

#### 2-3. `build_datasets` 호출 부분 (359번 줄)

```diff
- loader, data_config = build_datasets(config)
+ loader, data_config = build_datasets(config, device=device)
```

#### 2-4. wandb 샘플 로깅 (362~390번 줄)

기존 코드는 `_data.create_data_loader`를 호출하고 `observation.to_dict()`(JAX `Observation` 클래스)를 사용합니다. piutil은 plain dict를 반환하므로 단순해집니다:

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
+     observation, actions = sample_batch  # observation은 plain dict

      images_to_log = []
-     batch_size = next(iter(sample_batch["image"].values())).shape[0]
+     # observation["image"]는 {카메라명: tensor} dict
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

#### 2-5. 학습 루프 — 디바이스 전송 (514~522번 줄)

핵심 변경입니다. 기존 `jax.tree.map`을 제거합니다:

```diff
  for observation, actions in loader:
      if global_step >= config.num_train_steps:
          break

-     observation = jax.tree.map(lambda x: x.to(device), observation)
-     actions = actions.to(torch.float32)
-     actions = actions.to(device)
+     # build_datasets에서 device= 를 넘겼으면 이미 GPU에 있음.
+     # 안 넘겼다면 여기서 수동으로:
+     # observation = tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, observation)
+     # actions = actions.to(device, dtype=torch.float32)
+     actions = actions.to(torch.float32)
```

`build_datasets`에서 `device=device`를 넘겼다면 모든 텐서가 이미 GPU에 있습니다. 안 넘겼다면 `tree_map`을 사용하세요.

#### 2-6. Forward pass — observation 포맷 (529번 줄)

모델의 `forward(observation, actions)`는 내부에서 `_preprocess_observation`을 호출하며, `Observation` 객체(`.images`, `.image_masks`, `.state`, `.tokenized_prompt` 등)를 기대합니다. 이것은 OpenPI의 transform 파이프라인이 생성합니다.

**두 가지 방법:**

**방법 A: `create_torch_data_loader`에 OpenPI transform 포함 (권장)**

`create_torch_data_loader(config, ...)`는 OpenPI와 동일한 transform 파이프라인을 자동 적용합니다 (repack → data transforms → normalize → model transforms). 출력 dict에 `image`, `image_mask`, `state`, `tokenized_prompt` 등이 이미 포함되어 있습니다.

모델에 넘기기 전에 `Observation`으로 감싸주세요:

```python
from openpi.models.model import Observation

for observation_dict, actions in loader:
    observation = Observation.from_dict(observation_dict)
    losses = model(observation, actions)
```

**방법 B: `TorchScalableDataLoader` 단독 사용 (OpenPI transform 없이)**

Transform 파이프라인을 생략하면 raw 데이터를 받게 됩니다:
```python
{"actions": tensor, "observation": {"image": tensor, "state": tensor}, "prompt": str}
```
모델에 넘기기 전에 직접 transform을 적용해야 합니다.

#### 전체 변경 요약

```
scripts/train_pytorch.py
  34번 줄:  - import jax
            + from piutil.data.torch_loader import create_torch_data_loader, tree_map
  47번 줄:  - import openpi.training.data_loader as _data
  125번 줄: build_datasets() → create_torch_data_loader 사용
  359번 줄: build_datasets(config) → build_datasets(config, device=device)
  364번 줄: wandb 샘플링 → create_torch_data_loader 사용
  520번 줄: - jax.tree.map(lambda x: x.to(device), observation)
            + tree_map(...) 또는 loader에 device= 전달
  529번 줄: model(Observation.from_dict(observation), actions)
```

변경 지점: **6곳**, 새 파일: **0개**, `import jax` **완전 제거**.

### 2단계 (JAX): `train.py`에 적용

```python
# 기존 (OpenPI)
from openpi.training.data_loader import create_data_loader

loader = create_data_loader(config, sharding=sharding, shuffle=True)

# 변경 후 (piutil)
from piutil.data import create_data_loader

loader = create_data_loader(
    config,
    shard_pattern="/data/shards/shard-{000000..001000}.tar",
    sharding=sharding,
    shuffle=True,
)
```

### 단독 사용 (OpenPI 없이)

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
    images = batch["observation"]["image"]      # GPU 위의 torch.Tensor
    actions = batch["actions"]                  # GPU 위의 torch.Tensor
```

### 선택사항: GPU 이미지 디코딩 (DALI)

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

## 프로젝트 구조

```
src/piutil/
├── __init__.py              # 최상위 re-export (하위호환)
├── profiling/               # 타이머 & 벤치마크
│   ├── timer.py
│   └── benchmark.py
└── data/                    # 대규모 데이터 로딩
    ├── loader.py            # JAX 호환 버전
    ├── torch_loader.py      # 순수 PyTorch 버전 (JAX 없음)
    ├── shard_writer.py      # 데이터 변환 유틸
    └── decode.py            # GPU 디코드 (DALI, 선택사항)
```

## 라이센스

MIT
