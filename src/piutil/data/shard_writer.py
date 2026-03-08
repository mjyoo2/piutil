"""Utilities to convert existing datasets to WebDataset shard format.

Supports conversion from:
- RLDS/DROID datasets (TensorFlow-based)
- LeRobot datasets
- Generic iterators of dict samples

One-time preprocessing step required before using ScalableDataLoader.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ShardWriter:
    """Write samples to WebDataset tar shards.

    Handles:
    - Automatic shard splitting by sample count or byte size
    - Image encoding (numpy array -> JPEG bytes)
    - Numpy array serialization
    - Text/metadata encoding

    Shard naming: {output_dir}/{prefix}-{shard_index:06d}.tar

    Sample format written to shards:
        {key}.image.jpg          - main camera image (uint8 HWC)
        {key}.wrist_image.jpg    - wrist camera image (uint8 HWC)
        {key}.actions.npy        - action chunk (float32)
        {key}.joint_position.npy - joint state (float32)
        {key}.gripper_position.npy - gripper state (float32)
        {key}.prompt.txt         - language instruction
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_samples_per_shard: int = 1000,
        max_bytes_per_shard: int | None = None,
        prefix: str = "shard",
        jpeg_quality: int = 95,
    ):
        """Initialize shard writer.

        Args:
            output_dir: Directory to write shards to.
            max_samples_per_shard: Max samples per tar file.
            max_bytes_per_shard: Max bytes per tar file (overrides sample count if set).
            prefix: Shard filename prefix.
            jpeg_quality: JPEG compression quality (1-100).
        """
        import webdataset as wds

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_samples = max_samples_per_shard
        self._max_bytes = max_bytes_per_shard
        self._prefix = prefix
        self._jpeg_quality = jpeg_quality

        self._shard_index = 0
        self._sample_index = 0
        self._current_shard_samples = 0
        self._current_shard_bytes = 0

        self._sink = wds.ShardWriter(
            str(self._output_dir / f"{prefix}-%06d.tar"),
            maxcount=max_samples_per_shard,
            maxsize=max_bytes_per_shard or (3 * 1024 ** 3),  # default 3GB per shard
        )

    def write_sample(self, sample: dict) -> None:
        """Write a single sample to the current shard.

        Args:
            sample: Dict with the sample data. Expected format:
                {
                    "actions": np.ndarray,
                    "observation": {
                        "image": np.ndarray (uint8, HWC),
                        "wrist_image": np.ndarray (uint8, HWC),
                        "joint_position": np.ndarray,
                        "gripper_position": np.ndarray,
                    },
                    "prompt": str,
                }
        """
        wds_sample = self._encode_sample(sample)
        self._sink.write(wds_sample)
        self._sample_index += 1

    def _encode_sample(self, sample: dict) -> dict:
        """Convert a nested dict sample to flat WebDataset format."""
        key = f"{self._sample_index:09d}"
        result = {"__key__": key}

        observation = sample.get("observation", {})

        # Encode all observation fields dynamically
        for field_name, field_value in observation.items():
            arr = np.asarray(field_value)
            # Images: 3D uint8 HWC arrays
            if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[-1] in (1, 3, 4):
                result[f"{field_name}.jpg"] = self._encode_image(arr)
            else:
                # State vectors, positions, etc.
                result[f"{field_name}.npy"] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr

        # Actions
        if "actions" in sample:
            result["actions.npy"] = np.asarray(sample["actions"], dtype=np.float32)

        # Prompt
        if "prompt" in sample:
            prompt = sample["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            elif not isinstance(prompt, str):
                prompt = str(prompt)
            result["prompt.txt"] = prompt

        # Any other top-level fields (as numpy or json)
        skip_keys = {"actions", "observation", "prompt", "step_id", "passes_filter"}
        for k, v in sample.items():
            if k in skip_keys:
                continue
            if isinstance(v, np.ndarray):
                result[f"{k}.npy"] = v
            elif isinstance(v, (int, float, bool, list)):
                result[f"{k}.json"] = json.dumps(v).encode("utf-8")

        return result

    def _encode_image(self, img: np.ndarray) -> bytes:
        """Encode numpy image to JPEG bytes."""
        try:
            import cv2
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            return buf.tobytes()
        except ImportError:
            pass

        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue()

    def close(self) -> str:
        """Close the writer and return the shard pattern.

        Returns:
            Shard pattern string for use with ScalableDataLoader.
        """
        n_shards = self._sink.shard  # must read before close
        self._sink.close()
        logger.info(
            f"Wrote {self._sample_index} samples to {n_shards} shards "
            f"in {self._output_dir}"
        )
        return str(self._output_dir / f"{self._prefix}-{{000000..{n_shards - 1:06d}}}.tar")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def convert_iterator_to_shards(
    iterator: Iterator[dict],
    output_dir: str | Path,
    *,
    max_samples_per_shard: int = 1000,
    prefix: str = "shard",
    jpeg_quality: int = 95,
    total: int | None = None,
) -> str:
    """Convert any iterator of dict samples to WebDataset shards.

    Args:
        iterator: Iterator yielding sample dicts (OpenPI format).
        output_dir: Directory to write shards to.
        max_samples_per_shard: Samples per shard.
        prefix: Shard filename prefix.
        jpeg_quality: JPEG quality.
        total: Total samples (for progress bar).

    Returns:
        Shard pattern string for use with ScalableDataLoader.
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=total, desc="Writing shards")
    except ImportError:
        pass

    with ShardWriter(
        output_dir,
        max_samples_per_shard=max_samples_per_shard,
        prefix=prefix,
        jpeg_quality=jpeg_quality,
    ) as writer:
        for sample in iterator:
            writer.write_sample(sample)
        return writer.close()


def convert_rlds_to_shards(
    data_dir: str,
    output_dir: str | Path,
    datasets: list[dict[str, Any]],
    *,
    action_space: str = "joint_position",
    action_chunk_size: int = 16,
    max_samples_per_shard: int = 1000,
    jpeg_quality: int = 95,
    max_episodes: int | None = None,
) -> str:
    """Convert RLDS/DROID datasets to WebDataset shards.

    This replaces the TensorFlow-based RLDS pipeline with a one-time
    conversion to an efficient streaming format.

    Args:
        data_dir: RLDS data directory.
        output_dir: Output directory for shards.
        datasets: List of dataset configs, each with keys:
            - name: dataset name
            - version: dataset version
            - filter_dict_path: optional path to filter dict JSON
        action_space: "joint_position" or "joint_velocity".
        action_chunk_size: Number of future actions per sample.
        max_samples_per_shard: Samples per shard file.
        jpeg_quality: JPEG compression quality.
        max_episodes: Max episodes to process (for debugging).

    Returns:
        Shard pattern string.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    try:
        import dlimp as dl
    except ImportError:
        raise ImportError("dlimp is required for RLDS conversion: pip install dlimp")

    tf.config.set_visible_devices([], "GPU")

    logger.info(f"Converting RLDS data from {data_dir} to {output_dir}")

    with ShardWriter(
        output_dir,
        max_samples_per_shard=max_samples_per_shard,
        jpeg_quality=jpeg_quality,
        prefix="droid",
    ) as writer:
        for ds_cfg in datasets:
            ds_name = ds_cfg["name"]
            version = ds_cfg["version"]
            filter_dict_path = ds_cfg.get("filter_dict_path")

            logger.info(f"Processing dataset: {ds_name}:{version}")
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            ds = builder.as_dataset(split="train")

            # Load filter dict if provided
            filter_dict = None
            if filter_dict_path is not None:
                with open(filter_dict_path) as f:
                    filter_dict = json.load(f)
                logger.info(f"Filter dict loaded: {len(filter_dict)} episodes")

            episode_count = 0
            for episode in ds:
                if max_episodes is not None and episode_count >= max_episodes:
                    break

                # Check success
                file_path = episode["traj_metadata"]["episode_metadata"]["file_path"][0].numpy().decode()
                if "success" not in file_path:
                    continue

                episode_count += 1
                recording_path = episode["traj_metadata"]["episode_metadata"]["recording_folderpath"][0].numpy().decode()

                # Extract data
                if action_space == "joint_position":
                    actions = episode["action_dict"]["joint_position"].numpy()
                else:
                    actions = episode["action_dict"]["joint_velocity"].numpy()
                gripper = episode["action_dict"]["gripper_position"].numpy()
                actions = np.concatenate([actions, gripper], axis=-1)

                obs = episode["observation"]
                ext_img_1 = obs["exterior_image_1_left"].numpy()
                ext_img_2 = obs["exterior_image_2_left"].numpy()
                wrist_img = obs["wrist_image_left"].numpy()
                joint_pos = obs["joint_position"].numpy()
                gripper_pos = obs["gripper_position"].numpy()

                instructions = [
                    episode["language_instruction"].numpy().decode(),
                    episode["language_instruction_2"].numpy().decode(),
                    episode["language_instruction_3"].numpy().decode(),
                ]

                traj_len = len(actions)

                for t in range(traj_len):
                    # Filter check
                    if filter_dict is not None:
                        step_id = f"{recording_path}--{file_path}--{t}"
                        if step_id not in filter_dict:
                            continue

                    # Action chunk
                    chunk_indices = np.minimum(np.arange(t, t + action_chunk_size), traj_len - 1)
                    action_chunk = actions[chunk_indices]

                    # Randomly pick exterior image (deterministic based on step for reproducibility)
                    ext_img = ext_img_1[t] if (t % 2 == 0) else ext_img_2[t]

                    # Decode images if they're encoded
                    if ext_img.ndim == 0 or (ext_img.ndim == 1 and ext_img.dtype == np.uint8):
                        ext_img = tf.io.decode_image(ext_img, expand_animations=False).numpy()
                    if wrist_img[t].ndim == 0 or (wrist_img[t].ndim == 1 and wrist_img[t].dtype == np.uint8):
                        wrist_decoded = tf.io.decode_image(wrist_img[t], expand_animations=False).numpy()
                    else:
                        wrist_decoded = wrist_img[t]

                    # Pick a random instruction (deterministic)
                    instruction = instructions[t % len(instructions)]

                    sample = {
                        "actions": action_chunk.astype(np.float32),
                        "observation": {
                            "image": ext_img,
                            "wrist_image": wrist_decoded,
                            "joint_position": joint_pos[t].astype(np.float32),
                            "gripper_position": gripper_pos[t].astype(np.float32),
                        },
                        "prompt": instruction,
                    }
                    writer.write_sample(sample)

            logger.info(f"Processed {episode_count} episodes from {ds_name}")

    return writer.close()


def convert_lerobot_to_shards(
    repo_id: str,
    output_dir: str | Path,
    *,
    action_horizon: int = 16,
    action_sequence_keys: tuple[str, ...] = ("actions",),
    max_samples_per_shard: int = 1000,
    jpeg_quality: int = 95,
) -> str:
    """Convert a LeRobot dataset to WebDataset shards.

    Args:
        repo_id: LeRobot dataset repo ID.
        output_dir: Output directory for shards.
        action_horizon: Action sequence length.
        action_sequence_keys: Keys for action sequences.
        max_samples_per_shard: Samples per shard.
        jpeg_quality: JPEG quality.

    Returns:
        Shard pattern string.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    logger.info(f"Converting LeRobot dataset {repo_id} to shards at {output_dir}")

    meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys
        },
    )

    def sample_iterator():
        for i in range(len(dataset)):
            sample = dataset[i]
            # Convert LeRobot format to OpenPI format
            result = {}

            # Actions
            if "actions" in sample:
                result["actions"] = np.asarray(sample["actions"], dtype=np.float32)

            # Observation - extract image keys and state keys
            observation = {}
            for key, value in sample.items():
                if key == "actions":
                    continue
                value = np.asarray(value)
                if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                    # Likely an image (HWC format)
                    observation[key] = value.astype(np.uint8) if value.dtype != np.uint8 else value
                elif value.ndim <= 2:
                    # State vector
                    observation[key] = value.astype(np.float32)

            if observation:
                result["observation"] = observation

            # Prompt from task if available
            if hasattr(sample, "get") and "task" in sample:
                result["prompt"] = str(sample["task"])

            yield result

    return convert_iterator_to_shards(
        sample_iterator(),
        output_dir,
        max_samples_per_shard=max_samples_per_shard,
        jpeg_quality=jpeg_quality,
        prefix="lerobot",
        total=len(dataset),
    )
