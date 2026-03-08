"""GPU-accelerated image decoding via NVIDIA DALI.

Optional module - falls back to CPU decoding if DALI is not available.
Provides a custom decoder function for ScalableDataLoader.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DALI_AVAILABLE = False
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import feed_ndarray
    _DALI_AVAILABLE = True
except ImportError:
    pass


def is_dali_available() -> bool:
    return _DALI_AVAILABLE


def create_dali_decoder(
    device_id: int = 0,
    output_size: tuple[int, int] | None = None,
    dtype: str = "uint8",
) -> Any:
    """Create a DALI-based image decoder for GPU-accelerated decoding.

    Usage with ScalableDataLoader:
        decoder = create_dali_decoder(device_id=0, output_size=(224, 224))
        loader = ScalableDataLoader(
            shard_pattern=pattern,
            batch_size=32,
            decoder_fn=decoder,
        )

    Args:
        device_id: CUDA device ID.
        output_size: If set, resize images to (height, width) during decode.
        dtype: Output dtype ("uint8" or "float32").

    Returns:
        Decoder function compatible with ScalableDataLoader.
    """
    if not _DALI_AVAILABLE:
        logger.warning(
            "NVIDIA DALI not available, falling back to CPU decode. "
            "Install with: pip install nvidia-dali-cuda120"
        )
        return _cpu_decode_fn(output_size)

    return _DaliDecoder(device_id=device_id, output_size=output_size, dtype=dtype)


class _DaliDecoder:
    """Stateful DALI decoder that processes WebDataset samples."""

    def __init__(self, device_id: int = 0, output_size: tuple[int, int] | None = None, dtype: str = "uint8"):
        self._device_id = device_id
        self._output_size = output_size
        self._dtype = dtype
        self._image_keys = None

    def __call__(self, sample: dict) -> dict:
        """Decode a single WebDataset sample using DALI for images."""
        result = {}
        for key, value in sample.items():
            if key.startswith("__"):
                result[key] = value
                continue

            if isinstance(value, bytes) and _is_image_key(key):
                result[key] = self._decode_image(value)
            elif isinstance(value, bytes) and key.endswith(".npy"):
                result[key] = np.load(np.lib.npyio.BytesIO(value), allow_pickle=False)
            elif isinstance(value, bytes) and key.endswith(".txt"):
                result[key] = value.decode("utf-8")
            else:
                result[key] = value

        return result

    def _decode_image(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to numpy array using DALI."""
        import nvidia.dali.fn as fn
        from nvidia.dali import pipeline_def, types
        import nvidia.dali.backend as backend

        # For single-image decode, use the external source approach
        @pipeline_def(batch_size=1, num_threads=1, device_id=self._device_id)
        def decode_pipe():
            encoded = types.Constant(np.frombuffer(jpeg_bytes, dtype=np.uint8), device="cpu")
            decoded = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
            if self._output_size is not None:
                decoded = fn.resize(decoded, size=self._output_size)
            return decoded

        pipe = decode_pipe()
        pipe.build()
        output = pipe.run()
        result = output[0].as_cpu().at(0)
        return result


def _cpu_decode_fn(output_size: tuple[int, int] | None = None):
    """CPU fallback decoder using PIL."""

    def decode(sample: dict) -> dict:
        result = {}
        for key, value in sample.items():
            if key.startswith("__"):
                result[key] = value
                continue

            if isinstance(value, bytes) and _is_image_key(key):
                result[key] = _cpu_decode_image(value, output_size)
            elif isinstance(value, bytes) and key.endswith(".npy"):
                import io
                result[key] = np.load(io.BytesIO(value), allow_pickle=False)
            elif isinstance(value, bytes) and key.endswith(".txt"):
                result[key] = value.decode("utf-8")
            else:
                result[key] = value

        return result

    return decode


def _cpu_decode_image(data: bytes, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Decode image bytes to numpy array using PIL."""
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if output_size is not None:
        img = img.resize((output_size[1], output_size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _is_image_key(key: str) -> bool:
    """Check if a WebDataset key corresponds to an image."""
    return any(key.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))
