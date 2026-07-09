import hashlib
from io import BytesIO

import numpy as np
from PIL import Image


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    v_min = float(arr.min())
    v_max = float(arr.max())

    if v_max - v_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - v_min) / (v_max - v_min)


def stable_seed_from_image(image: Image.Image) -> int:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    digest = hashlib.md5(buffer.getvalue()).hexdigest()
    return int(digest[:8], 16)