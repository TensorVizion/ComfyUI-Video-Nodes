"""
ComfyUI Image Sharpen Node
Sharpens (or softens) an IMAGE tensor using unsharp masking.
Config: configs/image_sharpen_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_sharpen_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


def _gaussian_blur_np(arr: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """Pure-NumPy separable Gaussian blur applied per-frame."""
    k = 2 * radius + 1
    x = np.arange(k) - radius
    kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    kernel /= kernel.sum()

    def convolve1d(a, ax):
        return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), ax, a)

    out = convolve1d(arr, -2)   # width
    out = convolve1d(out, -3)   # height
    return out


class ImageSharpenNode:
    """
    Applies unsharp masking to an image or frame batch.

    Unsharp mask formula:
        output = image + strength * (image − blur(image, radius, sigma))

    Setting strength to a negative value produces a soft/blur effect instead.

    Inputs
    ------
    image    : (N, H, W, C) or (H, W, C) float32 tensor
    strength : sharpening multiplier (1.0 = moderate, higher = stronger)
    radius   : blur radius used to compute the unsharp mask (pixels)
    sigma    : Gaussian sigma for the blur kernel (auto if ≤ 0)
    threshold: only sharpen pixels where the edge magnitude exceeds this value
               (0 = sharpen everything)

    Outputs
    -------
    IMAGE : sharpened tensor, clamped to [0, 1]
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": d["strength"], "min": -5.0, "max": 10.0, "step": 0.1,
                     "tooltip": "Sharpening amount. Negative values blur instead."},
                ),
                "radius": (
                    "INT",
                    {"default": d["radius"], "min": 1, "max": 20, "step": 1,
                     "tooltip": "Blur radius for the unsharp mask kernel."},
                ),
                "sigma": (
                    "FLOAT",
                    {"default": d["sigma"], "min": 0.0, "max": 20.0, "step": 0.1,
                     "tooltip": "Gaussian sigma. 0 = auto (radius / 2)."},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": d["threshold"], "min": 0.0, "max": 1.0, "step": 0.005,
                     "tooltip": "Only sharpen where edge strength exceeds this. 0 = sharpen all."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def sharpen_image(
        self,
        image: torch.Tensor,
        strength: float,
        radius: int,
        sigma: float,
        threshold: float,
    ):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        arr = image.numpy().astype(np.float32)
        _sigma = sigma if sigma > 0 else max(radius / 2.0, 0.5)

        blurred = _gaussian_blur_np(arr, radius, _sigma)
        detail  = arr - blurred   # high-frequency component

        if threshold > 0:
            # Only apply sharpening where the absolute detail exceeds threshold
            mask = (np.abs(detail) > threshold).astype(np.float32)
            out = np.clip(arr + strength * detail * mask, 0.0, 1.0)
        else:
            out = np.clip(arr + strength * detail, 0.0, 1.0)

        result = torch.from_numpy(out)
        if squeeze:
            result = result.squeeze(0)

        return (result,)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageSharpenNode": ImageSharpenNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSharpenNode": _CFG["display_name"],
}
