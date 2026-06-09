"""
ComfyUI Video Effects Node
Applies a chain of visual effects (colour grade, blur, vignette, film grain,
speed ramp) to a batch of image tensors.
Config: configs/video_effects_config.json
"""

import os
import json
import math
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_effects_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_colour_grade(arr: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Per-channel colour grade on float32 [0,1] HWC or NHWC array."""
    out = arr * brightness                          # brightness
    out = (out - 0.5) * contrast + 0.5             # contrast around mid-grey
    # Saturation via luminance mix
    lum = 0.299 * out[..., 0:1] + 0.587 * out[..., 1:2] + 0.114 * out[..., 2:3]
    out = lum + saturation * (out - lum)
    return np.clip(out, 0.0, 1.0)


def _apply_gaussian_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Simple box-blur approximation (no cv2 dependency)."""
    if radius < 1:
        return arr
    # Build 1-D kernel weights
    k = 2 * radius + 1
    kernel = np.ones((k,), dtype=np.float32) / k

    def blur1d(a, ax):
        return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), ax, a)

    out = blur1d(arr, -2)  # width axis
    out = blur1d(out, -3)  # height axis
    return out


def _apply_vignette(arr: np.ndarray, strength: float) -> np.ndarray:
    """Radial vignette that darkens corners."""
    if strength <= 0.0:
        return arr
    H, W = arr.shape[-3], arr.shape[-2]
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx ** 2 + yy ** 2) / math.sqrt(2.0)   # 0 centre → 1 corner
    mask = 1.0 - np.clip(dist * strength, 0.0, 1.0)       # (H,W)
    if arr.ndim == 4:
        mask = mask[np.newaxis, :, :, np.newaxis]          # (1,H,W,1)
    else:
        mask = mask[:, :, np.newaxis]                      # (H,W,1)
    return np.clip(arr * mask, 0.0, 1.0)


def _apply_film_grain(arr: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white-noise film grain."""
    if amount <= 0.0:
        return arr
    noise = rng.standard_normal(arr.shape).astype(np.float32) * amount
    return np.clip(arr + noise, 0.0, 1.0)


def _apply_speed_ramp(arr: np.ndarray, speed: float) -> np.ndarray:
    """
    Simple temporal re-sample.
    speed < 1 → slow motion (upsample frames with nearest-neighbour repeat)
    speed > 1 → fast forward (drop frames)
    """
    N = arr.shape[0]
    if abs(speed - 1.0) < 1e-4 or N < 2:
        return arr
    new_N = max(1, int(round(N / speed)))
    indices = np.round(np.linspace(0, N - 1, new_N)).astype(int)
    return arr[indices]


# ── Node ──────────────────────────────────────────────────────────────────────

class VideoEffectsNode:
    """
    Applies a configurable stack of post-processing effects to a video frame batch.

    Effects applied in order:
      1. Colour grade  (brightness / contrast / saturation)
      2. Gaussian blur
      3. Vignette
      4. Film grain
      5. Speed ramp   (temporal re-sample)
    """

    CATEGORY = _CFG["category"]
    FUNCTION  = _CFG["function"]
    RETURN_TYPES  = tuple(_CFG["return_types"])
    RETURN_NAMES  = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        return {
            "required": {
                "frames": ("IMAGE",),
                # ── Colour grade ──────────────────────────────────────────
                "brightness": (
                    "FLOAT",
                    {"default": d["brightness"], "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": d["contrast"], "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": d["saturation"], "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                # ── Blur ─────────────────────────────────────────────────
                "blur_radius": (
                    "INT",
                    {
                        "default": d["blur_radius"],
                        "min": 0,
                        "max": 20,
                        "step": 1,
                        "tooltip": "0 = disabled. Box-blur radius in pixels.",
                    },
                ),
                # ── Vignette ──────────────────────────────────────────────
                "vignette_strength": (
                    "FLOAT",
                    {
                        "default": d["vignette_strength"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "0 = disabled. Higher = darker corners.",
                    },
                ),
                # ── Film grain ────────────────────────────────────────────
                "grain_amount": (
                    "FLOAT",
                    {
                        "default": d["grain_amount"],
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Standard deviation of additive noise. 0 = disabled.",
                    },
                ),
                "grain_seed": (
                    "INT",
                    {
                        "default": d["grain_seed"],
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "RNG seed for reproducible grain. 0 = random each run.",
                    },
                ),
                # ── Speed ─────────────────────────────────────────────────
                "speed": (
                    "FLOAT",
                    {
                        "default": d["speed"],
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "1.0 = normal speed, 0.5 = half speed, 2.0 = double speed.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def apply_effects(
        self,
        frames: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
        blur_radius: int,
        vignette_strength: float,
        grain_amount: float,
        grain_seed: int,
        speed: float,
    ):
        if frames.ndim != 4:
            raise ValueError(f"VideoEffectsNode expects (N,H,W,C) tensor; got {frames.shape}")

        arr = frames.numpy().astype(np.float32)

        # 1. Colour grade
        arr = _apply_colour_grade(arr, brightness, contrast, saturation)

        # 2. Blur (applied per-frame to avoid excessive memory use)
        if blur_radius > 0:
            arr = np.stack([_apply_gaussian_blur(arr[i], blur_radius) for i in range(arr.shape[0])], axis=0)

        # 3. Vignette
        arr = _apply_vignette(arr, vignette_strength)

        # 4. Film grain
        if grain_amount > 0.0:
            seed = grain_seed if grain_seed != 0 else None
            rng  = np.random.default_rng(seed)
            arr  = _apply_film_grain(arr, grain_amount, rng)

        # 5. Speed ramp (may change frame count)
        arr = _apply_speed_ramp(arr, speed)

        result = torch.from_numpy(arr)
        return (result, result.shape[0])


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoEffectsNode": VideoEffectsNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoEffectsNode": _CFG["display_name"],
}
