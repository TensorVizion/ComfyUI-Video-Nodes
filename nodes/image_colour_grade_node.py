"""
ComfyUI Image Colour Grade Node
Applies per-channel colour grading (brightness, contrast, saturation,
hue shift, gamma, temperature) to a single image or batch of frames.
Config: configs/image_colour_grade_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_colour_grade_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV, input/output (N,H,W,3) float32 in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8

    h = np.where(cmax == r, (g - b) / delta % 6,
        np.where(cmax == g, (b - r) / delta + 2,
                             (r - g) / delta + 4)) / 6.0
    s = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB."""
    h, s, v = hsv[..., 0] * 6.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.stack([v, v, v], axis=-1).copy()
    for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                      (p, q, v), (t, p, v), (v, p, q)]):
        mask = (i == idx)[..., np.newaxis]
        rgb = np.where(mask, np.stack([r, g, b], axis=-1), rgb)
    return np.clip(rgb, 0.0, 1.0)


def _apply_temperature(arr: np.ndarray, temp: float) -> np.ndarray:
    """Warm (>0) adds red/yellow; cool (<0) adds blue. Range [-1, 1]."""
    if abs(temp) < 1e-4:
        return arr
    out = arr.copy()
    if temp > 0:
        out[..., 0] = np.clip(out[..., 0] + temp * 0.2, 0, 1)   # red up
        out[..., 2] = np.clip(out[..., 2] - temp * 0.1, 0, 1)   # blue down
    else:
        out[..., 2] = np.clip(out[..., 2] - temp * 0.2, 0, 1)   # blue up
        out[..., 0] = np.clip(out[..., 0] + temp * 0.1, 0, 1)   # red down
    return out


class ImageColourGradeNode:
    """
    Full-featured colour-grading node for still images and video frame batches.

    Processing order
    ----------------
    1. Temperature shift
    2. Brightness (multiplicative)
    3. Contrast    (pivot at 0.5)
    4. Gamma       (power curve)
    5. Hue shift   (via HSV)
    6. Saturation  (via HSV)

    Inputs
    ------
    image       : (N, H, W, C) or (H, W, C) float32 tensor
    brightness  : multiplicative brightness scale (1.0 = neutral)
    contrast    : contrast factor pivoted at mid-grey (1.0 = neutral)
    saturation  : saturation multiplier (0 = greyscale, 1 = neutral, 2 = vivid)
    hue_shift   : hue rotation in degrees (-180 to 180)
    gamma       : power-curve gamma (1.0 = linear, <1 = brighter mids)
    temperature : colour temperature shift (-1 cool → +1 warm)

    Outputs
    -------
    IMAGE : colour-graded tensor, same shape as input
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
                "hue_shift": (
                    "FLOAT",
                    {"default": d["hue_shift"], "min": -180.0, "max": 180.0, "step": 1.0,
                     "tooltip": "Hue rotation in degrees."},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": d["gamma"], "min": 0.1, "max": 5.0, "step": 0.05,
                     "tooltip": "Power-curve gamma. 1.0 = linear. <1 = brighter midtones."},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": d["temperature"], "min": -1.0, "max": 1.0, "step": 0.02,
                     "tooltip": "Colour temperature: -1 = cool/blue, +1 = warm/orange."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def colour_grade(
        self,
        image: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
        hue_shift: float,
        gamma: float,
        temperature: float,
    ):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        arr = image.numpy().astype(np.float32)

        # 1. Temperature
        arr = _apply_temperature(arr, temperature)

        # 2. Brightness
        arr = np.clip(arr * brightness, 0.0, 1.0)

        # 3. Contrast (pivot at 0.5)
        arr = np.clip((arr - 0.5) * contrast + 0.5, 0.0, 1.0)

        # 4. Gamma
        arr = np.clip(np.power(np.clip(arr, 0.0, 1.0), 1.0 / gamma), 0.0, 1.0)

        # 5 & 6. Hue shift + Saturation (HSV space)
        if abs(hue_shift) > 1e-4 or abs(saturation - 1.0) > 1e-4:
            hsv = _rgb_to_hsv(arr)
            if abs(hue_shift) > 1e-4:
                hsv[..., 0] = (hsv[..., 0] + hue_shift / 360.0) % 1.0
            if abs(saturation - 1.0) > 1e-4:
                hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
            arr = _hsv_to_rgb(hsv)

        out = torch.from_numpy(arr)
        if squeeze:
            out = out.squeeze(0)

        return (out,)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageColourGradeNode": ImageColourGradeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColourGradeNode": _CFG["display_name"],
}
