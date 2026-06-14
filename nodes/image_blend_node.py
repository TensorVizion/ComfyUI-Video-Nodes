"""
ComfyUI Image Blend Node
Blends two IMAGE tensors using a blend mode and an alpha mix weight.
Config: configs/image_blend_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_blend_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Blend mode implementations ────────────────────────────────────────────────

def _blend_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b

def _blend_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def _blend_screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - a) * (1.0 - b)

def _blend_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))

def _blend_soft_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(
        b <= 0.5,
        a - (1.0 - 2.0 * b) * a * (1.0 - a),
        a + (2.0 * b - 1.0) * (_soft_d(a) - a),
    )

def _soft_d(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.25, ((16.0 * a - 12.0) * a + 4.0) * a, np.sqrt(a))

def _blend_hard_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _blend_overlay(b, a)

def _blend_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b)

def _blend_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a + b, 0.0, 1.0)

def _blend_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a - b, 0.0, 1.0)

def _blend_dodge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a / (1.0 - b + 1e-7), 0.0, 1.0)

def _blend_burn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - (1.0 - a) / (b + 1e-7), 0.0, 1.0)


_BLEND_FNS = {
    "normal":      _blend_normal,
    "multiply":    _blend_multiply,
    "screen":      _blend_screen,
    "overlay":     _blend_overlay,
    "soft_light":  _blend_soft_light,
    "hard_light":  _blend_hard_light,
    "difference":  _blend_difference,
    "add":         _blend_add,
    "subtract":    _blend_subtract,
    "dodge":       _blend_dodge,
    "burn":        _blend_burn,
}


class ImageBlendNode:
    """
    Blends two image tensors pixel-by-pixel using a Photoshop-style blend mode,
    then mixes the result with the base image via an alpha weight.

    Both images are resized to the base image's resolution if they differ in size.

    Inputs
    ------
    image_a    : base image  (N, H, W, C) or (H, W, C)
    image_b    : blend image (same or different resolution)
    blend_mode : one of the Photoshop-style modes listed above
    alpha      : 0 = fully image_a, 1 = fully blended result

    Outputs
    -------
    IMAGE : blended result, same shape as image_a
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        modes = list(_BLEND_FNS.keys())
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend_mode": (
                    modes,
                    {"default": d["blend_mode"]},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": d["alpha"], "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "0 = only image_a, 1 = fully blended result."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def blend_images(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        blend_mode: str,
        alpha: float,
    ):
        import torch.nn.functional as F  # noqa: PLC0415

        def to4d(t):
            return t.unsqueeze(0) if t.ndim == 3 else t

        squeeze = image_a.ndim == 3
        a = to4d(image_a).float()
        b = to4d(image_b).float()

        # Match spatial size of b to a
        if b.shape[1:3] != a.shape[1:3]:
            H, W = a.shape[1], a.shape[2]
            b = F.interpolate(b.permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)

        # Match batch size (broadcast single-frame b over a batch)
        if b.shape[0] == 1 and a.shape[0] > 1:
            b = b.expand(a.shape[0], -1, -1, -1)

        a_np = a.numpy()
        b_np = b.numpy()

        fn = _BLEND_FNS.get(blend_mode, _blend_normal)
        blended = np.clip(fn(a_np, b_np), 0.0, 1.0)

        out_np = a_np * (1.0 - alpha) + blended * alpha
        out = torch.from_numpy(np.clip(out_np, 0.0, 1.0))

        if squeeze:
            out = out.squeeze(0)

        return (out,)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageBlendNode": ImageBlendNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendNode": _CFG["display_name"],
}
