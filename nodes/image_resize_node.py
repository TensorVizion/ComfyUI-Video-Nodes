"""
ComfyUI Image Resize Node
Resizes an IMAGE tensor to a target resolution using several interpolation modes.
Config: configs/image_resize_config.json
"""

import os
import json
import torch
import torch.nn.functional as F

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_resize_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)

_MODE_MAP = {
    "bilinear":  "bilinear",
    "nearest":   "nearest",
    "bicubic":   "bicubic",
    "area":      "area",
}


class ImageResizeNode:
    """
    Resizes a batch of image tensors to an explicit width × height.

    Supports four interpolation modes and an optional keep-aspect-ratio flag
    that adds letterbox/pillarbox padding rather than stretching the image.

    Inputs
    ------
    image        : (N, H, W, C) float32 tensor
    width        : target width  (px)
    height       : target height (px)
    interpolation: bilinear | nearest | bicubic | area
    keep_aspect  : if True, scale to fit and pad to exact size
    pad_value    : greyscale value used for padding (0 = black, 1 = white)

    Outputs
    -------
    IMAGE : resized (and optionally padded) tensor
    INT   : output width
    INT   : output height
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        modes = list(_MODE_MAP.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": d["width"], "min": 1, "max": 16384, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": d["height"], "min": 1, "max": 16384, "step": 1},
                ),
                "interpolation": (
                    modes,
                    {"default": d["interpolation"]},
                ),
                "keep_aspect": (
                    "BOOLEAN",
                    {"default": d["keep_aspect"],
                     "tooltip": "Scale to fit the target box, then pad rather than stretch."},
                ),
                "pad_value": (
                    "FLOAT",
                    {"default": d["pad_value"], "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Padding fill value (0 = black, 1 = white)."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def resize_image(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        interpolation: str,
        keep_aspect: bool,
        pad_value: float,
    ):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # (N, H, W, C) → (N, C, H, W) for F.interpolate
        N, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2).float()

        mode = _MODE_MAP.get(interpolation, "bilinear")
        align = False if mode in ("nearest", "area") else True

        if keep_aspect:
            scale = min(width / W, height / H)
            new_w = max(1, int(round(W * scale)))
            new_h = max(1, int(round(H * scale)))
            resized = F.interpolate(x, size=(new_h, new_w), mode=mode,
                                    align_corners=align if mode not in ("nearest", "area") else None)
            # Pad to exact target
            pad_l = (width  - new_w) // 2
            pad_r =  width  - new_w - pad_l
            pad_t = (height - new_h) // 2
            pad_b =  height - new_h - pad_t
            out = F.pad(resized, (pad_l, pad_r, pad_t, pad_b), value=pad_value)
        else:
            kw = {"align_corners": align} if mode not in ("nearest", "area") else {}
            out = F.interpolate(x, size=(height, width), mode=mode, **kw)

        # (N, C, H, W) → (N, H, W, C)
        out = out.permute(0, 2, 3, 1)

        if squeeze:
            out = out.squeeze(0)

        out_h, out_w = out.shape[-3], out.shape[-2]
        return (out, out_w, out_h)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageResizeNode": ImageResizeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeNode": _CFG["display_name"],
}
