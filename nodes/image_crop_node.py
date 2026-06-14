"""
ComfyUI Image Crop Node
Crops a rectangular region from an IMAGE tensor (or batch of frames).
Config: configs/image_crop_config.json
"""

import os
import json
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_crop_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


class ImageCropNode:
    """
    Crops a rectangular region from an IMAGE tensor.

    Inputs
    ------
    image  : (N, H, W, C) or (H, W, C) float32 tensor
    x      : left edge of the crop box (pixels)
    y      : top  edge of the crop box (pixels)
    width  : width  of the crop box (pixels; clamped to image bounds)
    height : height of the crop box (pixels; clamped to image bounds)

    Outputs
    -------
    IMAGE  : cropped tensor, same dtype and channel count as input
    INT    : resulting crop width
    INT    : resulting crop height
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
                "x": (
                    "INT",
                    {"default": d["x"], "min": 0, "max": 16384, "step": 1,
                     "tooltip": "Left edge of the crop region (pixels)."},
                ),
                "y": (
                    "INT",
                    {"default": d["y"], "min": 0, "max": 16384, "step": 1,
                     "tooltip": "Top edge of the crop region (pixels)."},
                ),
                "width": (
                    "INT",
                    {"default": d["width"], "min": 1, "max": 16384, "step": 1,
                     "tooltip": "Desired crop width. Clamped to image bounds."},
                ),
                "height": (
                    "INT",
                    {"default": d["height"], "min": 1, "max": 16384, "step": 1,
                     "tooltip": "Desired crop height. Clamped to image bounds."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def crop_image(self, image: torch.Tensor, x: int, y: int, width: int, height: int):
        # Normalise to 4-D: (N, H, W, C)
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        _, H, W, _ = image.shape

        # Clamp to valid bounds
        x1 = min(x, W - 1)
        y1 = min(y, H - 1)
        x2 = min(x1 + width,  W)
        y2 = min(y1 + height, H)

        cropped = image[:, y1:y2, x1:x2, :]

        if squeeze:
            cropped = cropped.squeeze(0)

        out_h = y2 - y1
        out_w = x2 - x1
        return (cropped, out_w, out_h)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageCropNode": ImageCropNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropNode": _CFG["display_name"],
}
