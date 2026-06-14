"""
ComfyUI Image Padding Node
Adds constant, reflect, or edge-replicate padding to an IMAGE tensor.
Config: configs/image_padding_config.json
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_padding_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)

_PAD_MODES = {
    "constant": "constant",
    "reflect":  "reflect",
    "replicate":"replicate",
}


class ImagePaddingNode:
    """
    Adds padding to each edge of an image or frame batch.

    Modes
    -----
    constant  : fills with a solid colour (controlled by pad_value)
    reflect   : mirror-reflects the image pixels at the border
    replicate : repeats the edge pixels outward

    Inputs
    ------
    image       : (N, H, W, C) or (H, W, C) float32 tensor
    pad_top     : pixels to add at the top
    pad_bottom  : pixels to add at the bottom
    pad_left    : pixels to add on the left
    pad_right   : pixels to add on the right
    mode        : constant | reflect | replicate
    pad_value   : fill colour for constant mode (0 = black, 1 = white)

    Outputs
    -------
    IMAGE : padded tensor
    INT   : new width
    INT   : new height
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        modes = list(_PAD_MODES.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_top": (
                    "INT",
                    {"default": d["pad_top"], "min": 0, "max": 4096, "step": 1},
                ),
                "pad_bottom": (
                    "INT",
                    {"default": d["pad_bottom"], "min": 0, "max": 4096, "step": 1},
                ),
                "pad_left": (
                    "INT",
                    {"default": d["pad_left"], "min": 0, "max": 4096, "step": 1},
                ),
                "pad_right": (
                    "INT",
                    {"default": d["pad_right"], "min": 0, "max": 4096, "step": 1},
                ),
                "mode": (
                    modes,
                    {"default": d["mode"]},
                ),
                "pad_value": (
                    "FLOAT",
                    {"default": d["pad_value"], "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Fill value for constant mode (0 = black, 1 = white)."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def pad_image(
        self,
        image: torch.Tensor,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        mode: str,
        pad_value: float,
    ):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # (N, H, W, C) → (N, C, H, W) for F.pad
        x = image.permute(0, 3, 1, 2).float()

        torch_mode = _PAD_MODES.get(mode, "constant")
        # F.pad padding order: (left, right, top, bottom) for 4-D spatial
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        if torch_mode == "constant":
            out = F.pad(x, padding, mode="constant", value=pad_value)
        else:
            # reflect / replicate require padding ≤ spatial size
            _, _, H, W = x.shape
            if pad_left >= W or pad_right >= W or pad_top >= H or pad_bottom >= H:
                # Fall back to constant padding if image too small
                out = F.pad(x, padding, mode="constant", value=pad_value)
            else:
                out = F.pad(x, padding, mode=torch_mode)

        # (N, C, H, W) → (N, H, W, C)
        out = out.permute(0, 2, 3, 1)

        if squeeze:
            out = out.squeeze(0)

        out_h, out_w = out.shape[-3], out.shape[-2]
        return (out, out_w, out_h)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImagePaddingNode": ImagePaddingNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePaddingNode": _CFG["display_name"],
}
