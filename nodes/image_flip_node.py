"""
ComfyUI Image Flip Node
Flips an IMAGE tensor horizontally, vertically, or both.
Config: configs/image_flip_config.json
"""

import os
import json
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_flip_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


class ImageFlipNode:
    """
    Flips a batch of image tensors along one or both spatial axes.

    Inputs
    ------
    image      : (N, H, W, C) or (H, W, C) float32 tensor
    flip_h     : flip left–right (mirror)
    flip_v     : flip top–bottom

    Outputs
    -------
    IMAGE : flipped tensor (same shape as input)
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
                "flip_h": (
                    "BOOLEAN",
                    {"default": d["flip_h"],
                     "tooltip": "Flip left–right (horizontal mirror)."},
                ),
                "flip_v": (
                    "BOOLEAN",
                    {"default": d["flip_v"],
                     "tooltip": "Flip top–bottom (vertical mirror)."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def flip_image(self, image: torch.Tensor, flip_h: bool, flip_v: bool):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        out = image
        # In (N, H, W, C) layout: dim 2 = width, dim 1 = height
        if flip_h:
            out = torch.flip(out, dims=[2])
        if flip_v:
            out = torch.flip(out, dims=[1])

        if squeeze:
            out = out.squeeze(0)

        return (out,)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageFlipNode": ImageFlipNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFlipNode": _CFG["display_name"],
}
