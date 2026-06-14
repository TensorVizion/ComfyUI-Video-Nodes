"""
ComfyUI Image Rotate Node
Rotates an IMAGE tensor by an arbitrary angle with optional auto-expand canvas.
Config: configs/image_rotate_config.json
"""

import os
import json
import math
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_rotate_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


def _rotate_frame(frame: np.ndarray, angle_deg: float, expand: bool, fill: float) -> np.ndarray:
    """Rotate a single (H, W, C) float32 frame using an affine warp."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "ImageRotateNode requires OpenCV. "
            "Install with: pip install opencv-python-headless"
        ) from exc

    H, W = frame.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    if expand:
        # Compute new canvas size that fits the rotated image without clipping
        rad = math.radians(angle_deg)
        new_w = int(abs(W * math.cos(rad)) + abs(H * math.sin(rad)))
        new_h = int(abs(W * math.sin(rad)) + abs(H * math.cos(rad)))
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        # Shift the image to the centre of the new canvas
        M[0, 2] += (new_w - W) / 2.0
        M[1, 2] += (new_h - H) / 2.0
        out_size = (new_w, new_h)
    else:
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        out_size = (W, H)

    border_val = (fill, fill, fill) if frame.ndim == 3 else fill
    rotated = cv2.warpAffine(
        frame, M, out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


class ImageRotateNode:
    """
    Rotates every frame in an IMAGE batch by a given angle (degrees).

    Positive angles rotate counter-clockwise (OpenCV convention).
    Set expand=True to automatically resize the canvas so no pixels are clipped.

    Inputs
    ------
    image      : (N, H, W, C) or (H, W, C) float32 tensor
    angle      : rotation in degrees (positive = CCW)
    expand     : grow the canvas to fit the full rotated image
    fill_value : constant colour used for border padding (0 = black, 1 = white)

    Outputs
    -------
    IMAGE : rotated tensor
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
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": (
                    "FLOAT",
                    {"default": d["angle"], "min": -360.0, "max": 360.0, "step": 0.5,
                     "tooltip": "Rotation angle in degrees. Positive = counter-clockwise."},
                ),
                "expand": (
                    "BOOLEAN",
                    {"default": d["expand"],
                     "tooltip": "Expand canvas so the full rotated image fits without clipping."},
                ),
                "fill_value": (
                    "FLOAT",
                    {"default": d["fill_value"], "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Border fill value (0 = black, 1 = white)."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def rotate_image(self, image: torch.Tensor, angle: float, expand: bool, fill_value: float):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        arr = image.numpy().astype(np.float32)
        rotated_frames = [_rotate_frame(arr[i], angle, expand, fill_value) for i in range(arr.shape[0])]
        out_arr = np.stack(rotated_frames, axis=0)
        out = torch.from_numpy(out_arr)

        if squeeze:
            out = out.squeeze(0)

        out_h, out_w = out.shape[-3], out.shape[-2]
        return (out, out_w, out_h)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageRotateNode": ImageRotateNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRotateNode": _CFG["display_name"],
}
