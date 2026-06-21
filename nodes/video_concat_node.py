"""
ComfyUI Video Concat Node
Concatenates two or three IMAGE frame batches into a single sequence
with optional crossfade blending at each join point.
Config: configs/video_concat_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_concat_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_to(frames: torch.Tensor, H: int, W: int) -> torch.Tensor:
    import torch.nn.functional as F
    if frames.shape[1] == H and frames.shape[2] == W:
        return frames
    x = F.interpolate(
        frames.permute(0, 3, 1, 2).float(), size=(H, W),
        mode="bilinear", align_corners=False,
    )
    return x.permute(0, 2, 3, 1)


def _target_hw(clips: list, mode: str) -> tuple:
    """Return (H, W) target based on resolution_mode."""
    shapes = [(c.shape[1], c.shape[2]) for c in clips]
    if mode == "resize_to_largest":
        H = max(s[0] for s in shapes)
        W = max(s[1] for s in shapes)
    elif mode == "resize_to_smallest":
        H = min(s[0] for s in shapes)
        W = min(s[1] for s in shapes)
    else:  # resize_to_first
        H, W = shapes[0]
    return H, W


def _crossfade_join(arr_a: np.ndarray, arr_b: np.ndarray, n: int) -> np.ndarray:
    """
    Blend the last *n* frames of A with the first *n* frames of B
    using a linear crossfade, then concatenate with non-overlapping frames.
    """
    if n <= 0 or len(arr_a) < n or len(arr_b) < n:
        return np.concatenate([arr_a, arr_b], axis=0)

    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None, None, None]
    blended  = arr_a[-n:] * (1.0 - t) + arr_b[:n] * t
    return np.concatenate([arr_a[:-n], blended, arr_b[n:]], axis=0)


# ── Node ──────────────────────────────────────────────────────────────────────

class VideoConcatNode:
    """
    Joins two or three IMAGE frame batches into one continuous sequence.

    Resolution handling
    -------------------
    If clips have different spatial dimensions, match_resolution determines
    how they are reconciled before joining.

    Crossfade
    ---------
    crossfade_frames > 0 blends the tail of each clip with the head of the
    next, reducing the hard cut.  The blended region replaces (not extends)
    those frames, so the total duration is shorter by crossfade_frames per join.

    Inputs
    ------
    clip_1            : first IMAGE batch  (required)
    clip_2            : second IMAGE batch (required)
    clip_3            : third IMAGE batch  (optional — leave disconnected to skip)
    crossfade_frames  : frames to blend at each join (0 = hard cut)
    match_resolution  : resize_to_first | resize_to_largest | resize_to_smallest

    Outputs
    -------
    IMAGE : concatenated frame sequence
    INT   : total frame count
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
                "clip_1": ("IMAGE",),
                "clip_2": ("IMAGE",),
                "crossfade_frames": (
                    "INT",
                    {
                        "default": d["crossfade_frames"],
                        "min": 0, "max": 60, "step": 1,
                        "tooltip": "0 = hard cut. >0 blends the junction instead of replacing frames.",
                    },
                ),
                "match_resolution": (
                    _CFG["resolution_modes"],
                    {
                        "default": d["match_resolution"],
                        "tooltip": "How to reconcile clips with different spatial dimensions.",
                    },
                ),
            },
            "optional": {
                "clip_3": ("IMAGE",),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def concat_videos(
        self,
        clip_1: torch.Tensor,
        clip_2: torch.Tensor,
        crossfade_frames: int,
        match_resolution: str,
        clip_3: torch.Tensor | None = None,
    ):
        def ensure4d(t):
            return t.unsqueeze(0) if t.ndim == 3 else t

        clips = [ensure4d(c).float() for c in [clip_1, clip_2] if c is not None]
        if clip_3 is not None:
            clips.append(ensure4d(clip_3).float())

        # Unify spatial resolution
        H, W = _target_hw(clips, match_resolution)
        clips = [_resize_to(c, H, W) for c in clips]

        # Join sequentially with optional crossfade
        arr = clips[0].numpy()
        for nxt in clips[1:]:
            arr = _crossfade_join(arr, nxt.numpy(), crossfade_frames)

        result = torch.from_numpy(arr.astype(np.float32))
        print(
            f"[VideoConcatNode] joined {len(clips)} clip(s) → "
            f"{result.shape[0]} frames  ({H}×{W})  crossfade={crossfade_frames}"
        )
        return (result, result.shape[0])


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoConcatNode": VideoConcatNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcatNode": _CFG["display_name"],
}
