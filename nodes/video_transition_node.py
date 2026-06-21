"""
ComfyUI Video Transition Node
Stitches two frame batches together with an animated transition region
(crossfade, directional wipe, or zoom-blur).
Config: configs/video_transition_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_transition_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Easing functions ──────────────────────────────────────────────────────────

def _ease_linear(t: np.ndarray) -> np.ndarray:
    return t

def _ease_smooth(t: np.ndarray) -> np.ndarray:
    """Cubic ease-in-out: 3t² − 2t³"""
    return t * t * (3.0 - 2.0 * t)

def _ease_sharp(t: np.ndarray) -> np.ndarray:
    """Step-like: slow centre, fast edges via quintic."""
    return np.where(t < 0.5,
                    4.0 * t * t * t,
                    1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)

_EASE_FNS = {"linear": _ease_linear, "smooth": _ease_smooth, "sharp": _ease_sharp}


# ── Transition implementations ────────────────────────────────────────────────

def _crossfade(fa: np.ndarray, fb: np.ndarray, alpha: float) -> np.ndarray:
    return fa * (1.0 - alpha) + fb * alpha


def _wipe(fa: np.ndarray, fb: np.ndarray, alpha: float, axis: int, reverse: bool) -> np.ndarray:
    """Hard-edge wipe along *axis* (1=H, 2=W)."""
    size = fa.shape[axis]
    cut  = int(round(alpha * size))
    out  = fa.copy()
    if axis == 1:   # vertical wipe (up/down)
        if reverse:
            out[:cut, :, :] = fb[:cut, :, :]
        else:
            out[size - cut:, :, :] = fb[size - cut:, :, :]
    else:           # horizontal wipe (left/right)
        if reverse:
            out[:, :cut, :] = fb[:, :cut, :]
        else:
            out[:, size - cut:, :] = fb[:, size - cut:, :]
    return out


def _zoom_blur(fa: np.ndarray, fb: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple zoom-blur transition: outgoing clip scales up and fades,
    incoming clip scales down from centre and fades in.
    Pure NumPy — no cv2 needed.
    """
    H, W = fa.shape[:2]
    # Scale fa up (zoom out feel) using nearest-neighbour crop
    scale_a = 1.0 + alpha * 0.3
    new_h = int(H / scale_a)
    new_w = int(W / scale_a)
    y0 = (H - new_h) // 2
    x0 = (W - new_w) // 2
    cropped_a = fa[y0: y0 + new_h, x0: x0 + new_w, :]
    # Resize back via repeat (fast nearest)
    rh = np.repeat(cropped_a, int(np.ceil(H / new_h)), axis=0)[:H]
    rw = np.repeat(rh,        int(np.ceil(W / new_w)), axis=1)[:, :W, :]
    zoomed_a = rw

    return zoomed_a * (1.0 - alpha) + fb * alpha


def _build_transition_frame(
    fa: np.ndarray, fb: np.ndarray,
    alpha: float, transition_type: str,
) -> np.ndarray:
    if transition_type == "crossfade":
        frame = _crossfade(fa, fb, alpha)
    elif transition_type == "wipe_left":
        frame = _wipe(fa, fb, alpha, axis=2, reverse=False)
    elif transition_type == "wipe_right":
        frame = _wipe(fa, fb, alpha, axis=2, reverse=True)
    elif transition_type == "wipe_up":
        frame = _wipe(fa, fb, alpha, axis=1, reverse=False)
    elif transition_type == "wipe_down":
        frame = _wipe(fa, fb, alpha, axis=1, reverse=True)
    elif transition_type == "zoom_blur":
        frame = _zoom_blur(fa, fb, alpha)
    else:
        frame = _crossfade(fa, fb, alpha)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


# ── Node ──────────────────────────────────────────────────────────────────────

class VideoTransitionNode:
    """
    Stitches two IMAGE batches with an animated transition zone.

    Output layout: [all frames of clip_a] + [transition_frames] + [all frames of clip_b]

    The last frame of clip_a and the first frame of clip_b serve as the
    start/end anchors of the transition — they are not repeated in the output.

    Inputs
    ------
    frames_a         : leading clip   (N_a, H, W, C)
    frames_b         : trailing clip  (N_b, H, W, C) — resized to match frames_a if needed
    transition_type  : crossfade | wipe_left | wipe_right | wipe_up | wipe_down | zoom_blur
    transition_frames: number of interpolated frames in the transition zone
    ease             : blend curve — linear | smooth | sharp

    Outputs
    -------
    IMAGE : combined frame sequence
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
                "frames_a": ("IMAGE",),
                "frames_b": ("IMAGE",),
                "transition_type": (
                    _CFG["transition_types"],
                    {"default": d["transition_type"]},
                ),
                "transition_frames": (
                    "INT",
                    {
                        "default": d["transition_frames"],
                        "min": 1, "max": 120, "step": 1,
                        "tooltip": "Number of blended frames in the transition zone.",
                    },
                ),
                "ease": (
                    _CFG["ease_types"],
                    {
                        "default": d["ease"],
                        "tooltip": (
                            "linear = constant speed | "
                            "smooth = cubic ease-in-out | "
                            "sharp = snaps quickly at the midpoint"
                        ),
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def apply_transition(
        self,
        frames_a: torch.Tensor,
        frames_b: torch.Tensor,
        transition_type: str,
        transition_frames: int,
        ease: str,
    ):
        import torch.nn.functional as F

        def ensure4d(t):
            return t.unsqueeze(0) if t.ndim == 3 else t

        fa = ensure4d(frames_a).float()
        fb = ensure4d(frames_b).float()

        # Resize fb to match fa's spatial dims if they differ
        if fb.shape[1:3] != fa.shape[1:3]:
            H, W = fa.shape[1], fa.shape[2]
            fb = F.interpolate(
                fb.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        arr_a = fa.numpy()  # (Na, H, W, C)
        arr_b = fb.numpy()  # (Nb, H, W, C)

        # Anchor frames: last of A, first of B
        anchor_a = arr_a[-1]
        anchor_b = arr_b[0]

        ease_fn = _EASE_FNS.get(ease, _ease_smooth)
        t_vals  = ease_fn(np.linspace(0.0, 1.0, transition_frames, dtype=np.float32))

        transition = np.stack([
            _build_transition_frame(anchor_a, anchor_b, float(t), transition_type)
            for t in t_vals
        ])  # (transition_frames, H, W, C)

        # Concatenate: all A (including anchor) + transition + all B (including anchor)
        combined = np.concatenate([arr_a, transition, arr_b], axis=0)
        result   = torch.from_numpy(combined.astype(np.float32))

        return (result, result.shape[0])


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoTransitionNode": VideoTransitionNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTransitionNode": _CFG["display_name"],
}
