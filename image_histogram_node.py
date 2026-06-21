"""
ComfyUI Image Histogram Node
Analyses tonal distribution and applies optional histogram corrections
(equalize, CLAHE, stretch). Outputs both a corrected IMAGE and a
histogram visualisation tensor.
Config: configs/image_histogram_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_histogram_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Histogram correction implementations ──────────────────────────────────────

def _equalize_channel(ch: np.ndarray) -> np.ndarray:
    """Global histogram equalisation on a single [0,1] float channel."""
    u8   = (ch * 255).clip(0, 255).astype(np.uint8)
    hist, _ = np.histogram(u8.ravel(), bins=256, range=(0, 255))
    cdf  = hist.cumsum().astype(np.float32)
    cdf  = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
    lut  = (cdf * 255).astype(np.uint8)
    return lut[u8].astype(np.float32) / 255.0


def _stretch_channel(ch: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Linear contrast stretch: remaps [plo, phi] to [0, 1]."""
    plo = np.percentile(ch, lo_pct)
    phi = np.percentile(ch, hi_pct)
    if phi - plo < 1e-6:
        return ch
    return np.clip((ch - plo) / (phi - plo), 0.0, 1.0)


def _clahe_channel(ch: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE via OpenCV (already a project dependency)."""
    import cv2
    u8    = (ch * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out   = clahe.apply(u8)
    return out.astype(np.float32) / 255.0


def _apply_mode(
    img: np.ndarray,   # (H, W, 3) float32
    mode: str,
    channel: str,
    clip_limit: float,
) -> np.ndarray:
    if mode == "none":
        return img

    # Determine which channels to process
    if channel == "rgb":
        ch_idxs = [0, 1, 2]
    elif channel == "r":
        ch_idxs = [0]
    elif channel == "g":
        ch_idxs = [1]
    elif channel == "b":
        ch_idxs = [2]
    else:  # luminance — work on Y in YCbCr
        ch_idxs = []   # handled separately below

    out = img.copy()

    if channel == "luminance":
        # Convert to YCbCr, process Y, convert back
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        Y   =  0.299  * r + 0.587  * g + 0.114  * b
        Cb  = -0.1687 * r - 0.3313 * g + 0.5    * b + 0.5
        Cr  =  0.5    * r - 0.4187 * g - 0.0813 * b + 0.5

        if mode == "equalize":
            Y = _equalize_channel(Y)
        elif mode == "clahe":
            Y = _clahe_channel(Y, clip_limit)
        elif mode == "stretch":
            Y = _stretch_channel(Y)

        R = Y + 1.402  * (Cr - 0.5)
        G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
        B = Y + 1.772  * (Cb - 0.5)
        out = np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0).astype(np.float32)
    else:
        for idx in ch_idxs:
            ch = out[..., idx]
            if mode == "equalize":
                out[..., idx] = _equalize_channel(ch)
            elif mode == "clahe":
                out[..., idx] = _clahe_channel(ch, clip_limit)
            elif mode == "stretch":
                out[..., idx] = _stretch_channel(ch)

    return out


# ── Histogram visualisation ───────────────────────────────────────────────────

def _draw_histogram(img: np.ndarray, vis_w: int, vis_h: int) -> np.ndarray:
    """
    Render an RGB histogram as a (vis_h, vis_w, 3) float32 image.
    Each channel is drawn in its own colour (R/G/B), overlaid on a dark background.
    """
    canvas = np.zeros((vis_h, vis_w, 3), dtype=np.float32)
    canvas[:] = 0.08   # dark background

    colours = [(0, 0.25, 1.0), (0, 1.0, 0.25), (1.0, 0.25, 0.1)]  # B-ish, G-ish, R-ish

    for c_idx, colour in enumerate(colours):
        ch = img[..., c_idx].ravel()
        hist, _ = np.histogram(ch, bins=vis_w, range=(0.0, 1.0))
        hist     = hist.astype(np.float32)
        max_val  = hist.max()
        if max_val > 0:
            hist /= max_val

        for x in range(vis_w):
            bar_h = int(round(hist[x] * (vis_h - 2)))
            if bar_h > 0:
                y0 = vis_h - 1 - bar_h
                y1 = vis_h - 1
                for c in range(3):
                    canvas[y0:y1, x, c] = np.maximum(canvas[y0:y1, x, c], colour[c])

    # Thin separator line at bottom
    canvas[-1, :, :] = 0.3
    return canvas


def _compute_stats(img: np.ndarray) -> str:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lines = []
    for name, ch in [("R", r), ("G", g), ("B", b), ("Lum", lum)]:
        lines.append(
            f"{name}: mean={ch.mean():.3f}  std={ch.std():.3f}  "
            f"min={ch.min():.3f}  max={ch.max():.3f}"
        )
    return " | ".join(lines)


# ── Node ──────────────────────────────────────────────────────────────────────

class ImageHistogramNode:
    """
    Analyses tonal distribution and applies optional histogram correction.

    Modes
    -----
    none      : pass-through — just compute the histogram visualisation and stats
    equalize  : global histogram equalisation (pure NumPy, no dependencies)
    clahe     : Contrast Limited Adaptive Histogram Equalisation via OpenCV
                (clip_limit controls contrast amplification; typically 1–4)
    stretch   : linear contrast stretch — remaps [0.5th, 99.5th] percentile to [0, 1]

    Channel
    -------
    rgb       : apply to all three channels independently
    r/g/b     : apply to one channel only
    luminance : convert to YCbCr, apply to Y only (preserves colour)

    Outputs
    -------
    IMAGE : processed image (or pass-through if mode=none)
    IMAGE : histogram visualisation — connect to a Preview Image node
    STRING: per-channel mean / std / min / max stats
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
                "mode": (
                    _CFG["modes"],
                    {
                        "default": d["mode"],
                        "tooltip": "none = analyse only. equalize / clahe / stretch adjust tones.",
                    },
                ),
                "channel": (
                    _CFG["channels"],
                    {
                        "default": d["channel"],
                        "tooltip": "Which channel(s) the correction is applied to.",
                    },
                ),
                "clip_limit": (
                    "FLOAT",
                    {
                        "default": d["clip_limit"],
                        "min": 0.5, "max": 40.0, "step": 0.5,
                        "tooltip": "CLAHE only — higher values allow more contrast amplification.",
                    },
                ),
                "vis_width": (
                    "INT",
                    {"default": d["vis_width"], "min": 128, "max": 2048, "step": 64,
                     "tooltip": "Width in pixels of the histogram visualisation image."},
                ),
                "vis_height": (
                    "INT",
                    {"default": d["vis_height"], "min": 64, "max": 1024, "step": 32},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def process_histogram(
        self,
        image: torch.Tensor,
        mode: str,
        channel: str,
        clip_limit: float,
        vis_width: int,
        vis_height: int,
    ):
        squeeze = image.ndim == 3
        if squeeze:
            image = image.unsqueeze(0)

        arr = image.float().numpy()          # (N, H, W, 3)
        N   = arr.shape[0]

        out_frames   = []
        hist_frames  = []
        stats_parts  = []

        for i in range(N):
            frame = arr[i]                   # (H, W, 3)
            processed = _apply_mode(frame, mode, channel, clip_limit)
            out_frames.append(processed)

            hist_img = _draw_histogram(frame, vis_width, vis_height)
            hist_frames.append(hist_img)

            if i == 0:    # stats from first frame only (perf)
                stats_parts.append(_compute_stats(frame))

        out_t  = torch.from_numpy(np.stack(out_frames,  axis=0).astype(np.float32))
        hist_t = torch.from_numpy(np.stack(hist_frames, axis=0).astype(np.float32))
        stats  = " || ".join(stats_parts)

        if squeeze:
            out_t  = out_t.squeeze(0)
            hist_t = hist_t[0]          # keep single histogram frame as (H, W, C)
            hist_t = hist_t.unsqueeze(0)  # re-wrap as (1, H, W, C) for ComfyUI Preview

        return (out_t, hist_t, stats)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageHistogramNode": ImageHistogramNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageHistogramNode": _CFG["display_name"],
}
