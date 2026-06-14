"""
ComfyUI Image Noise Node
Adds controllable noise (Gaussian, Salt-and-Pepper, or Perlin-style)
to an IMAGE tensor. Useful for augmentation and stylistic effects.
Config: configs/image_noise_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "image_noise_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Noise generators ──────────────────────────────────────────────────────────

def _gaussian_noise(arr: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(arr.shape).astype(np.float32) * amount
    return np.clip(arr + noise, 0.0, 1.0)


def _salt_and_pepper(arr: np.ndarray, amount: float, salt_ratio: float,
                     rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    mask = rng.random(arr.shape[:3])           # (N, H, W) — same across channels
    salt_mask   = mask < amount * salt_ratio
    pepper_mask = (mask >= amount * salt_ratio) & (mask < amount)
    out[salt_mask]   = 1.0
    out[pepper_mask] = 0.0
    return out


def _uniform_noise(arr: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    noise = (rng.random(arr.shape).astype(np.float32) - 0.5) * 2.0 * amount
    return np.clip(arr + noise, 0.0, 1.0)


def _fade_noise(arr: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simple value-noise approximation: smooth tiled low-frequency noise.
    Not true Perlin noise but creates a similar large-scale grain effect.
    """
    N, H, W, C = arr.shape
    scale = max(1, min(H, W) // 8)
    small_h = max(1, H // scale)
    small_w = max(1, W // scale)
    raw = rng.random((N, small_h, small_w, C)).astype(np.float32)

    # Upscale with bilinear interpolation
    try:
        import torch.nn.functional as F  # noqa: PLC0415
        t = torch.from_numpy(raw).permute(0, 3, 1, 2)
        t_up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        noise = t_up.permute(0, 2, 3, 1).numpy() - 0.5
    except Exception:
        # Fallback: tile the raw noise
        noise = np.repeat(np.repeat(raw, scale, axis=1), scale, axis=2)[:, :H, :W, :] - 0.5

    return np.clip(arr + noise * amount * 2.0, 0.0, 1.0)


_NOISE_FNS = {
    "gaussian":        _gaussian_noise,
    "salt_and_pepper": _salt_and_pepper,
    "uniform":         _uniform_noise,
    "fade":            _fade_noise,
}


class ImageNoiseNode:
    """
    Adds noise to a still image or video frame batch.

    Noise types
    -----------
    gaussian        : normally-distributed additive noise (classic film grain feel)
    salt_and_pepper : random white/black pixel speckle
    uniform         : flat random additive noise
    fade            : low-frequency smooth value noise (large-scale grain)

    Inputs
    ------
    image       : (N, H, W, C) or (H, W, C) float32 tensor
    noise_type  : gaussian | salt_and_pepper | uniform | fade
    amount      : noise intensity (0 = none, 1 = extreme)
    salt_ratio  : for salt_and_pepper — fraction of noise pixels that are white
    seed        : RNG seed (0 = random each run)
    monochrome  : if True, the same noise pattern is applied to all channels

    Outputs
    -------
    IMAGE : noisy tensor, clamped to [0, 1]
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d = cls._DEFAULTS
        noise_types = list(_NOISE_FNS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_type": (
                    noise_types,
                    {"default": d["noise_type"]},
                ),
                "amount": (
                    "FLOAT",
                    {"default": d["amount"], "min": 0.0, "max": 1.0, "step": 0.005,
                     "tooltip": "Noise intensity. 0 = no noise, 1 = maximum."},
                ),
                "salt_ratio": (
                    "FLOAT",
                    {"default": d["salt_ratio"], "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "Salt-and-pepper only: fraction of noise that is white (rest is black)."},
                ),
                "seed": (
                    "INT",
                    {"default": d["seed"], "min": 0, "max": 2**31 - 1, "step": 1,
                     "tooltip": "RNG seed for reproducibility. 0 = random each run."},
                ),
                "monochrome": (
                    "BOOLEAN",
                    {"default": d["monochrome"],
                     "tooltip": "Apply identical noise pattern to all colour channels (greyscale noise)."},
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def add_noise(
        self,
        image: torch.Tensor,
        noise_type: str,
        amount: float,
        salt_ratio: float,
        seed: int,
        monochrome: bool,
    ):
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        arr = image.numpy().astype(np.float32)
        rng_seed = seed if seed != 0 else None
        rng = np.random.default_rng(rng_seed)

        if monochrome and arr.shape[-1] > 1:
            # Generate noise on a single-channel copy, then broadcast
            mono = arr[..., :1]
            fn = _NOISE_FNS.get(noise_type, _gaussian_noise)
            if noise_type == "salt_and_pepper":
                noisy_mono = fn(mono, amount, salt_ratio, rng)
            else:
                noisy_mono = fn(mono, amount, rng)
            # Compute the delta and apply to all channels
            delta = noisy_mono - mono
            out = np.clip(arr + delta, 0.0, 1.0)
        else:
            fn = _NOISE_FNS.get(noise_type, _gaussian_noise)
            if noise_type == "salt_and_pepper":
                out = fn(arr, amount, salt_ratio, rng)
            else:
                out = fn(arr, amount, rng)

        result = torch.from_numpy(out)
        if squeeze:
            result = result.squeeze(0)

        return (result,)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ImageNoiseNode": ImageNoiseNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageNoiseNode": _CFG["display_name"],
}
