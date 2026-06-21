"""
ComfyUI VAE Quantize Node
Converts a loaded VAE's weights to fp16, bf16, or fp32 to trade off
VRAM usage against decoding quality.
Config: configs/vae_quantize_config.json
"""

import os
import json
import copy
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "vae_quantize_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def _vram_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1_048_576

def _cast_model(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype)
    for buf in model.buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype)
    return model


# ── Node ──────────────────────────────────────────────────────────────────────

class VAEQuantizeNode:
    """
    Lowers (or raises) the numeric precision of a VAE to control VRAM and quality.

    Why VAE is handled separately from MODEL
    -----------------------------------------
    The VAE encoder/decoder is highly sensitive to quantisation:
    - INT8  causes severe colour and detail loss — intentionally unsupported.
    - fp16  saves ~50 % VRAM vs fp32 with minor artefacts on some models.
    - bf16  same size as fp16, often better for VAEs on Ampere+ GPUs.
    - fp32  use this to upcast a half-precision VAE when you see banding.

    Inputs
    ------
    vae       : ComfyUI VAE
    precision : fp16 | bf16 | fp32

    Outputs
    -------
    VAE    : precision-adjusted VAE (deep copy)
    STRING : summary string
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
                "vae": ("VAE",),
                "precision": (
                    _CFG["precisions"],
                    {
                        "default": d["precision"],
                        "tooltip": (
                            "fp16/bf16 halve VRAM. "
                            "fp32 upcasts a half-precision VAE to fix colour banding."
                        ),
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def quantize_vae(self, vae, precision: str):
        out  = copy.deepcopy(vae)
        # ComfyUI VAE wraps the model at .first_stage_model
        inner: torch.nn.Module = out.first_stage_model

        before_mb = _vram_mb(inner)
        dtype     = _DTYPE_MAP[precision]
        _cast_model(inner, dtype)
        after_mb  = _vram_mb(inner)

        info = (
            f"precision={precision} | "
            f"before={before_mb:.0f} MB | after={after_mb:.0f} MB | "
            f"saved≈{before_mb - after_mb:.0f} MB"
        )
        print(f"[VAEQuantizeNode] {info}")
        return (out, info)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VAEQuantizeNode": VAEQuantizeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEQuantizeNode": _CFG["display_name"],
}
