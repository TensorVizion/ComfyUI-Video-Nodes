"""
ComfyUI Model Quantize Node
Converts a loaded diffusion model's weights to a lower-precision dtype,
reducing VRAM usage without reloading from disk.
Config: configs/model_quantize_config.json
"""

import os
import json
import copy
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "model_quantize_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dtype_from_str(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)


def _vram_mb(model: torch.nn.Module) -> float:
    """Estimate parameter memory in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1_048_576


def _apply_dtype(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """Cast all floating-point parameters to *dtype*, leave int/bool untouched."""
    for name, param in model.named_parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype)
    for name, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype)
    return model


def _apply_int8(model: torch.nn.Module, layers: str) -> torch.nn.Module:
    """
    Dynamic INT8 quantisation via torch.ao.quantization (falls back to
    torch.quantization for older PyTorch builds).
    Works best on CPU; GPU inference after INT8 quant is model-dependent.
    """
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore[no-redef]

    target = {torch.nn.Linear}
    if layers == "all":
        target |= {torch.nn.Conv2d}

    return quantize_dynamic(model, target, dtype=torch.qint8)


# ── Node ──────────────────────────────────────────────────────────────────────

class ModelQuantizeNode:
    """
    Lowers the numeric precision of a MODEL's weights to save VRAM.

    Precision options
    -----------------
    fp16  : float16 — halves VRAM vs fp32, GPU-compatible, most widely supported
    bf16  : bfloat16 — same size as fp16 but wider dynamic range; best on A-series GPUs
    int8  : dynamic quantisation of Linear (and optionally Conv2d) layers;
            reduces VRAM by ~75 % vs fp32 but may require CPU inference

    Inputs
    ------
    model     : any ComfyUI MODEL
    precision : target precision (fp16 | bf16 | int8)
    layers    : which layer types to quantise for int8 (linear_only | all)

    Outputs
    -------
    MODEL  : quantised model patcher (in-place modified deep copy)
    STRING : summary — precision applied, parameter count, estimated VRAM delta
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
                "model": ("MODEL",),
                "precision": (
                    _CFG["precisions"],
                    {
                        "default": d["precision"],
                        "tooltip": (
                            "fp16/bf16 halve VRAM and work on GPU. "
                            "int8 reduces VRAM ~75% but is best on CPU."
                        ),
                    },
                ),
                "layers": (
                    _CFG["layer_targets"],
                    {
                        "default": d["layers"],
                        "tooltip": "Only used when precision=int8. 'all' also quantises Conv2d.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def quantize_model(self, model, precision: str, layers: str):
        import copy

        out = copy.deepcopy(model)
        inner: torch.nn.Module = out.model

        before_mb = _vram_mb(inner)

        if precision == "int8":
            out.model = _apply_int8(inner, layers)
        else:
            dtype = _dtype_from_str(precision)
            _apply_dtype(inner, dtype)

        after_mb = _vram_mb(out.model)
        saved_mb  = before_mb - after_mb
        n_params  = sum(p.numel() for p in out.model.parameters())

        info = (
            f"precision={precision} | layers={layers if precision == 'int8' else 'all'} | "
            f"params={n_params:,} | "
            f"before={before_mb:.0f} MB | after={after_mb:.0f} MB | "
            f"saved≈{saved_mb:.0f} MB"
        )
        print(f"[ModelQuantizeNode] {info}")
        return (out, info)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "ModelQuantizeNode": ModelQuantizeNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelQuantizeNode": _CFG["display_name"],
}
