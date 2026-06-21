"""
ComfyUI Checkpoint Merger Node
Merges two loaded MODEL checkpoints via weighted sum or add-difference,
producing a blended model without writing anything to disk.
Config: configs/checkpoint_merger_config.json
"""

import os
import json
import copy
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "checkpoint_merger_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Merge implementations ─────────────────────────────────────────────────────

def _merge_weighted_sum(
    sd_a: dict, sd_b: dict, ratio: float
) -> dict:
    """out[k] = A[k] * ratio + B[k] * (1 - ratio)"""
    merged = {}
    all_keys = set(sd_a) | set(sd_b)
    for k in all_keys:
        if k in sd_a and k in sd_b:
            ta = sd_a[k].float()
            tb = sd_b[k].float()
            merged[k] = ta * ratio + tb * (1.0 - ratio)
        elif k in sd_a:
            merged[k] = sd_a[k].clone()
        else:
            merged[k] = sd_b[k].clone()
    return merged


def _merge_add_difference(
    sd_a: dict, sd_b: dict, ratio: float
) -> dict:
    """out[k] = A[k] + (A[k] - B[k]) * ratio"""
    merged = {}
    all_keys = set(sd_a) | set(sd_b)
    for k in all_keys:
        if k in sd_a and k in sd_b:
            ta = sd_a[k].float()
            tb = sd_b[k].float()
            merged[k] = ta + (ta - tb) * ratio
        elif k in sd_a:
            merged[k] = sd_a[k].clone()
        else:
            merged[k] = sd_b[k].clone()
    return merged


_MERGE_FNS = {
    "weighted_sum":   _merge_weighted_sum,
    "add_difference": _merge_add_difference,
}

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


# ── Node ──────────────────────────────────────────────────────────────────────

class CheckpointMergerNode:
    """
    Merges two MODEL checkpoints in memory — no files written to disk.

    Merge methods
    -------------
    weighted_sum   : classic merge — A×ratio + B×(1−ratio).
                     ratio=1.0 → pure A, ratio=0.5 → equal mix, ratio=0.0 → pure B.
    add_difference : A + (A−B)×ratio — pushes A further from B or pulls it closer.
                     Useful for injecting/subtracting a style without losing A's identity.

    Inputs
    ------
    model_a      : base MODEL
    model_b      : MODEL to blend in
    ratio        : blending weight (0.0–1.0, role depends on method)
    merge_method : weighted_sum | add_difference
    precision    : output dtype — fp16 saves VRAM; fp32 preserves precision

    Outputs
    -------
    MODEL  : merged model patcher
    STRING : summary — method, ratio, key stats
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
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "ratio": (
                    "FLOAT",
                    {
                        "default": d["ratio"],
                        "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": (
                            "weighted_sum: 1.0 = pure A, 0.5 = equal mix, 0.0 = pure B. "
                            "add_difference: 0.0 = no change to A, 1.0 = full difference applied."
                        ),
                    },
                ),
                "merge_method": (
                    _CFG["merge_methods"],
                    {"default": d["merge_method"]},
                ),
                "precision": (
                    _CFG["precisions"],
                    {
                        "default": d["precision"],
                        "tooltip": "Output dtype. fp16 saves VRAM; fp32 keeps maximum precision.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def merge_checkpoints(
        self,
        model_a,
        model_b,
        ratio: float,
        merge_method: str,
        precision: str,
    ):
        sd_a = model_a.model.state_dict()
        sd_b = model_b.model.state_dict()

        merge_fn  = _MERGE_FNS[merge_method]
        merged_sd = merge_fn(sd_a, sd_b, ratio)

        # Cast to target dtype
        dtype      = _DTYPE_MAP[precision]
        merged_sd  = {
            k: (v.to(dtype) if v.is_floating_point() else v)
            for k, v in merged_sd.items()
        }

        # Load merged weights into a deep copy of model_a
        out = copy.deepcopy(model_a)
        missing, unexpected = out.model.load_state_dict(merged_sd, strict=False)

        n_merged   = len([k for k in merged_sd if k in sd_a and k in sd_b])
        n_only_a   = len([k for k in merged_sd if k in sd_a and k not in sd_b])
        n_only_b   = len([k for k in merged_sd if k not in sd_a and k in sd_b])

        info = (
            f"method={merge_method} | ratio={ratio:.2f} | precision={precision} | "
            f"merged={n_merged:,} keys | only_A={n_only_a} | only_B={n_only_b} | "
            f"missing={len(missing)} | unexpected={len(unexpected)}"
        )
        print(f"[CheckpointMergerNode] {info}")
        return (out, info)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "CheckpointMergerNode": CheckpointMergerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointMergerNode": _CFG["display_name"],
}
