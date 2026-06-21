"""
ComfyUI LoRA Merger Node
Stacks up to three LoRA adapters onto a MODEL + CLIP at individual strengths,
using ComfyUI's own LoRA loading engine for maximum compatibility.
Config: configs/lora_merger_config.json
"""

import os
import json

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_merger_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NONE_SENTINEL = "none"


def _lora_names() -> list:
    """Return sorted list of LoRA filenames from ComfyUI's loras folder."""
    try:
        import folder_paths
        names = folder_paths.get_filename_list("loras")
    except Exception:
        names = []
    return [_NONE_SENTINEL] + sorted(names)


def _apply_single_lora(model, clip, lora_name: str, model_weight: float, clip_weight: float):
    """Load one LoRA by name and apply it via ComfyUI's internal API."""
    if lora_name == _NONE_SENTINEL or not lora_name:
        return model, clip, None

    import folder_paths
    import comfy.utils
    import comfy.sd

    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        print(f"[LoRAMergerNode] WARNING — LoRA file not found: {lora_name}")
        return model, clip, None

    lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
    model_out, clip_out = comfy.sd.load_lora_for_models(
        model, clip, lora_sd, model_weight, clip_weight
    )
    return model_out, clip_out, lora_name


# ── Node ──────────────────────────────────────────────────────────────────────

class LoRAMergerNode:
    """
    Stacks up to three LoRA adapters onto a MODEL + CLIP pair.

    Each LoRA is applied sequentially at its own model and CLIP strength,
    equivalent to chaining three Load LoRA nodes but in a single step.

    LoRA 2 and LoRA 3 are optional — set their dropdown to 'none' to skip.
    The clip_weight slider applies to all three LoRAs.

    Inputs
    ------
    model         : BASE MODEL
    clip          : BASE CLIP
    lora_1        : primary LoRA (required — pick a file or 'none')
    lora_1_weight : UNet strength for LoRA 1
    lora_2        : second LoRA (optional)
    lora_2_weight : UNet strength for LoRA 2
    lora_3        : third LoRA (optional)
    lora_3_weight : UNet strength for LoRA 3
    clip_weight   : CLIP strength applied to all selected LoRAs

    Outputs
    -------
    MODEL  : patched model with all LoRAs applied
    CLIP   : patched CLIP
    STRING : summary of applied LoRAs and weights
    """

    CATEGORY     = _CFG["category"]
    FUNCTION     = _CFG["function"]
    RETURN_TYPES = tuple(_CFG["return_types"])
    RETURN_NAMES = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        d  = cls._DEFAULTS
        ls = _lora_names()
        w  = {"min": -2.0, "max": 2.0, "step": 0.01}
        return {
            "required": {
                "model": ("MODEL",),
                "clip":  ("CLIP",),
                "lora_1":        (ls,  {"default": _NONE_SENTINEL}),
                "lora_1_weight": ("FLOAT", {"default": d["lora_1_weight"], **w,
                                            "tooltip": "UNet strength for LoRA 1 (negative values subtract the LoRA)."}),
                "lora_2":        (ls,  {"default": _NONE_SENTINEL}),
                "lora_2_weight": ("FLOAT", {"default": d["lora_2_weight"], **w}),
                "lora_3":        (ls,  {"default": _NONE_SENTINEL}),
                "lora_3_weight": ("FLOAT", {"default": d["lora_3_weight"], **w}),
                "clip_weight":   ("FLOAT", {"default": d["clip_weight"],   **w,
                                            "tooltip": "CLIP strength applied to ALL selected LoRAs."}),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def apply_loras(
        self,
        model, clip,
        lora_1, lora_1_weight,
        lora_2, lora_2_weight,
        lora_3, lora_3_weight,
        clip_weight,
    ):
        applied = []
        m, c = model, clip

        for name, mw in [
            (lora_1, lora_1_weight),
            (lora_2, lora_2_weight),
            (lora_3, lora_3_weight),
        ]:
            m, c, applied_name = _apply_single_lora(m, c, name, mw, clip_weight)
            if applied_name:
                applied.append(f"{applied_name}@{mw:.2f}")

        if not applied:
            info = "No LoRAs selected — model and CLIP returned unchanged."
        else:
            info = (
                f"Applied {len(applied)} LoRA(s): {' + '.join(applied)} "
                f"| clip_weight={clip_weight:.2f}"
            )

        print(f"[LoRAMergerNode] {info}")
        return (m, c, info)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "LoRAMergerNode": LoRAMergerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAMergerNode": _CFG["display_name"],
}
