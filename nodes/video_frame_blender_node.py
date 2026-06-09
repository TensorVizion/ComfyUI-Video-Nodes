"""
ComfyUI Video Frame Blender Node
Interpolates new frames between existing ones to increase apparent frame rate
or create smooth slow-motion effects.
Config: configs/video_frame_blender_config.json
"""

import os
import json
import numpy as np
import torch

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_frame_blender_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


class VideoFrameBlenderNode:
    """
    Inserts blended (interpolated) frames between each pair of input frames,
    multiplying the effective frame count by *interpolation_steps + 1*.

    Blend modes
    -----------
    linear  : simple linear mix  α·A + (1-α)·B
    cubic   : smooth-step curve  (3t² − 2t³)
    optical : placeholder for future optical-flow integration
    """

    CATEGORY = _CFG["category"]
    FUNCTION  = _CFG["function"]
    RETURN_TYPES  = tuple(_CFG["return_types"])
    RETURN_NAMES  = tuple(_CFG["return_names"])

    _DEFAULTS = _CFG["defaults"]
    _BLEND_MODES = _CFG["blend_modes"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "interpolation_steps": (
                    "INT",
                    {
                        "default": cls._DEFAULTS["interpolation_steps"],
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Number of new frames to insert between each pair.",
                    },
                ),
                "blend_mode": (cls._BLEND_MODES,),
                "loop": (
                    "BOOLEAN",
                    {
                        "default": cls._DEFAULTS["loop"],
                        "tooltip": "Interpolate between the last and first frame to create a seamless loop.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _smoothstep(t: float) -> float:
        """Cubic smooth-step: 3t² − 2t³"""
        return t * t * (3.0 - 2.0 * t)

    def blend_frames(self, frames: torch.Tensor, interpolation_steps: int, blend_mode: str, loop: bool):
        """
        Parameters
        ----------
        frames : (N, H, W, C) float32 IMAGE tensor
        """
        if frames.ndim != 4:
            raise ValueError(f"VideoFrameBlenderNode expects 4-D tensor, got shape {frames.shape}")

        N = frames.shape[0]
        if N < 2:
            # Nothing to interpolate
            return (frames, N)

        arr = frames.numpy()  # work in numpy for speed

        # Build list of source frame pairs (optionally wrapping around)
        pairs = list(zip(range(N - 1), range(1, N)))
        if loop:
            pairs.append((N - 1, 0))

        out_frames = []
        steps = interpolation_steps + 1  # total segments per gap

        for idx_a, idx_b in pairs:
            fa = arr[idx_a]
            fb = arr[idx_b]
            out_frames.append(fa)  # always include the source frame

            for s in range(1, steps):
                t = s / steps
                if blend_mode == "cubic":
                    t = self._smoothstep(t)
                elif blend_mode == "optical":
                    # Optical-flow interpolation would go here;
                    # fall back to linear until a flow estimator is wired in.
                    t = t

                blended = (1.0 - t) * fa + t * fb
                out_frames.append(blended.astype(np.float32))

        # Append final frame (not added inside the loop)
        if not loop:
            out_frames.append(arr[-1])

        result = torch.from_numpy(np.stack(out_frames, axis=0))
        return (result, result.shape[0])


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoFrameBlenderNode": VideoFrameBlenderNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameBlenderNode": _CFG["display_name"],
}
