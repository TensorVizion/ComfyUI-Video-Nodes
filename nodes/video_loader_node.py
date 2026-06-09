"""
ComfyUI Video Loader Node
Loads a video file from disk and extracts frames as image tensors.
Config: configs/video_loader_config.json
"""

import os
import json
import numpy as np
import torch
import folder_paths

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_loader_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


class VideoLoaderNode:
    """
    Reads a video file and returns a batch of image tensors plus audio metadata.

    Outputs
    -------
    IMAGE   : (N, H, W, 3) float32 tensor, values in [0, 1]
    INT     : total frame count
    FLOAT   : detected frames-per-second
    STRING  : resolved absolute path of the loaded file
    """

    CATEGORY = _CFG["category"]
    FUNCTION  = _CFG["function"]
    RETURN_TYPES  = tuple(_CFG["return_types"])
    RETURN_NAMES  = tuple(_CFG["return_names"])

    # ── Class-level defaults pulled from config ──────────────────────────────
    _DEFAULTS = _CFG["defaults"]

    @classmethod
    def INPUT_TYPES(cls):
        supported_exts = _CFG["supported_extensions"]
        video_dir = folder_paths.get_input_directory()

        # Gather all supported video files from ComfyUI's input folder
        video_files = []
        if os.path.isdir(video_dir):
            for fname in sorted(os.listdir(video_dir)):
                if any(fname.lower().endswith(ext) for ext in supported_exts):
                    video_files.append(fname)

        if not video_files:
            video_files = ["<no videos found>"]

        return {
            "required": {
                "video": (video_files,),
                "start_frame": (
                    "INT",
                    {
                        "default": cls._DEFAULTS["start_frame"],
                        "min": 0,
                        "max": 99_999,
                        "step": 1,
                    },
                ),
                "max_frames": (
                    "INT",
                    {
                        "default": cls._DEFAULTS["max_frames"],
                        "min": 1,
                        "max": 10_000,
                        "step": 1,
                    },
                ),
                "frame_skip": (
                    "INT",
                    {
                        "default": cls._DEFAULTS["frame_skip"],
                        "min": 0,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Extract every N+1 frames (0 = every frame).",
                    },
                ),
            },
            "optional": {
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute path overrides the dropdown selection.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def load_video(self, video, start_frame, max_frames, frame_skip, custom_path=""):
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for VideoLoaderNode. "
                "Install it with:  pip install opencv-python-headless"
            ) from exc

        # Resolve file path
        if custom_path and os.path.isfile(custom_path):
            video_path = custom_path
        else:
            video_path = os.path.join(folder_paths.get_input_directory(), video)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"VideoLoaderNode: file not found → {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"VideoLoaderNode: cannot open → {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 24.0

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        collected = 0
        step = max(1, frame_skip + 1)

        while collected < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames according to frame_skip
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if (current_pos - start_frame - 1) % step != 0:
                continue

            # BGR → RGB, uint8 → float32 [0,1]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb.astype(np.float32) / 255.0)
            collected += 1

        cap.release()

        if not frames:
            raise RuntimeError("VideoLoaderNode: no frames were extracted.")

        tensor = torch.from_numpy(np.stack(frames, axis=0))  # (N,H,W,3)
        return (tensor, total_frames, float(fps), video_path)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoLoaderNode": VideoLoaderNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoaderNode": _CFG["display_name"],
}
