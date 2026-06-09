"""
ComfyUI Video Saver Node
Encodes a batch of image tensors into a video file and saves it to disk.
Config: configs/video_saver_config.json
"""

import os
import json
import datetime
import numpy as np
import torch
import folder_paths

# ── Load node config ──────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "video_saver_config.json")
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


class VideoSaverNode:
    """
    Writes an (N, H, W, 3) IMAGE tensor batch to a video file.

    Supports MP4 (H.264), AVI (XVID), and WebM (VP80) containers.
    Files are saved into ComfyUI's output directory with an optional
    timestamp suffix to prevent overwriting.

    Outputs
    -------
    STRING  : absolute path of the saved video file
    INT     : number of frames written
    """

    CATEGORY = _CFG["category"]
    FUNCTION  = _CFG["function"]
    RETURN_TYPES  = tuple(_CFG["return_types"])
    RETURN_NAMES  = tuple(_CFG["return_names"])
    OUTPUT_NODE   = True   # tells ComfyUI this node has a visible side-effect

    _DEFAULTS   = _CFG["defaults"]
    _CONTAINERS = _CFG["containers"]
    _CODECS     = _CFG["codecs"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": cls._DEFAULTS["filename_prefix"]},
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": cls._DEFAULTS["fps"],
                        "min": 1.0,
                        "max": 240.0,
                        "step": 0.5,
                    },
                ),
                "container": (list(cls._CONTAINERS.keys()),),
                "quality_crf": (
                    "INT",
                    {
                        "default": cls._DEFAULTS["quality_crf"],
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "CRF quality (0=lossless, 51=worst). H.264 default 18–28.",
                    },
                ),
                "add_timestamp": (
                    "BOOLEAN",
                    {
                        "default": cls._DEFAULTS["add_timestamp"],
                        "tooltip": "Append a datetime stamp to avoid overwriting previous saves.",
                    },
                ),
            },
            "optional": {
                "output_dir_override": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Leave blank to use ComfyUI's default output directory.",
                    },
                ),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    def save_video(
        self,
        frames: torch.Tensor,
        filename_prefix: str,
        fps: float,
        container: str,
        quality_crf: int,
        add_timestamp: bool,
        output_dir_override: str = "",
    ):
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for VideoSaverNode. "
                "Install with:  pip install opencv-python-headless"
            ) from exc

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"VideoSaverNode expects (N,H,W,3) tensor; got {frames.shape}"
            )

        # Resolve output directory
        out_dir = output_dir_override.strip() or folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)

        # Build filename
        ext = self._CONTAINERS[container]
        ts  = ("_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) if add_timestamp else ""
        filename  = f"{filename_prefix}{ts}{ext}"
        out_path  = os.path.join(out_dir, filename)

        # Pick codec
        fourcc_str = self._CODECS.get(container, "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        N, H, W, _ = frames.shape
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(
                f"VideoSaverNode: could not open VideoWriter for {out_path}. "
                "Check that the codec is installed."
            )

        arr = (frames.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        for i in range(N):
            bgr = arr[i, :, :, ::-1]  # RGB → BGR
            writer.write(bgr)

        writer.release()

        # Optionally re-encode with ffmpeg for proper CRF (OpenCV doesn't expose CRF)
        if quality_crf != self._DEFAULTS["quality_crf"] and container in ("MP4 (H.264)", "WebM (VP8)"):
            try:
                import subprocess  # noqa: PLC0415
                tmp_path = out_path + ".tmp" + ext
                os.rename(out_path, tmp_path)
                vcodec = "libx264" if container == "MP4 (H.264)" else "libvpx"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", tmp_path,
                        "-c:v", vcodec, "-crf", str(quality_crf),
                        out_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                os.remove(tmp_path)
            except (FileNotFoundError, Exception):
                # ffmpeg not available — keep the OpenCV-encoded file
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, out_path)

        return (out_path, N)


# ── Node registration ─────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "VideoSaverNode": VideoSaverNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSaverNode": _CFG["display_name"],
}
