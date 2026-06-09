"""
comfyui_video_nodes — top-level __init__.py
Aggregates all node class and display-name mappings so ComfyUI can discover
the whole package by scanning this single file.
"""

from .nodes.video_loader_node    import NODE_CLASS_MAPPINGS as _LOADER_CLASSES,   NODE_DISPLAY_NAME_MAPPINGS as _LOADER_NAMES
from .nodes.video_frame_blender_node import NODE_CLASS_MAPPINGS as _BLENDER_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _BLENDER_NAMES
from .nodes.video_saver_node     import NODE_CLASS_MAPPINGS as _SAVER_CLASSES,    NODE_DISPLAY_NAME_MAPPINGS as _SAVER_NAMES
from .nodes.video_effects_node   import NODE_CLASS_MAPPINGS as _EFFECTS_CLASSES,  NODE_DISPLAY_NAME_MAPPINGS as _EFFECTS_NAMES

# ── Merge all mappings ────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    **_LOADER_CLASSES,
    **_BLENDER_CLASSES,
    **_SAVER_CLASSES,
    **_EFFECTS_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_LOADER_NAMES,
    **_BLENDER_NAMES,
    **_SAVER_NAMES,
    **_EFFECTS_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
