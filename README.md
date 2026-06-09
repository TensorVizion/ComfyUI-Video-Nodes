# comfyui_video_nodes

A collection of four ComfyUI custom nodes for end-to-end video workflows.

## Nodes

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| `VideoLoaderNode` | 🎬 Video Loader | Video/IO | Load a video file; extract frames as IMAGE tensor |
| `VideoFrameBlenderNode` | 🎞️ Video Frame Blender | Video/Processing | Interpolate new frames between existing ones |
| `VideoEffectsNode` | ✨ Video Effects | Video/Processing | Apply colour grade, blur, vignette, grain, speed ramp |
| `VideoSaverNode` | 💾 Video Saver | Video/IO | Encode frames back to MP4 / AVI / WebM |

---

## Installation

1. Copy the `comfyui_video_nodes` folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
└── custom_nodes/
    └── comfyui_video_nodes/   ← place it here
```

2. Install the required dependency (OpenCV is needed by the Loader and Saver):

```bash
pip install opencv-python-headless
```

3. Restart ComfyUI. The four nodes will appear under the **Video/** categories in the node browser.

---

## Node Details

### 🎬 Video Loader
- **Config:** `configs/video_loader_config.json`
- **Inputs:** video file (dropdown), start frame, max frames, frame skip, optional custom path
- **Outputs:** `IMAGE` tensor (N×H×W×3), total frame count, FPS, resolved path
- Scans ComfyUI's `input/` directory for supported video files (`.mp4 .avi .mov .mkv .webm .gif`).

### 🎞️ Video Frame Blender
- **Config:** `configs/video_frame_blender_config.json`
- **Inputs:** `IMAGE` frames, interpolation steps, blend mode (`linear` / `cubic` / `optical`), loop toggle
- **Outputs:** upsampled `IMAGE` tensor, new frame count
- Inserts *N* synthetic frames between every pair of source frames. `cubic` mode uses a smooth-step curve for more natural motion.

### ✨ Video Effects
- **Config:** `configs/video_effects_config.json`
- **Inputs:** `IMAGE` frames, brightness, contrast, saturation, blur radius, vignette strength, grain amount + seed, speed multiplier
- **Outputs:** processed `IMAGE` tensor, frame count
- All effects are pure NumPy — no extra dependencies. Set any parameter to its default / zero to bypass that stage.

### 💾 Video Saver
- **Config:** `configs/video_saver_config.json`
- **Inputs:** `IMAGE` frames, filename prefix, FPS, container format, CRF quality, add timestamp toggle, optional output directory override
- **Outputs:** saved file path, frames written count
- Writes via OpenCV. If `ffmpeg` is on the system PATH, CRF quality is applied via `libx264` / `libvpx` re-encode.

---

## Typical Workflow

```
[Video Loader] → [Video Effects] → [Video Frame Blender] → [Video Saver]
```

You can also branch the IMAGE output from any node into standard ComfyUI image-processing nodes (e.g. KSampler, VAE Encode) and then feed the result back into the saver.

---

## Configuration

Each node reads its parameters from a JSON file in the `configs/` folder at startup. You can change defaults (e.g. default FPS, max frames, blend modes list) by editing the JSON — no Python changes required.

```
comfyui_video_nodes/
├── __init__.py
├── README.md
├── nodes/
│   ├── __init__.py
│   ├── video_loader_node.py
│   ├── video_frame_blender_node.py
│   ├── video_effects_node.py
│   └── video_saver_node.py
└── configs/
    ├── video_loader_config.json
    ├── video_frame_blender_config.json
    ├── video_effects_config.json
    └── video_saver_config.json
```

---

## Dependencies

| Package | Required by | Install |
|---|---|---|
| `torch` / `numpy` | all nodes | bundled with ComfyUI |
| `opencv-python-headless` | Loader, Saver | `pip install opencv-python-headless` |
| `ffmpeg` (system) | Saver (optional CRF) | system package manager |
