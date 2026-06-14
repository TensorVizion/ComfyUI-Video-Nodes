# comfyui-video-nodes

A collection of ComfyUI custom nodes for end-to-end **video** and **image** workflows — load, process, grade, composite, and export without leaving ComfyUI.

**v0.2.0** adds 8 new image-processing nodes on top of the original 4 video nodes.

---

## All Nodes at a Glance

### 🎬 Video Nodes (v0.1.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| `VideoLoaderNode` | 🎬 Video Loader | Video/IO | Load a video file; extract frames as IMAGE tensor |
| `VideoFrameBlenderNode` | 🎞️ Video Frame Blender | Video/Processing | Interpolate new frames between existing ones |
| `VideoEffectsNode` | ✨ Video Effects | Video/Processing | Colour grade, blur, vignette, grain, speed ramp |
| `VideoSaverNode` | 💾 Video Saver | Video/IO | Encode frames back to MP4 / AVI / WebM |

### 🖼️ Image Nodes (v0.2.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| `ImageCropNode` | ✂️ Image Crop | Image/Transform | Crop a rectangular region from an image or batch |
| `ImageResizeNode` | 📐 Image Resize | Image/Transform | Resize to explicit dimensions with 4 interpolation modes |
| `ImageFlipNode` | 🔁 Image Flip | Image/Transform | Flip horizontally, vertically, or both |
| `ImageRotateNode` | 🔄 Image Rotate | Image/Transform | Rotate by arbitrary angle with optional canvas expand |
| `ImageColourGradeNode` | 🎨 Image Colour Grade | Image/Colour | Brightness, contrast, saturation, hue, gamma, temperature |
| `ImageBlendNode` | 🖼️ Image Blend | Image/Compositing | Blend two images with 11 Photoshop-style blend modes |
| `ImageSharpenNode` | 🔍 Image Sharpen | Image/Filters | Unsharp mask sharpening (or softening) with threshold |
| `ImageNoiseNode` | 📡 Image Noise | Image/Filters | Add Gaussian, salt-and-pepper, uniform, or fade noise |
| `ImagePaddingNode` | 🖼 Image Padding | Image/Transform | Pad edges with constant fill, reflect, or replicate modes |

---

## Installation

1. Copy the `comfyui-video-nodes` folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
└── custom_nodes/
    └── comfyui-video-nodes/   ← place it here
```

2. Install dependencies:

```bash
pip install opencv-python-headless
```

3. Restart ComfyUI. All nodes will appear under the **Video/** and **Image/** categories in the node browser.

---

## Video Node Details

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
- All effects are pure NumPy — no extra dependencies. Set any parameter to its default/zero to bypass that stage.

### 💾 Video Saver
- **Config:** `configs/video_saver_config.json`
- **Inputs:** `IMAGE` frames, filename prefix, FPS, container format, CRF quality, add timestamp toggle, optional output directory override
- **Outputs:** saved file path, frames written count
- Writes via OpenCV. If `ffmpeg` is on the system PATH, CRF quality is applied via `libx264` / `libvpx` re-encode.

---

## Image Node Details

### ✂️ Image Crop
- **Config:** `configs/image_crop_config.json`
- **Inputs:** `IMAGE`, x, y, width, height
- **Outputs:** `IMAGE`, crop width (INT), crop height (INT)
- Crops a rectangular region from a single image or batch. x/y define the top-left corner. Values are automatically clamped to image bounds so no out-of-range error is raised.

### 📐 Image Resize
- **Config:** `configs/image_resize_config.json`
- **Inputs:** `IMAGE`, width, height, interpolation (`bilinear` / `nearest` / `bicubic` / `area`), keep_aspect, pad_value
- **Outputs:** `IMAGE`, output width (INT), output height (INT)
- Resizes to an explicit target resolution. When `keep_aspect` is enabled the image is scaled to fit inside the target box and remaining space is padded with `pad_value` — no distortion.

### 🔁 Image Flip
- **Config:** `configs/image_flip_config.json`
- **Inputs:** `IMAGE`, flip_h (bool), flip_v (bool)
- **Outputs:** `IMAGE`
- Lossless pixel rearrangement — no interpolation. Can flip horizontal, vertical, or both simultaneously. Works on single images and batches.

### 🔄 Image Rotate
- **Config:** `configs/image_rotate_config.json`
- **Inputs:** `IMAGE`, angle (°), expand (bool), fill_value
- **Outputs:** `IMAGE`, output width (INT), output height (INT)
- Rotates by an arbitrary angle using an affine warp (OpenCV). Positive = counter-clockwise. `expand=True` auto-enlarges the canvas so no pixels are clipped. Requires `opencv-python-headless`.

### 🎨 Image Colour Grade
- **Config:** `configs/image_colour_grade_config.json`
- **Inputs:** `IMAGE`, brightness, contrast, saturation, hue_shift, gamma, temperature
- **Outputs:** `IMAGE`
- Full colour-grading pipeline in one node. Processing order: temperature → brightness → contrast → gamma → hue → saturation. Set any parameter to its neutral value (`1.0` / `0.0`) to skip that stage. Pure NumPy, no extra dependencies.

| Parameter | Neutral | Range | Effect |
|---|---|---|---|
| brightness | 1.0 | 0 – 4 | Multiplicative exposure |
| contrast | 1.0 | 0 – 4 | Pivot at mid-grey |
| saturation | 1.0 | 0 – 4 | 0 = greyscale, 2 = vivid |
| hue_shift | 0.0 | −180 – 180 | Degrees of hue rotation |
| gamma | 1.0 | 0.1 – 5 | Power curve; <1 = bright mids |
| temperature | 0.0 | −1 – 1 | −1 = cool/blue, +1 = warm/orange |

### 🖼️ Image Blend
- **Config:** `configs/image_blend_config.json`
- **Inputs:** `image_a`, `image_b`, blend_mode, alpha
- **Outputs:** `IMAGE`
- Blends two images pixel-by-pixel using a choice of 11 Photoshop-style blend modes. `image_b` is auto-resized to match `image_a` if dimensions differ. `alpha` (0–1) controls how much of the blended result is mixed back with the original.

| Blend Modes | | | |
|---|---|---|---|
| normal | multiply | screen | overlay |
| soft_light | hard_light | difference | add |
| subtract | dodge | burn | |

### 🔍 Image Sharpen
- **Config:** `configs/image_sharpen_config.json`
- **Inputs:** `IMAGE`, strength, radius, sigma, threshold
- **Outputs:** `IMAGE`
- Unsharp masking: `output = image + strength × (image − blur(image))`. Negative `strength` values produce a blur/soften effect. `threshold` restricts sharpening to edge pixels above the given magnitude, reducing noise amplification in flat regions. Pure NumPy.

### 📡 Image Noise
- **Config:** `configs/image_noise_config.json`
- **Inputs:** `IMAGE`, noise_type, amount, salt_ratio, seed, monochrome
- **Outputs:** `IMAGE`
- Adds controllable noise to a still image or frame batch. Seed = 0 randomises each run; any other value is reproducible.

| Type | Description |
|---|---|
| `gaussian` | Normally-distributed additive noise — classic film grain |
| `salt_and_pepper` | Random black/white pixel speckle |
| `uniform` | Flat-distribution additive noise |
| `fade` | Low-frequency smooth value noise — large-scale grain |

### 🖼 Image Padding
- **Config:** `configs/image_padding_config.json`
- **Inputs:** `IMAGE`, pad_top, pad_bottom, pad_left, pad_right, mode, pad_value
- **Outputs:** `IMAGE`, output width (INT), output height (INT)
- Adds padding to any combination of edges. Falls back to `constant` mode automatically if the requested padding exceeds the image size in `reflect` or `replicate` mode.

| Mode | Description |
|---|---|
| `constant` | Solid fill colour (controlled by `pad_value`) |
| `reflect` | Mirror-reflects image pixels at the border |
| `replicate` | Repeats the outermost edge pixels outward |

---

## Typical Workflows

**Video pipeline**
```
[Video Loader] → [Video Effects] → [Video Frame Blender] → [Video Saver]
```

**Image processing pipeline**
```
[Image Crop] → [Image Resize] → [Image Colour Grade] → [Image Sharpen]
```

**Compositing pipeline**
```
[Image A] ──┐
             ├→ [Image Blend] → [Image Colour Grade] → [Video Saver]
[Image B] ──┘
```

**Mixed video + image pipeline**
```
[Video Loader] → [Image Colour Grade] → [Image Noise] → [Video Saver]
```

You can pass any `IMAGE` tensor (single frame or batch) between video and image nodes freely — they all use the same `(N, H, W, C)` float32 format that ComfyUI uses natively.

---

## Project Structure

```
comfyui-video-nodes/
├── __init__.py
├── README.md
├── pyproject.toml
├── nodes/
│   ├── __init__.py
│   │
│   ├── video_loader_node.py
│   ├── video_frame_blender_node.py
│   ├── video_effects_node.py
│   ├── video_saver_node.py
│   │
│   ├── image_crop_node.py
│   ├── image_resize_node.py
│   ├── image_flip_node.py
│   ├── image_rotate_node.py
│   ├── image_colour_grade_node.py
│   ├── image_blend_node.py
│   ├── image_sharpen_node.py
│   ├── image_noise_node.py
│   └── image_padding_node.py
│
└── configs/
    ├── video_loader_config.json
    ├── video_frame_blender_config.json
    ├── video_effects_config.json
    ├── video_saver_config.json
    │
    ├── image_crop_config.json
    ├── image_resize_config.json
    ├── image_flip_config.json
    ├── image_rotate_config.json
    ├── image_colour_grade_config.json
    ├── image_blend_config.json
    ├── image_sharpen_config.json
    ├── image_noise_config.json
    └── image_padding_config.json
```

---

## Configuration

Every node reads its defaults from a JSON file in `configs/` at startup. You can change defaults (e.g. default resolution, blend modes list, noise amount) by editing the JSON — no Python edits required. Changes take effect on next ComfyUI restart.

---

## Dependencies

| Package | Required by | Install |
|---|---|---|
| `torch` / `numpy` | all nodes | bundled with ComfyUI |
| `opencv-python-headless` | Video Loader, Video Saver, Image Rotate | `pip install opencv-python-headless` |
| `ffmpeg` (system binary) | Video Saver (optional CRF quality) | system package manager |

---

## Changelog

### v0.2.0
- Added 8 new image processing nodes: Crop, Resize, Flip, Rotate, Colour Grade, Blend, Sharpen, Noise, Padding
- Updated `__init__.py` to register all image nodes alongside existing video nodes
- Bumped version in `pyproject.toml`

### v0.1.0
- Initial release with 4 video nodes: Loader, Frame Blender, Effects, Saver
