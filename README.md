# comfyui-video-nodes

A collection of ComfyUI custom nodes for end-to-end video and image workflows — load, process, grade, composite, quantize, merge, and export without leaving ComfyUI.

v0.3.0 adds 7 new nodes covering model/VAE quantization, checkpoint & LoRA merging, video transitions, and histogram analysis on top of the original video and image processing suite.

## All Nodes at a Glance

### 🎬 Video Nodes (v0.1.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| VideoLoaderNode | 🎬 Video Loader | Video/IO | Load a video file; extract frames as IMAGE tensor |
| VideoFrameBlenderNode | 🎞️ Video Frame Blender | Video/Processing | Interpolate new frames between existing ones |
| VideoEffectsNode | ✨ Video Effects | Video/Processing | Colour grade, blur, vignette, grain, speed ramp |
| VideoSaverNode | 💾 Video Saver | Video/IO | Encode frames back to MP4 / AVI / WebM |

### 🖼️ Image Nodes (v0.2.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| ImageCropNode | ✂️ Image Crop | Image/Transform | Crop a rectangular region from an image or batch |
| ImageResizeNode | 📐 Image Resize | Image/Transform | Resize to explicit dimensions with 4 interpolation modes |
| ImageFlipNode | 🔁 Image Flip | Image/Transform | Flip horizontally, vertically, or both |
| ImageRotateNode | 🔄 Image Rotate | Image/Transform | Rotate by arbitrary angle with optional canvas expand |
| ImageColourGradeNode | 🎨 Image Colour Grade | Image/Colour | Brightness, contrast, saturation, hue, gamma, temperature |
| ImageBlendNode | 🖼️ Image Blend | Image/Compositing | Blend two images with 11 Photoshop-style blend modes |
| ImageSharpenNode | 🔍 Image Sharpen | Image/Filters | Unsharp mask sharpening (or softening) with threshold |
| ImageNoiseNode | 📡 Image Noise | Image/Filters | Add Gaussian, salt-and-pepper, uniform, or fade noise |
| ImagePaddingNode | 🖼 Image Padding | Image/Transform | Pad edges with constant fill, reflect, or replicate modes |

### ⚙️ Model & Quantization Nodes (v0.3.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| ModelQuantizeNode | 🗜️ Model Quantize | Model/Quantization | Quantize UNet/checkpoint to INT8, BF16, or FP16 via torch.quantization or bitsandbytes |
| VAEQuantizeNode | 🗜️ VAE Quantize | Model/Quantization | Precision-aware VAE quantization with separate sensitivity handling |
| CheckpointMergerNode | 🔀 Checkpoint Merger | Model/Merging | Weighted merge of two full SD checkpoints; supports add-difference mode |
| LoRAMergerNode | 🔗 LoRA Merger | Model/Merging | Combine multiple LoRA .safetensors into one at configurable weights |

### 🎥 Extended Video Nodes (v0.3.0)

| Node | Display Name | Category | Purpose |
|---|---|---|---|
| VideoTransitionNode | 🎭 Video Transition | Video/Transitions | Crossfade, wipe, and zoom-blur between two IMAGE batches |
| VideoConcatNode | 📎 Video Concat | Video/IO | Join multiple IMAGE batches end-to-end with optional crossfade |
| ImageHistogramNode | 📊 Image Histogram | Image/Analysis | Output histogram visualization tensor + auto-levels/equalization |

## Installation

Copy the `comfyui-video-nodes` folder into your ComfyUI `custom_nodes` directory:

ComfyUI/
└── custom_nodes/
└── comfyui-video-nodes/ ← place it here

Install dependencies:

```bash
pip install opencv-python-headless bitsandbytes

Restart ComfyUI. All nodes will appear under the Video/, Image/, Model/, and Video/Transitions categories in the node browser.
Video Node Details

🎬 Video Loader
Config: configs/video_loader_config.json
Inputs: video file (dropdown), start frame, max frames, frame skip, optional custom path
Outputs: IMAGE tensor (N×H×W×3), total frame count, FPS, resolved path
Scans ComfyUI's input/ directory for supported video files (.mp4 .avi .mov .mkv .webm .gif).

🎞️ Video Frame Blender
Config: configs/video_frame_blender_config.json
Inputs: IMAGE frames, interpolation steps, blend mode (linear / cubic / optical), loop toggle
Outputs: upsampled IMAGE tensor, new frame count
Inserts N synthetic frames between every pair of source frames. cubic mode uses a smooth-step curve for more natural motion.

✨ Video Effects
Config: configs/video_effects_config.json
Inputs: IMAGE frames, brightness, contrast, saturation, blur radius, vignette strength, grain amount + seed, speed multiplier
Outputs: processed IMAGE tensor, frame count
All effects are pure NumPy — no extra dependencies. Set any parameter to its default/zero to bypass that stage.

💾 Video Saver
Config: configs/video_saver_config.json
Inputs: IMAGE frames, filename prefix, FPS, container format, CRF quality, add timestamp toggle, optional output directory override
Outputs: saved file path, frames written count
Writes via OpenCV. If ffmpeg is on the system PATH, CRF quality is applied via libx264 / libvpx re-encode.

Image Node Details
✂️ Image Crop
Config: configs/image_crop_config.json
Inputs: IMAGE, x, y, width, height
Outputs: IMAGE, crop width (INT), crop height (INT)
Crops a rectangular region from a single image or batch. x/y define the top-left corner. Values are automatically clamped to image bounds so no out-of-range error is raised.

📐 Image Resize
Config: configs/image_resize_config.json
Inputs: IMAGE, width, height, interpolation (bilinear / nearest / bicubic / area), keep_aspect, pad_value
Outputs: IMAGE, output width (INT), output height (INT)
Resizes to an explicit target resolution. When keep_aspect is enabled the image is scaled to fit inside the target box and remaining space is padded with pad_value — no distortion.

🔁 Image Flip
Config: configs/image_flip_config.json
Inputs: IMAGE, flip_h (bool), flip_v (bool)
Outputs: IMAGE
Lossless pixel rearrangement — no interpolation. Can flip horizontal, vertical, or both simultaneously. Works on single images and batches.

🔄 Image Rotate
Config: configs/image_rotate_config.json
Inputs: IMAGE, angle (°), expand (bool), fill_value
Outputs: IMAGE, output width (INT), output height (INT)
Rotates by an arbitrary angle using an affine warp (OpenCV). Positive = counter-clockwise. expand=True auto-enlarges the canvas so no pixels are clipped. Requires opencv-python-headless.

🎨 Image Colour Grade
Config: configs/image_colour_grade_config.json
Inputs: IMAGE, brightness, contrast, saturation, hue_shift, gamma, temperature
Outputs: IMAGE

🖼️ Image Blend
Config: configs/image_blend_config.json
Inputs: image_a, image_b, blend_mode, alpha
Outputs: IMAGE
Blends two images pixel-by-pixel using a choice of 11 Photoshop-style blend modes. image_b is auto-resized to match image_a if dimensions differ. alpha (0–1) controls how much of the blended result is mixed back with the original
Blend Modes

🔍 Image Sharpen
Config: configs/image_sharpen_config.json
Inputs: IMAGE, strength, radius, sigma, threshold
Outputs: IMAGE
Unsharp masking: output = image + strength × (image − blur(image)). Negative strength values produce a blur/soften effect. threshold restricts sharpening to edge pixels above the given magnitude, reducing noise amplification in flat regions. Pure NumPy.

📡 Image Noise
Config: configs/image_noise_config.json
Inputs: IMAGE, noise_type, amount, salt_ratio, seed, monochrome
Outputs: IMAGE
Adds controllable noise to a still image or frame batch. Seed = 0 randomises each run; any other value is reproducible.

🖼 Image Padding
Config: configs/image_padding_config.json
Inputs: IMAGE, pad_top, pad_bottom, pad_left, pad_right, mode, pad_value
Outputs: IMAGE, output width (INT), output height (INT)
Adds padding to any combination of edges. Falls back to constant mode automatically if the requested padding exceeds the image size in reflect or replicate mode.

Model & Quantization Node Details (v0.3.0)

🗜️ Model Quantize
Config: configs/model_quantize_config.json
Inputs: MODEL, precision (INT8 / BF16 / FP16), backend (torch / bitsandbytes)
Outputs: quantized MODEL
Reduces checkpoint VRAM footprint by converting weights to lower precision. Uses torch.quantization for INT8/FP16 or bitsandbytes for NF4/INT8. Pairs naturally with custom checkpoints trained via Modal DreamBooth scripts — quantize after training to run larger models on limited VRAM without retraining.

🗜️ VAE Quantize
Config: configs/vae_quantize_config.json
Inputs: VAE, precision (BF16 / FP16), clamp_range
Outputs: quantized VAE
VAE-specific quantization with separate precision handling. VAEs are more sensitive to precision loss than UNets; this node applies gentler clamping and avoids INT8 for VAE decoding to prevent banding artefacts. Use alongside Model Quantize for maximum VRAM savings.

🔀 Checkpoint Merger
Config: configs/checkpoint_merger_config.json
Inputs: model_a, model_b, weight (0.0–1.0), merge_mode (weighted_sum / add_difference), save_dtype
Outputs: merged MODEL
Merges two full SD checkpoints with a weighted ratio: A × weight + B × (1 - weight). Supports add-difference mode for applying the delta between two models onto a third base. One of the most-requested features in the community — now available natively without external tools.

🔗 LoRA Merger
Config: configs/lora_merger_config.json
Inputs: lora_paths (list), weights (list), save_dtype, clamp_alpha
Outputs: merged LoRA .safetensors path
Combines multiple LoRA files into a single merged LoRA at configurable per-file weights. Designed to complement the Modal training workflow: train separate LoRAs for different concepts, then merge them into one portable adapter without leaving ComfyUI.

Extended Video Node Details (v0.3.0)

🎭 Video Transition
Config: configs/video_transition_config.json
Inputs: frames_a, frames_b, transition_type (crossfade / wipe_left / wipe_right / zoom_blur), duration_frames, easing
Outputs: transitioned IMAGE tensor
Creates clip-to-clip transitions between two IMAGE batches. Fills the gap left by Video Frame Blender (which handles intra-clip interpolation, not inter-clip transitions). Easing curves (linear / ease_in / ease_out) control transition pacing.

📎 Video Concat
Config: configs/video_concat_config.json
Inputs: frame_batches (list), crossfade_frames (optional)
Outputs: concatenated IMAGE tensor, total frame count
Joins multiple IMAGE batches end-to-end. Optional crossfade parameter blends the last N frames of each clip with the first N frames of the next for seamless joins. Simple but essential for multi-shot video pipelines.

📊 Image Histogram
Config: configs/image_histogram_config.json
Inputs: IMAGE, mode (visualize / auto_levels / equalize), clip_percentile
Outputs: IMAGE (processed or histogram visualization), histogram tensor
Outputs a histogram visualization tensor and/or applies automatic level correction. auto_levels stretches the tonal range based on percentile clipping; equalize flattens the distribution. Complements Image Colour Grade by providing objective tonal analysis before manual grading.

Typical Workflows
Video pipeline
[Video Loader] → [Video Effects] → [Video Frame Blender] → [Video Saver]

Image processing pipeline
[Image Crop] → [Image Resize] → [Image Colour Grade] → [Image Sharpen]

Compositing pipeline
[Image A] ──┐
             ├→ [Image Blend] → [Image Colour Grade] → [Video Saver]
[Image B] ──┘

Mixed video + image pipeline
[Video Loader] → [Image Colour Grade] → [Image Noise] → [Video Saver]

Multi-shot video with transitions
[Clip A] ──┐
            ├→ [Video Transition] → [Video Concat] → [Video Effects] → [Video Saver]
[Clip B] ──┘

VRAM-constrained generation
[Load Checkpoint] → [Model Quantize (INT8)] → [KSampler]
[Load VAE]        → [VAE Quantize (BF16)]  → [VAE Decode]

Train → Merge → Generate pipeline
[Modal Trained LoRA A] ──┐
                          ├→ [LoRA Merger] → [Apply LoRA] → [KSampler]
[Modal Trained LoRA B] ──┘

You can pass any IMAGE tensor (single frame or batch) between video and image nodes freely — they all use the same (N, H, W, C) float32 format that ComfyUI uses natively.

Project Structure
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
│   ├── image_padding_node.py
│   │
│   ├── model_quantize_node.py
│   ├── vae_quantize_node.py
│   ├── checkpoint_merger_node.py
│   ├── lora_merger_node.py
│   │
│   ├── video_transition_node.py
│   ├── video_concat_node.py
│   └── image_histogram_node.py
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
    ├── image_padding_config.json
    │
    ├── model_quantize_config.json
    ├── vae_quantize_config.json
    ├── checkpoint_merger_config.json
    ├── lora_merger_config.json
    │
    ├── video_transition_config.json
    ├── video_concat_config.json
    └── image_histogram_config.json

Configuration
Every node reads its defaults from a JSON file in configs/ at startup. You can change defaults (e.g. default resolution, blend modes list, noise amount, quantization backend) by editing the JSON — no Python edits required. Changes take effect on next ComfyUI restart.
Dependencies:
Torch
Numpy
Bits And Bytes
opencv-python-headless
FFMPEG
safetensors

v0.3.0
Added Model Quantize node — INT8/BF16/FP16 checkpoint quantization via torch.quantization and bitsandbytes
Added VAE Quantize node — precision-aware VAE quantization with separate sensitivity handling
Added Checkpoint Merger node — weighted sum and add-difference merging of full SD checkpoints
Added LoRA Merger node — combine multiple LoRA adapters at configurable weights
Added Video Transition node — crossfade, wipe, and zoom-blur between IMAGE batches
Added Video Concat node — join multiple IMAGE batches with optional crossfade
Added Image Histogram node — histogram visualization + auto-levels/equalization
Updated __init__.py to register all new nodes
Added config files for all new nodes
Updated dependencies to include bitsandbytes
Bumped version in pyproject.toml

v0.2.0
Added 8 new image processing nodes: Crop, Resize, Flip, Rotate, Colour Grade, Blend, Sharpen, Noise, Padding
Updated __init__.py to register all image nodes alongside existing video nodes
Bumped version in pyproject.toml

v0.1.0
Initial release with 4 video nodes: Loader, Frame Blender, Effects, Save
