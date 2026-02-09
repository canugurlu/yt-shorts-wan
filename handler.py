"""
Runpod Serverless Handler for Wan2.2-I2V
YouTube Shorts Image-to-Video Generator
VRAM Optimized for 24GB GPUs (A40, RTX 6000 Ada)
- Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Input: base64 encoded image (from FLUX output)
- Output: MP4 video (5 seconds @ 16fps for Shorts)
"""

import os
import base64
import io
import gc
import torch
import numpy as np
import runpod
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")

# Global pipeline variable (loaded once)
pipeline = None


def load_model():
    """Load Wan2.2-I2V pipeline - VRAM optimized for 24GB GPUs"""
    global pipeline

    print(f"Loading model: {MODEL_ID}")

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load model with optimizations
    pipeline = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # Move to CUDA
    pipeline = pipeline.to("cuda")

    # Enable memory optimizations
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    # No CPU offload - 24GB VRAM is sufficient

    print("Model loaded successfully with VRAM optimizations")


def clear_cache():
    """Clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def encode_video_to_base64(video_path):
    """Encode video file to base64 string"""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode("utf-8")


def calculate_video_dimensions(image, max_area=480 * 832):
    """
    Calculate video dimensions maintaining aspect ratio.
    Default optimized for 480P vertical video (Shorts format).
    """
    aspect_ratio = image.height / image.width

    # VAE scale factor and patch size for dimension alignment
    mod_value = 64  # 16 (VAE) * 4 (patch_size) = 64

    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

    return width, height


def generate_video(image, prompt="", negative_prompt="",
                   num_frames=81, guidance_scale=3.5,
                   num_inference_steps=40, seed=None,
                   fps=16):
    """
    Generate video using Wan2.2-I2V

    Args:
        image: PIL Image or base64 string
        prompt: Text prompt for video generation
        negative_prompt: Negative prompt (Wan2.2 supports this!)
        num_frames: Number of frames (81 = 5 seconds @ 16fps)
        guidance_scale: CFG scale (3.5-5.0 recommended)
        num_inference_steps: Denoising steps (30-50 recommended)
        seed: Random seed for reproducibility
        fps: Output video FPS

    Returns:
        dict with video_base64, fps, num_frames
    """
    global pipeline

    if pipeline is None:
        load_model()

    clear_cache()

    # Decode image if base64
    if isinstance(image, str):
        image = decode_base64_image(image)

    # Calculate dimensions
    width, height = calculate_video_dimensions(image)
    image = image.resize((width, height))

    print(f"Generating video: {width}x{height}, {num_frames} frames")

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Default negative prompt for quality (Wan2.2 supports negative prompts)
    default_negative = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
        "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
        "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走"
    )

    if negative_prompt:
        negative_prompt = f"{negative_prompt}, {default_negative}"
    else:
        negative_prompt = default_negative

    # Generate video
    output = pipeline(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    frames = output.frames[0]

    # Export to MP4
    video_path = "/tmp/output.mp4"
    export_to_video(frames, video_path, fps=fps)

    clear_cache()

    return {
        "video_base64": encode_video_to_base64(video_path),
        "fps": fps,
        "num_frames": len(frames),
        "width": width,
        "height": height,
    }


def handler(job):
    """
    Runpod serverless handler for Wan2.2-I2V

    Input:
    {
        "image": "base64_encoded_image_string",  // Required
        "prompt": "A cinematic beach scene...",  // Optional
        "negative_prompt": "blurry, low quality",  // Optional
        "num_frames": 81,  // Optional (default: 81 = 5s @ 16fps)
        "guidance_scale": 3.5,  // Optional (default: 3.5)
        "num_inference_steps": 40,  // Optional (default: 40)
        "seed": 42,  // Optional
        "fps": 16,  // Optional (default: 16)
        "max_area": 399360  // Optional (480*832 for Shorts)
    }

    Output:
    {
        "status": "success",
        "video": {
            "video_base64": "...",
            "fps": 16,
            "num_frames": 81,
            "width": 832,
            "height": 480
        },
        "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    }
    """
    # Load model if not loaded
    if pipeline is None:
        load_model()

    input_data = job.get("input", {})

    # Get required parameters
    image = input_data.get("image")
    if not image:
        return {
            "status": "error",
            "error": "Missing required parameter: image (base64 string)"
        }

    # Get optional parameters
    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "")
    num_frames = input_data.get("num_frames", 81)  # 5 seconds @ 16fps
    guidance_scale = input_data.get("guidance_scale", 3.5)
    num_inference_steps = input_data.get("num_inference_steps", 40)
    seed = input_data.get("seed", None)
    fps = input_data.get("fps", 16)
    max_area = input_data.get("max_area", 480 * 832)  # Default 480P vertical

    try:
        result = generate_video(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            fps=fps,
        )

        return {
            "status": "success",
            "video": result,
            "model": MODEL_ID,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Start the Runpod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
