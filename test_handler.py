"""
Test script for Wan2.2-I2V handler
Tests the handler logic locally without actual inference
"""

import base64
import io
from PIL import Image


def create_test_image_base64(width=832, height=1536):
    """Create a simple test image and return as base64"""
    # Create a simple gradient image
    img = Image.new('RGB', (width, height), color='skyblue')
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            # Create a simple gradient
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels[x, y] = (r, g, b)

    # Convert to base64
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    return base64.b64encode(img_bytes).decode('utf-8')


def test_handler_input():
    """Generate a test input for the handler"""

    test_image = create_test_image_base64()

    test_input = {
        "input": {
            "image": test_image,
            "prompt": "Gentle camera movement, cinematic lighting, smooth motion",
            "negative_prompt": "blurry, shaky, fast movement",
            "num_frames": 81,
            "guidance_scale": 3.5,
            "num_inference_steps": 40,
            "seed": 42,
            "fps": 16,
            "max_area": 480 * 832  # For 480P vertical video
        }
    }

    return test_input


if __name__ == "__main__":
    import json

    # Create test input
    test_input = test_handler_input()

    print("=== Test Input for Wan2.2-I2V Handler ===")
    print(json.dumps(test_input, indent=2))

    # Save to file for testing
    with open("test_input.json", "w") as f:
        json.dump(test_input, f, indent=2)

    print("\nTest input saved to test_input.json")
    print("\nYou can test the handler with:")
    print("  python handler.py  # (when running in RunPod environment)")
