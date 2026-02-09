# Runpod Serverless Dockerfile for Wan2.2-I2V
# YouTube Shorts Image-to-Video Generator

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Hugging Face token will be passed as build arg
ARG HF_TOKEN

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/workspace/models \
    PYTHONWARNINGS="ignore::FutureWarning"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory with model cache
WORKDIR /workspace

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from GitHub main branch (Wan2.2 requires latest features)
RUN pip3 install git+https://github.com/huggingface/diffusers

# Install core dependencies
RUN pip3 install \
    transformers \
    accelerate \
    sentencepiece \
    huggingface-hub \
    safetensors \
    pillow \
    numpy \
    opencv-python \
    imageio-ffmpeg

# Login to Hugging Face (required for gated models)
RUN huggingface-cli login --token ${HF_TOKEN}

# Install Runpod serverless SDK
RUN pip3 install runpod

# Copy handler
COPY handler.py /workspace/handler.py

# Set the handler as the entrypoint
ENV HANDLER="handler.py"

# Runpod serverless will start the handler automatically
CMD ["python3", "-u", "handler.py"]
