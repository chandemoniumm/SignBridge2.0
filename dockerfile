# GPU-ready base (PyTorch + CUDA 12.1 + cuDNN)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs: OpenCV, ffmpeg, GUI (Tk), video4linux tools (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg tk tcl v4l-utils \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code, data config, and model weights (runs folder)
COPY training.py test.py realtime.py data.yaml /app/
COPY runs /app/runs

# Default working directory
WORKDIR /app

# Default command (interactive shell)
CMD ["bash"]

