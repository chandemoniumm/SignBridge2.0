# SignBridge2.0
## Indian Sign Language Detection using YOLOv8

This project detects static hand gestures of the Indian Sign Language (A–Z and 0–9) using YOLOv8.
It includes:
- Training script (`training.py`)
- Testing script (`test_images.py`)
- Real-time detection app (`realtime_isl.py`)
- Dockerfile for easy sharing and deployment

---

## Setup for linux

```bash
git clone https://github.com/chandemoniumm/SignBridge2.0
cd sign-language-yolo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup for Windows

```bash
1) Clone
git clone https://github.com/chandemoniumm/SignBridge2.0
cd SignBridge2.0

2) Create & activate venv
py -3 -m venv venv
# If activation is blocked, run once:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\venv\Scripts\Activate.ps1
# (use: .\venv\Scripts\activate.bat in CMD)

3) Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
# (Optional GPU) Install CUDA-enabled PyTorch matching your driver, e.g.:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

