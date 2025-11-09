# test_images.py
from ultralytics import YOLO
import os

# ===== CHANGE THIS =====
MODEL_PATH = "/home/chandemonium/Downloads/Dataset/runs/isl_yolo_det/weights/best.pt"
TEST_DIR   = "/home/chandemonium/Downloads/Dataset/test/images"
SAVE_DIR   = "/home/chandemonium/Downloads/Dataset/test_results"
CONF_THRES = 0.5
IMG_SIZE   = 640
# ========================

# Load trained model
model = YOLO(MODEL_PATH)

# Make sure output folder exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Run inference on all test images
results = model.predict(
    source=TEST_DIR,
    conf=CONF_THRES,
    imgsz=IMG_SIZE,
    save=True,        # saves annotated images automatically
    project=SAVE_DIR, # where results are stored
    name='predictions',
    exist_ok=True     # avoids duplicate run folders
)

print(f"\n✅ Predictions saved in: {SAVE_DIR}/predictions")
print("Total images processed:", len(results))
