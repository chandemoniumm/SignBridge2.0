from ultralytics import YOLO
import os

DATA = "data.yaml"
PROJECT_DIR = "runs"  # where results go

print("data.yaml lives in:", os.path.dirname(DATA))
print("entries there:", os.listdir(os.path.dirname(DATA)))

# Use DETECTION weights (bbox labels)
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data=DATA,
    project=PROJECT_DIR,
    name="isl_yolo_det",
    epochs=50,
    batch=4,
    imgsz=640
)

# Load best and evaluate on val split from YAML
best_path = os.path.join(PROJECT_DIR, "isl_yolo_det", "weights", "best.pt")
m = YOLO(best_path)
metrics = m.val(data=DATA)  # uses 'val:' from yaml
print(metrics)
