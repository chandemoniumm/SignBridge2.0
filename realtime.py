# realtime_isl.py
import os
import time
import threading
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# ========= CHANGE ME (paths / settings) =========
MODEL_PATH   = "/home/chandemonium/Downloads/Dataset/runs/isl_yolo_det2/weights/best.pt"
CAMERA_INDEX = 0          # 0 = default webcam
CONF_THRES   = 0.7     # detection confidence
IMG_SIZE     = 640        # 320 for faster, 640 for better
FLIP_FRAME   = True       # mirror effect for webcam
WINDOW_W, WINDOW_H = 1700, 720
# ===============================================

def pick_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class RealtimeYOLOApp(tk.Tk):
    def __init__(self, model_path: str):
        super().__init__()
        self.title("YOLOv8 ISL Realtime")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")

        # Load model
        self.model = YOLO(model_path)
        self.device = pick_device()

        # UI
        self.left = tk.Label(self)   # original
        self.left.pack(side=tk.LEFT, padx=8, pady=8)
        self.right = tk.Label(self)  # overlay
        self.right.pack(side=tk.RIGHT, padx=8, pady=8)

        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(btn_frame, text="Start", command=self.start).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(btn_frame, text="Stop",  command=self.stop).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.running = False
        self.cap = None
        self.thread = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"[ERR] Could not open camera index {CAMERA_INDEX}")
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.left.config(image="");  self.left.image = None
        self.right.config(image=""); self.right.image = None

    def on_close(self):
        self.stop()
        self.destroy()

    def loop(self):
        t_prev = time.time()
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                continue

            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)

            # Inference
            t0 = time.time()
            results = self.model.predict(
                frame,
                conf=CONF_THRES,
                imgsz=IMG_SIZE,
                device=self.device,
                verbose=False
            )
            inf_time = time.time() - t0

            # Draw overlay (works for detect or seg; for classify we handle below)
            overlay_bgr = results[0].plot()

            # If this is a classification model, add the top-1 label
            if getattr(results[0], "probs", None) is not None:
                top1 = results[0].probs.top1
                conf = float(results[0].probs.top1conf)
                name = results[0].names[top1]
                cv2.putText(
                    overlay_bgr,
                    f"{name} {conf:.2f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # FPS text (smoothing)
            t_now = time.time()
            fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            cv2.putText(
                overlay_bgr,
                f"FPS: {fps:.1f}   infer: {inf_time*1000:.0f} ms   dev: {self.device}",
                (10, 30 if getattr(results[0], "probs", None) is None else 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Convert to Tk images (RGB)
            orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            over_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            # Resize for the GUI panels
            left_img = ImageTk.PhotoImage(Image.fromarray(orig_rgb).resize((800, 600)))
            right_img = ImageTk.PhotoImage(Image.fromarray(over_rgb).resize((800, 600)))

            # Push to UI
            self.left.config(image=left_img);   self.left.image = left_img
            self.right.config(image=right_img); self.right.image = right_img

            # Keep Tk responsive
            self.update_idletasks()

        # cleanup
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    # tip: run inside your venv
    #   cd ~/Downloads/Dataset
    #   source venv/bin/activate
    #   python realtime_isl.py
    app = RealtimeYOLOApp(MODEL_PATH)
    app.mainloop()
