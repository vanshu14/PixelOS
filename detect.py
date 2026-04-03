# detect.py
# Lightweight wrapper that uses ultralytics YOLO model to run inference on frames.
from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, weights='yolov8n.pt', device='cpu'):
        # weights can be path or model name
        self.model = YOLO(weights)
    def predict(self, frame):
        # frame: BGR numpy array
        results = self.model.predict(frame, imgsz=640, device='cpu', verbose=False)
        # results is a list (one per input); we used a single image
        out = []
        if len(results) == 0:
            return out
        r = results[0]
        boxes = r.boxes
        # each box: xyxy, conf, cls
        for b in boxes:
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            out.append({
                'xyxy': xyxy,
                'conf': conf,
                'class': cls
            })
        return out
