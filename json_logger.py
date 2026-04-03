# json_logger.py
import json, os
class JSONLogger:
    def __init__(self, out_dir='output'):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log = []
    def add_frame(self, frame_idx, detections):
        self.log.append({'frame': frame_idx, 'objects': detections})
    def save(self, fname='output/detections.json'):
        with open(fname, 'w') as f:
            json.dump(self.log, f, indent=2)
