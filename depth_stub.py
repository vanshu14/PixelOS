# depth_stub.py
# Placeholder helpers for depth estimation. Replace with ZoeDepth/Marigold integration for accurate depth.
import numpy as np

def estimate_distance_by_bbox_height(bbox, frame_height, focal_length_px, real_height_m=1.5):
    '''
    Approximate distance (meters) using pinhole camera model:
        distance = (real_height * focal) / pixel_height
    bbox: [x1,y1,x2,y2]
    '''
    x1,y1,x2,y2 = bbox
    pixel_h = max(1, (y2 - y1))
    distance_m = (real_height_m * focal_length_px) / pixel_h
    return float(distance_m)

def dummy_depth_map(frame):
    # returns a fake depth map (placeholder)
    h,w = frame.shape[:2]
    return np.full((h,w), 10.0, dtype=float)  # 10 meters everywhere
