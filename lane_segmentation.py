# lane_segmentation.py
# Simple lane segmentation using color thresholds + perspective transform (bird's-eye)
import cv2
import numpy as np

def get_perspective_transform(frame):
    h, w = frame.shape[:2]
    # source points (trapezoid) - tuned for forward-facing dashcam
    src = np.float32([
        [int(0.12*w), int(0.95*h)],
        [int(0.88*w), int(0.95*h)],
        [int(0.6*w), int(0.6*h)],
        [int(0.4*w), int(0.6*h)]
    ])
    dst = np.float32([
        [int(0.25*w), h],
        [int(0.75*w), h],
        [int(0.75*w), 0],
        [int(0.25*w), 0]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def lane_mask_from_birdeye(frame):
    """Return a binary mask of lane pixels (same size as frame)."""
    h, w = frame.shape[:2]
    M, Minv = get_perspective_transform(frame)
    bird = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)

    # thresholds for white lanes
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # thresholds for yellow lanes
    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white, mask_yellow)
    # morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # warp back to original perspective
    mask_warped = cv2.warpPerspective(mask, Minv, (w, h), flags=cv2.INTER_LINEAR)
    mask_warped = (mask_warped > 0).astype('uint8') * 255
    return mask_warped
