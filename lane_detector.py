# lane_detector.py
# Simple lane detection using Canny edge detection + HoughLinesP.
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    vertices = np.array(vertices, dtype=np.int32)  # ✅ ensure int32
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ✅ ROI coordinates cast to int
    vertices = np.array([
        [0, h],
        [w, h],
        [int(w * 0.6), int(h * 0.6)],
        [int(w * 0.4), int(h * 0.6)]
    ], dtype=np.int32)

    cropped = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(
        cropped, rho=1, theta=np.pi/180, 
        threshold=50, minLineLength=50, maxLineGap=150
    )

    output_lines = []
    if lines is None:
        return output_lines

    for l in lines:
        x1, y1, x2, y2 = l[0]
        output_lines.append((int(x1), int(y1), int(x2), int(y2)))

    return output_lines
