# lane.py - Classical lane detection (Canny + Hough)

import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines):
    if lines is None:
        return img

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < -0.5:  # left lane
            left_fit.append((slope, intercept))
        elif slope > 0.5:  # right lane
            right_fit.append((slope, intercept))

    y1 = img.shape[0]
    y2 = int(y1 * 0.6)

    line_img = img.copy()

    for fit in (left_fit, right_fit):
        if len(fit) > 0:
            slope, intercept = np.mean(fit, axis=0)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return line_img

def lane_overlay(frame):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    roi_vertices = np.array([[
        (0, height),
        (width//2 - 50, height//2),
        (width//2 + 50, height//2),
        (width, height)
    ]])

    roi = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(
        roi,
        rho=2,
        theta=np.pi/180,
        threshold=60,
        minLineLength=40,
        maxLineGap=100
    )

    return draw_lines(frame, lines)
