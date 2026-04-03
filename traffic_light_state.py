# traffic_light_state.py
# Placeholder simple color-based traffic light detector

import cv2
import numpy as np

def get_traffic_light_state(frame):
    """
    Returns: "Red", "Yellow", "Green", or "None"
    This is a dummy color-based detector (we will upgrade to ML soon)
    """

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color ranges for red, yellow, green lights
    red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_mask = red1 | red2

    yellow_mask = cv2.inRange(hsv, (15, 120, 120), (35, 255, 255))
    green_mask  = cv2.inRange(hsv, (36, 120, 120), (89, 255, 255))

    red_count = np.sum(red_mask)
    yellow_count = np.sum(yellow_mask)
    green_count = np.sum(green_mask)

    # Decision logic
    if red_count > yellow_count and red_count > green_count and red_count > 500:
        return "Red"
    elif yellow_count > red_count and yellow_count > green_count and yellow_count > 500:
        return "Yellow"
    elif green_count > red_count and green_count > yellow_count and green_count > 500:
        return "Green"
    else:
        return "None"
