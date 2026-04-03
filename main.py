# main.py
# Run: python main.py --source 0 --yolo_weights yolov8n.pt
import argparse, time, os
import cv2
import json
from detect import Detector
from lane_detector import detect_lanes
from depth_stub import estimate_distance_by_bbox_height
from json_logger import JSONLogger

def draw_annotations(frame, detections, lanes, distances):
    for det, dist in zip(detections, distances):
        x1,y1,x2,y2 = map(int, det['xyxy'])
        conf = det['conf']
        cls = det['class']
        label = f"{cls}:{conf:.2f}"
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        if dist is not None:
            cv2.putText(frame, f"{dist:.1f}m", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
    # draw lanes
    for l in lanes:
        x1,y1,x2,y2 = l
        cv2.line(frame, (x1,y1),(x2,y2), (0,0,255), 3)
    return frame

def main(args):
    detector = Detector(weights=args.yolo_weights)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    logger = JSONLogger(out_dir='output')
    frame_idx = 0
    focal_px = args.focal
    real_car_h = args.car_h
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.predict(frame)
        # compute distances with bbox-height heuristic
        distances = []
        detections_for_log = []
        for d in detections:
            bbox = d['xyxy']
            dist = estimate_distance_by_bbox_height(bbox, frame.shape[0], focal_px, real_car_h)
            distances.append(dist)
            detections_for_log.append({
                'class': d['class'],
                'conf': d['conf'],
                'xyxy': d['xyxy'],
                'distance_m': dist
            })
        lanes = detect_lanes(frame)
        annotated = draw_annotations(frame, detections, lanes, distances)
        cv2.imshow('Mini-ADAS', annotated)
        logger.add_frame(frame_idx, detections_for_log)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    logger.save()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0', help='video source (0 for webcam or path to file)')
    p.add_argument('--yolo_weights', required=True, help='path to yolov8 weights (.pt)')
    p.add_argument('--focal', type=float, default=1600.0, help='focal length in pixels (approx)')
    p.add_argument('--car_h', type=float, default=1.5, help='assumed real car height in meters')
    args = p.parse_args()
    main(args)
