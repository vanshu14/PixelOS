# Mini-ADAS — One-day demo project
**What this is:** A minimal, runnable demo that performs:
- Object detection using an external YOLOv8/Ultralytics model (you provide the weights)
- Simple lane detection using Canny + Hough lines
- Simple distance estimation heuristic using bounding box height (approx)
- Logging detections to JSON

**What's included**
- `main.py` — main demo script (processes video or webcam)
- `detect.py` — wrapper for object detection (Ultralytics)
- `lane_detector.py` — simple lane detection utilities
- `depth_stub.py` — placeholder/depth helper (for future ZoeDepth/Marigold integration)
- `json_logger.py` — logging detected frames -> JSON
- `requirements.txt` — Python dependencies
- `example_config.json` — adjustable parameters (focal, real object heights)
- `README.md` — this file

**Quick start (assumes you have Python 3.10+):**
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare a YOLOv8 weights file, e.g. `yolov8n.pt` and note its path.
   You can also use `ultralytics` pretrained models by name (if available).
4. Run the demo on a sample video:
   ```bash
   python main.py --source sample_video.mp4 --yolo_weights /path/to/yolov8n.pt
   ```
   Or run webcam:
   ```bash
   python main.py --source 0 --yolo_weights /path/to/yolov8n.pt
   ```
5. Output:
   - Annotated video frames will be displayed in a window.
   - A JSON log `output/detections.json` will be produced with frame-wise detections.

**Notes & limitations**
- This project is intentionally minimal so it can be completed in one day.
- Depth estimation is approximate (bbox-height heuristic). For accurate depth, integrate ZoeDepth/Marigold and update `depth_stub.py`.
- 3D Blender visualization is not included (can be added later by exporting the JSON logs).

If you want, I can:
- Add a pre-filled sample video (requires you to upload it)
- Integrate an actual depth model (I can provide instructions)
- Create a Blender importer script

