import cv2, threading, time, os
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
from lane import lane_overlay   # your lane code

app = Flask(__name__)

model = None
current_frame = None
running = False
selected_video = None  # video selected by user


def init_model(weights):
    global model
    if model is None:
        print("🔍 Loading YOLO...")
        model = YOLO(weights)
        print("✅ YOLO Loaded")


def process_loop(video, weights):
    global current_frame, running

    init_model(weights)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video}")
        running = False
        return

    print(f"🎬 Processing video: {video}")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video ended")
            break

        # YOLO detection
        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, results.names[cls], (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Lane detection
        frame = lane_overlay(frame)

        current_frame = frame

    cap.release()
    running = False
    print("🛑 Stopped processing")


def gen_frames():
    global current_frame
    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode('.jpg', current_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n')


@app.route("/")
def index():
    videos = [f for f in os.listdir("uploads") if f.lower().endswith(('.mp4','.avi','.mov'))]
    return render_template("index.html", videos=videos)


@app.route("/start", methods=["POST"])
def start():
    global running, selected_video

    selected_video = request.json.get("video")
    if not selected_video:
        return jsonify({"error": "No video selected"}), 400

    video_path = os.path.join("uploads", selected_video)

    if running:
        return jsonify({"status": "already running"})

    running = True
    threading.Thread(target=process_loop, args=(video_path, "yolov8n.pt"), daemon=True).start()

    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stop")
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    print("✅ Server Ready. Put videos inside /uploads")
    app.run(host="0.0.0.0", port=5002)
