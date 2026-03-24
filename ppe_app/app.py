from flask import Flask, request, send_file
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
from detect_mod import run as run_model1
from detect import run as run_model2
import cv2
import os
import time
from threading import Event
import signal
import sys

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
detection_thread = None
cap = None  # Global cap for both models
stop_event = Event()
current_model = None

def run_detection(model):
    global detection_thread, stop_event, current_model, cap
    current_model = model
    stop_event.clear()
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Error: Could not open webcam for {model}")
        cap = None
        return
    print(f"Webcam opened successfully for {model}")
    try:
        if model == "model1":
            stats, missing = run_model1(
                weights="C:/Users/champ/yolov5/runs/train/ppe_detection_m_896_v112/weights/best.pt",
                source=cap,  # Pass VideoCapture object
                imgsz=(896, 896),
                conf_thres=0.25,
                device="0",
                socketio=socketio,
                is_running=lambda: not stop_event.is_set()
            )
        else:  # model2
            stats, missing = run_model2(
                weights="C:/Users/champ/yolov5/runs/train/exp6/weights/best.pt",
                source=cap,  # Pass VideoCapture object
                imgsz=(640, 640),
                conf_thres=0.25,
                device="0",
                socketio=socketio,
                is_running=lambda: not stop_event.is_set()
            )
        def convert_to_serializable(data):
            if isinstance(data, dict):
                return {k: convert_to_serializable(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_to_serializable(item) for item in data]
            elif hasattr(data, 'tolist'):
                return data.tolist()
            return data
        stats = convert_to_serializable(stats)
        missing = convert_to_serializable(missing)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print(f"Webcam released in run_detection for {model}")
        cap = None
        if stop_event.is_set():
            socketio.emit('detection_complete', {'stats': stats, 'missing': missing, 'model': model})

@app.route('/start_realtime', methods=['POST', 'OPTIONS'])
def start_realtime():
    if request.method == 'OPTIONS':
        return '', 200
    print("Received /start_realtime request")
    global detection_thread, stop_event, cap
    model = request.json.get('model', 'model1')
    if detection_thread is not None and detection_thread.is_alive():
        stop_event.set()
        detection_thread.join(timeout=2)
        if detection_thread.is_alive():
            print("Warning: Previous thread did not stop, forcing cleanup")
        detection_thread = None
    if cap is not None and cap.isOpened():
        cap.release()
        print("Previous webcam instance released before new run")
    cap = None
    stop_event.clear()
    detection_thread = threading.Thread(target=run_detection, args=(model,))
    detection_thread.start()
    return {"status": "started"}

@app.route('/stop', methods=['POST', 'OPTIONS'])
def stop_detection():
    if request.method == 'OPTIONS':
        return '', 200
    print("Received /stop request")
    global detection_thread, stop_event, cap
    stop_event.set()
    if detection_thread is not None:
        detection_thread.join(timeout=2)
        if detection_thread.is_alive():
            print("Warning: Thread did not stop within 2 seconds, forcing cleanup")
        else:
            print("Detection thread stopped successfully")
        detection_thread = None
    if cap is not None and cap.isOpened():
        cap.release()
        print("Webcam released in stop_detection")
    cap = None
    temp_cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    max_attempts = 5
    attempts = 0
    while temp_cap.isOpened() and attempts < max_attempts:
        temp_cap.release()
        time.sleep(0.5)
        temp_cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} to release webcam: still in use = {temp_cap.isOpened()}")
    if not temp_cap.isOpened():
        print("Webcam fully released after retries")
    else:
        print("Warning: Webcam may still be in use after max attempts")
    temp_cap.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    return {"status": "Detection stopped"}

@app.route('/start_video', methods=['POST', 'OPTIONS'])
def start_video():
    if request.method == 'OPTIONS':
        return '', 200
    video_path = request.json.get('video_path')
    if video_path:
        stats, missing = run_model1(
            weights="C:/Users/champ/yolov5/runs/train/ppe_detection_m_896_v112/weights/best.pt",
            source=video_path,  # Keep as string for video files
            imgsz=(896, 896),
            conf_thres=0.25,
            device="0",
            socketio=socketio
        )
        return {"status": "completed", "stats": stats, "missing": missing}
    return {"status": "error", "message": "No video path provided"}

@app.route('/val_image/<filename>', methods=['GET'])
def get_val_image(filename):
    global current_model
    val_dir = (
        "C:/Users/champ/yolov5/runs/val/exp5" if current_model == "model1"
        else "C:/Users/champ/yolov5/runs/train/exp6"
    )
    file_path = os.path.join(val_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return {"error": "File not found"}, 404

def shutdown_handler(signum, frame):
    global detection_thread, stop_event, cap
    print("Shutting down...")
    stop_event.set()
    if detection_thread is not None:
        detection_thread.join(timeout=2)
        if detection_thread.is_alive():
            print("Warning: Thread did not stop during shutdown")
        detection_thread = None
    if cap is not None and cap.isOpened():
        cap.release()
        print("Webcam released during shutdown")
    cap = None
    cv2.destroyAllWindows()
    time.sleep(1)
    socketio.stop()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)