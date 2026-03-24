# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
Enhanced to identify missing PPE gear and display appropriate bounding boxes.
"""
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import time
import base64
import simpleaudio as sa
from flask_socketio import SocketIO

# Global variables for SocketIO and stats
socketio_instance = None
detection_stats = {'person': [], 'vest': [], 'helmet': [], 'mask': [], 'gloves': [], 'glasses': []}
missing_ppe_counts = {'vest': 0, 'helmet': 0, 'mask': 0, 'gloves': 0, 'glasses': 0}
last_beep_time = 0
import torch
import numpy as np
import cv2
import mediapipe as mp

FILE = Path(__file__).resolve()
YOLO_ROOT = Path("C:/Users/champ/yolov5")  # Point to your YOLOv5 installation
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))  # Insert at beginning to prioritize
ROOT = YOLO_ROOT

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# Define PPE gear regions based on pose landmarks
def get_ppe_regions(landmarks, img_shape):
    """
    Define regions for PPE gear based on pose landmarks
    Returns: dict of ppe_item: [x1, y1, x2, y2] coordinates
    """
    h, w = img_shape[:2]
    
    if not landmarks:
        return {}
    
    # Convert landmarks to pixel coordinates
    landmark_px = {}
    for landmark_id, landmark in enumerate(landmarks.landmark):
        lm_x, lm_y = int(landmark.x * w), int(landmark.y * h)
        landmark_px[landmark_id] = (lm_x, lm_y, landmark.visibility)
    
    ppe_regions = {}
    
    # Helmet region (head)
    if all(id in landmark_px for id in [0, 1, 2, 3, 4, 5, 6, 7, 8]):
        nose = landmark_px[0]
        left_eye = landmark_px[2]
        right_eye = landmark_px[5]
        left_ear = landmark_px[7]
        right_ear = landmark_px[8]
        
        if nose[2] > 0.5 or left_eye[2] > 0.5 or right_eye[2] > 0.5:
            # Get the top-most point of the head
            top_y = min(landmark_px[0][1], landmark_px[1][1], landmark_px[4][1]) - int(0.2 * h)
            top_y = max(0, top_y)
            
            # Get the left and right boundaries
            left_x = min(landmark_px[7][0], landmark_px[3][0]) - int(0.1 * w)
            right_x = max(landmark_px[8][0], landmark_px[6][0]) + int(0.1 * w)
            
            # Bottom of the helmet at the eyes
            bottom_y = int((landmark_px[2][1] + landmark_px[5][1]) / 2)
            
            ppe_regions['helmet'] = [left_x, top_y, right_x, bottom_y]
    
    # Vest region (torso)
    if all(id in landmark_px for id in [11, 12, 23, 24]):
        left_shoulder = landmark_px[11]
        right_shoulder = landmark_px[12]
        left_hip = landmark_px[23]
        right_hip = landmark_px[24]
        
        if all(lm[2] > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
            # Add padding to the vest region
            pad_x = int(0.3 * w)
            pad_y = int(0.05 * h)
            
            left_x = min(left_shoulder[0], left_hip[0]) - pad_x
            right_x = max(right_shoulder[0], right_hip[0]) + pad_x
            top_y = min(left_shoulder[1], right_shoulder[1]) - pad_y
            bottom_y = max(left_hip[1], right_hip[1]) + pad_y
            
            ppe_regions['vest'] = [left_x, top_y, right_x, bottom_y]
    
    # Mask region (face)
    if all(id in landmark_px for id in [0, 1, 4, 9, 10]):
        nose = landmark_px[0]
        left_eye = landmark_px[1]
        right_eye = landmark_px[4]
        mouth_left = landmark_px[9]
        mouth_right = landmark_px[10]
        
        if nose[2] > 0.5:
            # Create a region around the nose and mouth
            pad_x = int(0.1 * w)
            pad_y_top = int(0.01 * h)
            pad_y_bottom = int(0.02 * h)
            
            left_x = min(nose[0], mouth_left[0]) - pad_x
            right_x = max(nose[0], mouth_right[0]) + pad_x
            top_y = nose[1] - pad_y_top
            bottom_y = max(mouth_left[1], mouth_right[1]) + pad_y_bottom
            
            ppe_regions['mask'] = [left_x, top_y, right_x, bottom_y]
    
    # Gloves region (hands)
    if all(id in landmark_px for id in [15, 16, 17, 18, 19, 20]):
        left_wrist = landmark_px[15]
        right_wrist = landmark_px[16]
        left_pinky = landmark_px[17]
        right_pinky = landmark_px[18]
        left_index = landmark_px[19]
        right_index = landmark_px[20]
        
        # Left glove
        if left_wrist[2] > 0.5:
            pad = int(0.03 * w)
            left_x = min(left_wrist[0], left_pinky[0], left_index[0]) - pad
            right_x = max(left_wrist[0], left_pinky[0], left_index[0]) + pad
            top_y = min(left_wrist[1], left_pinky[1], left_index[1]) - pad
            bottom_y = max(left_wrist[1], left_pinky[1], left_index[1]) + pad
            
            ppe_regions['gloves_left'] = [left_x, top_y, right_x, bottom_y]
        
        # Right glove
        if right_wrist[2] > 0.5:
            pad = int(0.03 * w)
            left_x = min(right_wrist[0], right_pinky[0], right_index[0]) - pad
            right_x = max(right_wrist[0], right_pinky[0], right_index[0]) + pad
            top_y = min(right_wrist[1], right_pinky[1], right_index[1]) - pad
            bottom_y = max(right_wrist[1], right_pinky[1], right_index[1]) + pad
            
            ppe_regions['gloves_right'] = [left_x, top_y, right_x, bottom_y]
    
    # Glasses region (eyes)
    if all(id in landmark_px for id in [1, 2, 3, 4, 5, 6]):
        left_eye_inner = landmark_px[1]
        left_eye = landmark_px[2]
        left_eye_outer = landmark_px[3]
        right_eye_inner = landmark_px[4]
        right_eye = landmark_px[5]
        right_eye_outer = landmark_px[6]
        
        if left_eye[2] > 0.5 and right_eye[2] > 0.5:
            pad_x = int(0.08 * w)
            pad_y = int(0.02 * h)
            
            left_x = min(left_eye_inner[0], left_eye_outer[0]) - pad_x
            right_x = max(right_eye_inner[0], right_eye_outer[0]) + pad_x
            top_y = min(left_eye[1], right_eye[1]) - pad_y
            bottom_y = max(left_eye[1], right_eye[1]) + pad_y
            
            ppe_regions['glasses'] = [left_x, top_y, right_x, bottom_y]
    
    # Ensure all coordinates are within image bounds
    for item, coords in ppe_regions.items():
        ppe_regions[item] = [
            max(0, min(w-1, coords[0])),
            max(0, min(h-1, coords[1])),
            max(0, min(w-1, coords[2])),
            max(0, min(h-1, coords[3]))
        ]
    
    return ppe_regions

# Missing PPE color map - different colors for missing PPE
missing_colors = {
    'helmet': (0, 0, 255),     # Red (BGR format)
    'vest': (0, 0, 255),       # Red
    'mask': (0, 165, 255),     # Orange
    'gloves_left': (255, 0, 0), # Blue
    'gloves_right': (255, 0, 0), # Blue
    'glasses': (255, 0, 255)   # Purple
}

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    socketio=None,
    is_running=lambda: True
):
    global socketio_instance, detection_stats, missing_ppe_counts, last_beep_time
    socketio_instance = socketio
    detection_stats = {'person': [], 'helmet': [], 'vest': [], 'mask': [], 'gloves': [], 'glasses': []}
    missing_ppe_counts = {'helmet': 0, 'vest': 0, 'mask': 0, 'gloves': 0, 'glasses': 0}
    last_beep_time = 0

    save_img = not nosave and not str(source).endswith(".txt")
    is_file = Path(str(source)).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    is_webcam = isinstance(source, cv2.VideoCapture)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if is_webcam:
        if not source.isOpened():
            raise ValueError("Provided VideoCapture object is not opened")
        def video_capture_iterator(cap):
            while is_running():
                ret, frame = cap.read()
                if not ret:
                    break
                yield None, frame, frame, cap, ""
            if cap.isOpened():
                print("Video capture not released by script; will be handled by caller")
        dataset = video_capture_iterator(source)
    elif is_url and is_file:
        source = check_file(source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    else:
        webcam = str(source).isnumeric() or str(source).endswith(".streams") or (is_url and not is_file)
        screenshot = str(source).lower().startswith("screen")
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    class_name_to_idx = {name: idx for idx, name in enumerate(names)}
    ppe_regions_map = {
        class_name_to_idx.get('helmet', -1): 'helmet',
        class_name_to_idx.get('vest', -1): 'vest',
        class_name_to_idx.get('mask', -1): 'mask',
        class_name_to_idx.get('gloves', -1): ['gloves_left', 'gloves_right'],
        class_name_to_idx.get('glasses', -1): 'glasses',
    }

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        print(f"Checking is_running: {is_running()}")  # Debug log
        if not is_running():
            if vid_cap is not None and vid_cap.isOpened():
                print("Video capture not released in loop; will be handled by caller")
            for writer in vid_writer:
                if writer is not None:
                    writer.release()
                    print("Video writer released due to stop event")
            print("Loop exiting due to is_running = False")
            break

        with dt[0]:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            im = torch.from_numpy(im).to(model.device)
            im = im.permute(2, 0, 1)  # HWC to CHW
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize = increment_path(save_dir / Path(str(path or "webcam")).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if is_webcam:
                p, im0, frame = path, im0s.copy(), seen
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                s += "{:g}x{:g} ".format(*im.shape[2:])

            p = Path(str(p or "webcam"))
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if not is_webcam else f"_{frame}")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            person_counter = 0
            detected_ppe = {
                'person': False, 'helmet': False, 'vest': False,
                'mask': False, 'gloves': False, 'glasses': False
            }

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    if int(c) < len(names):
                        class_name = names[int(c)]
                        if class_name in detected_ppe:
                            detected_ppe[class_name] = True

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    if names[c] == 'person':
                        person_counter += 1
                        label = f"Person {person_counter}" if hide_conf else f"Person {person_counter} {conf:.2f}"
                        bbox = xyxy.tolist() if isinstance(xyxy, torch.Tensor) else xyxy
                        detection_stats['person'].append({'conf': float(conf), 'frame': seen, 'bbox': bbox})
                        x1, y1, x2, y2 = map(int, bbox)
                        cropped = im0[y1:y2, x1:x2]
                        _, buffer = cv2.imencode('.jpg', cropped)
                        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                        socketio_instance.emit('instance', {'class': 'person', 'image': cropped_base64, 'conf': float(conf)})
                    else:
                        label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                        bbox = xyxy.tolist() if isinstance(xyxy, torch.Tensor) else xyxy
                        detection_stats[names[c]].append({'conf': float(conf), 'frame': seen, 'bbox': bbox})
                        x1, y1, x2, y2 = map(int, bbox)
                        cropped = im0[y1:y2, x1:x2]
                        _, buffer = cv2.imencode('.jpg', cropped)
                        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                        socketio_instance.emit('instance', {'class': names[c], 'image': cropped_base64, 'conf': float(conf)})
                    if save_txt:
                        if save_format == 0:
                            coords = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            if detected_ppe['person']:
                im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(im0_rgb)
                if pose_results.pose_landmarks:
                    ppe_regions = get_ppe_regions(pose_results.pose_landmarks, im0.shape)
                    for ppe_item, region in ppe_regions.items():
                        is_detected = ppe_item in ['gloves_left', 'gloves_right'] and detected_ppe['gloves'] or detected_ppe.get(ppe_item, False)
                        if not is_detected:
                            x1, y1, x2, y2 = region
                            color = missing_colors.get(ppe_item, (0, 0, 255))
                            cv2.rectangle(im0, (x1, y1), (x2, y2), color, line_thickness)
                            label_text = f"No {ppe_item.replace('_left', '').replace('_right', '')}"
                            txt_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)[0]
                            cv2.rectangle(im0, (x1, y1), (x1 + txt_size[0], y1 - txt_size[1] - 4), color, -1)
                            cv2.putText(im0, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            missing_ppe_counts[ppe_item.replace('_left', '').replace('_right', '')] += 1

                all_missing = all(not detected_ppe[item] for item in ['helmet', 'vest', 'mask', 'gloves', 'glasses'])
                if all_missing and time.time() - last_beep_time > 5:
                    wave_obj = sa.WaveObject.from_wave_file("beep.wav")
                    wave_obj.play()
                    last_beep_time = time.time()

            im0 = annotator.result()
            total_time = sum(dt_i.dt for dt_i in dt)
            fps = 1 / total_time if total_time > 0 else 0
            cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if socketio_instance:
                def convert_tensor_to_list(data):
                    if isinstance(data, torch.Tensor):
                        return data.cpu().numpy().tolist()
                    elif isinstance(data, dict):
                        return {key: convert_tensor_to_list(value) for key, value in data.items()}
                    elif isinstance(data, list):
                        return [convert_tensor_to_list(value) for value in data]
                    return data
                _, buffer = cv2.imencode('.jpg', im0)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                print("Emitting frame at:", time.time())
                socketio_instance.emit('frame', {
                    'image': frame_base64,
                    'stats': convert_tensor_to_list(detection_stats),
                    'missing': convert_tensor_to_list(missing_ppe_counts),
                    'fps': float(fps),
                    'timestamp': float(time.time())
                })

            if save_img:
                if not is_webcam:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path[0] != save_path:
                            vid_path[0] = save_path
                            if isinstance(vid_writer[0], cv2.VideoWriter):
                                vid_writer[0].release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))
                            vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[0].write(im0)
                else:
                    if vid_path[0] != save_path:
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()
                        fps, w, h = 30, im0.shape[1], im0.shape[0]  # Default for webcam
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[0].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])

    return detection_stats, missing_ppe_counts
    
    
#def stop_detection():
#    global is_running
    is_running = False

def parse_opt():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format", type=int, default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Main function."""
    # Check dependencies
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
    # Install MediaPipe if not present
    try:
        import mediapipe
    except ImportError:
        print("Installing MediaPipe dependency for pose estimation...")
        os.system("pip install mediapipe")
        
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)