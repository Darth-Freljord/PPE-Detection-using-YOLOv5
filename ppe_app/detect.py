import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
import cv2
import base64
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import check_requirements, print_args

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
    save_img = not nosave and not str(source).endswith(".txt")
    is_webcam = isinstance(source, cv2.VideoCapture)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
        # Dataloader
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
    else:
        webcam = str(source).isnumeric() or str(source).endswith(".streams") or str(source).lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        if webcam:
            view_img = check_imshow(warn=True) and view_img
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs
    stats = {'person': [], 'vest': [], 'helmet': []}
    missing = {'vest': 0, 'helmet': 0}

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        if not is_running():
            if vid_cap is not None and vid_cap.isOpened():
                print("Video capture not released in loop; will be handled by caller")
            for writer in vid_writer:
                if writer is not None:
                    writer.release()
                    print("Video writer released due to stop event")
            print("Loop exiting due to is_running = False")
            return stats, missing

        with dt[0]:
            # Convert BGR (HWC) to RGB (HWC) and then to CHW for PyTorch
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            im = torch.from_numpy(im).to(model.device)
            im = im.permute(2, 0, 1)  # HWC to CHW
            im = im.half() if model.fp16 else im.float()
            im /= 255  # Normalize to [0, 1]
            if len(im.shape) == 3:
                im = im[None]  # Add batch dimension

        with dt[1]:
            visualize = increment_path(save_dir / Path(path if path else "webcam").stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if is_webcam:
                p, im0 = "webcam", im0s
            else:
                p, im0 = path[i] if isinstance(path, list) else path, im0s[i].copy() if isinstance(im0s, list) else im0s.copy()
            p = Path(p)
            save_path = str(save_dir / p.name)
            s += "{:g}x{:g} ".format(*im.shape[2:])
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                    bbox = xyxy if isinstance(xyxy, list) else xyxy.tolist()
                    stats[names[c]].append({'conf': float(conf), 'bbox': bbox})
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # Emit cropped instance image
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped = im0[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', cropped)
                    cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('instance', {'class': names[c], 'image': cropped_base64, 'conf': float(conf)})

            im0 = annotator.result()
            # Add FPS
            total_time = sum(dt_i.dt for dt_i in dt)
            fps = 1 / total_time if total_time > 0 else 0
            cv2.putText(im0, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if socketio:
                _, buffer = cv2.imencode('.jpg', im0)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                # Convert stats to JSON-serializable format
                def convert_to_serializable(data):
                    if isinstance(data, dict):
                        return {k: convert_to_serializable(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [convert_to_serializable(item) for item in data]
                    elif hasattr(data, 'tolist'):  # For PyTorch tensors
                        return data.tolist()
                    return data
                serializable_stats = convert_to_serializable(stats)
                print("Emitting frame at:", time.time())
                socketio.emit('frame', {
                    'image': frame_base64,
                    'stats': serializable_stats,
                    'missing': missing,
                    'timestamp': float(time.time())
                })

            if save_img:
                if not is_webcam:
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w, h = im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    for writer in vid_writer:
        if writer is not None:
            writer.release()
    return stats, missing

def parse_opt():
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
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)