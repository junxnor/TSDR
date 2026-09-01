from time import time

import cv2
import numpy as np
from PIL import Image

from . import inference, storage
camera = None
last_live_record_time = 0.0
server_camera_seen_signs: set[str] = set()

def frame_to_jpeg(frame: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Could not encode camera frame.")
    return encoded.tobytes()

def camera_frames(conf: float, min_cls_conf: float, preprocessing_mode: str):
    global camera, last_live_record_time
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera unavailable", (130, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_to_jpeg(placeholder) + b"\r\n"
        return

    frame_index = 0
    annotated = None
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % 4 == 1 or annotated is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            annotated, detections = inference.run_tsdr(
                pil_image,
                conf,
                min_cls_conf=min_cls_conf,
                preprocessing_mode=preprocessing_mode,
                det_imgsz=640,
                cls_imgsz=224,
                max_box_area_ratio=0.22,
            )
            now = time()
            new_detections = [
                row for row in detections
                if str(row["class_name"]) not in server_camera_seen_signs
            ]
            if new_detections and now - last_live_record_time >= 2.0:
                storage.save_prediction("live_frame.jpg", annotated, new_detections, source_type="live")
                server_camera_seen_signs.update(
                    str(row["class_name"]) for row in new_detections
                )
                last_live_record_time = now
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_to_jpeg(annotated) + b"\r\n"

def stop() -> None:
    global camera
    if camera is not None:
        camera.release()
        camera = None
    server_camera_seen_signs.clear()