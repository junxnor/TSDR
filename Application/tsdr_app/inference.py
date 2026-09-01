from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.tsdr_common import clamp_box, human_label, load_class_labels, resolve_weights
from . import config

detector = None
recognizer = None
labels = load_class_labels(config.CLASS_LABELS_PATH)
face_cascades = None

def get_models():
    global detector, recognizer
    if detector is None or recognizer is None:
        from ultralytics import YOLO

        detector = YOLO(str(resolve_weights(None, config.DETECTOR_WEIGHTS)))
        recognizer = YOLO(str(resolve_weights(None, config.RECOGNIZER_WEIGHTS)))
    return detector, recognizer

def image_to_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise RuntimeError("Could not encode result image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"

def draw_label(image: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.1, image.shape[1] / 1400))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y1 - th - baseline - 10)
    cv2.rectangle(image, (x1, y_top), (min(image.shape[1] - 1, x1 + tw + 10), y_top + th + baseline + 10), (17, 17, 17), -1)
    cv2.putText(image, text, (x1 + 5, y_top + th + 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def parse_preprocessing_mode(value: object) -> str:
    mode = str(value or "clahe_gamma").strip().lower()
    return mode if mode in config.OPENCV_PREPROCESSING_MODES else "clahe_gamma"

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Improve local contrast while preserving colour better than global histogram equalisation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def apply_gamma_correction(image: np.ndarray) -> np.ndarray:
    """Adjust exposure automatically for dark or overly bright road images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 95:
        gamma = 1.35  # brighten low-light frames
    elif mean_brightness > 175:
        gamma = 0.75  # darken over-exposed frames
    else:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)

def preprocess_with_opencv(image: np.ndarray, mode: str) -> np.ndarray:
    """OpenCV preprocessing stage used before YOLO detection and crop recognition."""
    mode = parse_preprocessing_mode(mode)
    processed = image.copy()
    if mode in {"clahe", "clahe_gamma"}:
        processed = apply_clahe(processed)
    if mode in {"gamma", "clahe_gamma"}:
        processed = apply_gamma_correction(processed)
    return processed

def classify_crop_ensemble(
    cls_model: object,
    original_crop: np.ndarray,
    enhanced_crop: np.ndarray,
    imgsz: int,
) -> tuple[int, str, float, float]:
    """Average several safe crop views to reduce single-frame classification errors."""
    blurred = cv2.GaussianBlur(enhanced_crop, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced_crop, 1.45, blurred, -0.45, 0)
    variants = [original_crop, enhanced_crop, sharpened]
    cls_results = cls_model.predict(source=variants, imgsz=imgsz, verbose=False)

    probability_vectors: list[np.ndarray] = []
    names: dict[int, str] | list[str] | None = None
    for result in cls_results:
        if result.probs is None:
            continue
        probability_vectors.append(result.probs.data.detach().cpu().numpy())
        names = result.names
    if not probability_vectors or names is None:
        return -1, "unknown", 0.0, 0.0

    mean_probs = np.mean(np.stack(probability_vectors), axis=0)
    ranked = np.argsort(mean_probs)[::-1]
    class_id = int(ranked[0])
    confidence = float(mean_probs[class_id])
    second_confidence = float(mean_probs[int(ranked[1])]) if len(ranked) > 1 else 0.0
    class_name = names[class_id]
    return class_id, class_name, confidence, confidence - second_confidence

def get_face_cascades():
    """Return frontal and profile detectors used as a conservative false-positive guard."""
    global face_cascades
    if face_cascades is None:
        cascade_dir = Path(cv2.data.haarcascades)
        face_cascades = [
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml")),
        ]
    return [cascade for cascade in face_cascades if not cascade.empty()]

def crop_has_face(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(24, int(min(crop.shape[:2]) * 0.28))
    for cascade in get_face_cascades():
        faces = cascade.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=4, minSize=(min_side, min_side))
        if len(faces):
            return True
        # A mirrored pass catches profile faces looking in the opposite direction.
        faces = cascade.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.06, minNeighbors=4, minSize=(min_side, min_side))
        if len(faces):
            return True
    return False

def find_face_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find faces once on the complete frame, where facial context is most visible."""
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    min_side = max(32, int(min(image.shape[:2]) * 0.07))
    boxes: list[tuple[int, int, int, int]] = []
    for cascade in get_face_cascades():
        for x, y, w, h in cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_side, min_side)
        ):
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes

def overlaps_face(sign_box: tuple[int, int, int, int], faces: list[tuple[int, int, int, int]]) -> bool:
    sx1, sy1, sx2, sy2 = sign_box
    sign_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    for fx1, fy1, fx2, fy2 in faces:
        ix1, iy1 = max(sx1, fx1), max(sy1, fy1)
        ix2, iy2 = min(sx2, fx2), min(sy2, fy2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        if intersection / sign_area >= 0.22 or intersection / face_area >= 0.35:
            return True
    return False

def plausible_sign_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
    max_area_ratio: float,
) -> tuple[bool, str]:
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False, "invalid box"

    frame_area = max(1, image_width * image_height)
    area_ratio = (box_w * box_h) / frame_area
    aspect_ratio = box_w / max(1, box_h)
    min_side_ratio = min(box_w / max(1, image_width), box_h / max(1, image_height))

    if area_ratio > max_area_ratio:
        return False, "box too large for driving mode"
    if area_ratio < 0.00025 and min(image_width, image_height) >= 600:
        return False, "box too small"
    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        return False, "unlikely traffic-sign shape"
    if min_side_ratio > 0.72:
        return False, "crop fills too much of frame"
    return True, ""

def run_tsdr(
    pil_image: Image.Image,
    conf: float,
    min_cls_conf: float = 0.50,
    preprocessing_mode: str = "clahe_gamma",
    det_imgsz: int = 960,
    cls_imgsz: int = 320,
    max_box_area_ratio: float = 0.28,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    det_model, cls_model = get_models()
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    preprocessing_mode = parse_preprocessing_mode(preprocessing_mode)
    model_input = preprocess_with_opencv(bgr, preprocessing_mode)
    height, width = bgr.shape[:2]
    detections: list[dict[str, object]] = []
    face_boxes = find_face_boxes(model_input)

    det_results = det_model.predict(source=model_input, imgsz=det_imgsz, conf=conf, verbose=False)
    boxes = det_results[0].boxes
    if boxes is None:
        return bgr, detections

    for idx, box in enumerate(boxes):
        det_conf = float(box.conf[0]) if box.conf is not None else 0.0
        raw_box = tuple(box.xyxy[0].tolist())
        x1, y1, x2, y2 = clamp_box(raw_box, width, height, 0.12)
        crop_x1, crop_y1, crop_x2, crop_y2 = clamp_box(raw_box, width, height, 0.02)
        if x2 <= x1 or y2 <= y1:
            continue
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        plausible, reason = plausible_sign_box(x1, y1, x2, y2, width, height, max_box_area_ratio)
        if not plausible:
            continue
        if det_conf < 0.85 and overlaps_face((x1, y1, x2, y2), face_boxes):
            continue
        original_crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        enhanced_crop = model_input[crop_y1:crop_y2, crop_x1:crop_x2]
        if det_conf < 0.85 and crop_has_face(enhanced_crop):
            continue
        class_id, class_name, cls_conf, cls_margin = classify_crop_ensemble(
            cls_model,
            original_crop,
            enhanced_crop,
            cls_imgsz,
        )
        candidate_name = human_label(class_name, labels)
        recognition_accepted = cls_conf >= min_cls_conf and cls_margin >= 0.08
        low_confidence = not recognition_accepted or cls_conf < 0.82 or cls_margin < 0.15
        display_name = candidate_name if class_id >= 0 else "Uncertain traffic sign"
        shown_name = display_name if class_id >= 0 else "Traffic sign: uncertain"
        if low_confidence and class_id >= 0:
            shown_name = f"{display_name} review"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (17, 17, 17), 3)
        draw_label(bgr, shown_name, x1, y1)
        detections.append(
            {
                "detection_index": idx,
                "det_conf": det_conf,
                "class_id": class_id,
                "class_name": class_name,
                "display_name": display_name,
                "candidate_name": candidate_name,
                "cls_conf": cls_conf,
                "cls_margin": cls_margin,
                "recognition_accepted": recognition_accepted,
                "low_confidence": low_confidence,
                "opencv_preprocessing": config.OPENCV_PREPROCESSING_MODES[preprocessing_mode],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return bgr, detections
