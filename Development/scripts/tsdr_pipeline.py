from __future__ import annotations

import argparse
import csv
from pathlib import Path
import cv2

from tsdr_common import clamp_box, human_label, iter_images, load_class_labels, resolve_weights

DEFAULT_DETECTOR = Path("Model Weights/detection_best.pt")
DEFAULT_RECOGNIZER = Path("Model Weights/recognition_best.pt")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--recognizer", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True, help="Image or folder of road images.")
    parser.add_argument("--output", type=Path, default=Path("runs/tsdr/predictions"))
    parser.add_argument("--det-imgsz", type=int, default=960)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.12)
    return parser.parse_args()

def draw_label(image, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(image, (x1, y_top), (x1 + tw + 8, y_top + th + baseline + 8), (0, 128, 255), -1)
    cv2.putText(image, text, (x1 + 4, y_top + th + 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def main() -> None:
    args = parse_args()
    detector_weights = resolve_weights(args.detector, DEFAULT_DETECTOR)
    recognizer_weights = resolve_weights(args.recognizer, DEFAULT_RECOGNIZER)
    image_paths = list(iter_images(args.source))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.source}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "tsdr_predictions.csv"
    detector = YOLO(str(detector_weights))
    recognizer = YOLO(str(recognizer_weights))
    labels = load_class_labels()
    rows: list[list[object]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        det_results = detector.predict(source=image, imgsz=args.det_imgsz, conf=args.conf, verbose=False)
        boxes = det_results[0].boxes
        if boxes is not None:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = clamp_box(tuple(box.xyxy[0].tolist()), width, height, args.padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                cls_results = recognizer.predict(source=crop, imgsz=args.cls_imgsz, verbose=False)
                probs = cls_results[0].probs
                if probs is None:
                    class_id = -1
                    class_name = "unknown"
                    cls_conf = 0.0
                else:
                    class_id = int(probs.top1)
                    class_name = cls_results[0].names[class_id]
                    cls_conf = float(probs.top1conf)
                display_name = human_label(class_name, labels)
                det_conf = float(box.conf[0]) if box.conf is not None else 0.0
                label = f"{display_name} {cls_conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)
                draw_label(image, label, x1, y1)
                rows.append([image_path.as_posix(), idx, det_conf, class_id, class_name, display_name, cls_conf, x1, y1, x2, y2])

        out_path = args.output / f"{image_path.stem}_tsdr{image_path.suffix}"
        cv2.imwrite(str(out_path), image)

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_image", "detection_index", "det_conf", "class_id", "class_name", "display_name", "cls_conf", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"Processed {len(image_paths)} images.")
    print(f"Annotated outputs: {args.output}")
    print(f"Prediction metadata: {metadata_path}")

if __name__ == "__main__":
    main()
