"""Crop traffic signs detected by the trained detector from unseen images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from tsdr_common import clamp_box, iter_images, resolve_weights, safe_stem


DEFAULT_DETECTOR = Path("Model Weights/detection_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True, help="Image or folder of unseen road images.")
    parser.add_argument("--output", type=Path, default=Path("data/detected_sign_crops"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.force:
        import shutil

        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "detections.csv"

    weights = resolve_weights(args.weights, DEFAULT_DETECTOR)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc

    image_paths = list(iter_images(args.source))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.source}")

    detector = YOLO(str(weights))
    rows: list[list[object]] = []
    crop_count = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            results = detector.predict(source=rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)
            boxes = results[0].boxes
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = clamp_box(tuple(box.xyxy[0].tolist()), width, height, args.padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = rgb.crop((x1, y1, x2, y2))
                crop_name = f"{safe_stem(image_path)}_det{idx:03d}.jpg"
                crop_path = crop_dir / crop_name
                crop.save(crop_path, quality=95)
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                rows.append([image_path.as_posix(), crop_path.as_posix(), idx, conf, x1, y1, x2, y2])
                crop_count += 1

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_image", "crop_path", "detection_index", "det_conf", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"Cropped {crop_count} detected signs from {len(image_paths)} images.")
    print(f"Crops: {crop_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
