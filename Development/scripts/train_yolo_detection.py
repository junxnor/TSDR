from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mtsd_yolo_detection/data.yaml"))
    parser.add_argument("--model", default="yolo26n.pt", help="Use yolo26n.pt, yolo26s.pt, yolo11n.pt, etc.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default="-1", help="Batch size. Use -1 for Ultralytics auto batch.")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, or leave empty for auto.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="mtsd_yolo26n_detection")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true", help="Cache images if you have enough RAM/disk.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}. Run prepare_yolo_detection.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc
    batch: int | float
    batch = float(args.batch) if "." in str(args.batch) else int(args.batch)

    project = Path(args.project).resolve()
    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        resume=args.resume,
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,
    )
    print(results)

    best_weights = project / args.name / "weights" / "best.pt"
    
    if best_weights.exists():
        print(f"Validating best weights: {best_weights}")
        YOLO(str(best_weights)).val(data=str(args.data), imgsz=args.imgsz, split="test")

if __name__ == "__main__":
    main()