from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mtsd_recognition_balanced"))
    parser.add_argument("--model", default="yolo26n-cls.pt", help="Use yolo26n-cls.pt, yolo26s-cls.pt, yolo11n-cls.pt, etc.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/classify"))
    parser.add_argument("--name", default="mtsd_yolo26n_recognition_balanced")
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Recognition dataset not found: {args.data}. Run prepare_recognition_dataset.py first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc
    batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
    project = args.project.resolve()
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
        pretrained=True,
        cos_lr=True,
    )
    print(results)

    best_weights = project / args.name / "weights" / "best.pt"

    if best_weights.exists() and (args.data / "test").exists():
        print(f"Validating recognition model on test split: {best_weights}")
        YOLO(str(best_weights)).val(data=str(args.data), split="test", imgsz=args.imgsz)

if __name__ == "__main__":
    main()