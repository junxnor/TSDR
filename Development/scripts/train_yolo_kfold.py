from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, default=Path("data/mtsd_yolo_detection/folds"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect_kfold")
    parser.add_argument("--name-prefix", default="mtsd_fold")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    yamls = sorted(args.fold_dir.glob("fold_*.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No fold YAML files found in {args.fold_dir}. Prepare with --kfolds first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc
    batch = float(args.batch) if "." in str(args.batch) else int(args.batch)
    project = Path(args.project).resolve()
    for yaml_path in yamls:
        fold_name = yaml_path.stem
        print(f"Training {fold_name} using {yaml_path}")
        model = YOLO(args.model)
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=f"{args.name_prefix}_{fold_name}",
            pretrained=True,
            cos_lr=True,
            close_mosaic=10,
        )

if __name__ == "__main__":
    main()