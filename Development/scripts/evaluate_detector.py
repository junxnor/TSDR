"""Evaluate a trained YOLO traffic sign detector and optionally predict unseen images."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from tsdr_common import resolve_weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR = PROJECT_ROOT / "Model Weights" / "detection_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data/mtsd_yolo_detection/data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--source", type=Path, default=None, help="Optional unseen image or folder to run prediction on.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", type=Path, default=Path("runs/detect_eval"))
    parser.add_argument("--name", default="detector_unseen")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep Ultralytics evaluation files instead of removing them after metrics are printed.",
    )
    return parser.parse_args()


def print_metrics(metrics: object) -> None:
    """Print the detector's headline metrics in a terminal-friendly format."""
    box = metrics.box
    print("\n=== DETECTOR EVALUATION RESULT ===")
    print(f"Precision:   {box.mp * 100:.2f}%")
    print(f"Recall:      {box.mr * 100:.2f}%")
    print(f"mAP@50:      {box.map50 * 100:.2f}%")
    print(f"mAP@50-95:   {box.map * 100:.2f}%")


def main() -> None:
    args = parse_args()
    if args.weights and not args.weights.is_absolute():
        args.weights = PROJECT_ROOT / args.weights
    if not args.data.is_absolute():
        args.data = PROJECT_ROOT / args.data
    if not args.project.is_absolute():
        args.project = PROJECT_ROOT / args.project
    weights = resolve_weights(args.weights, DEFAULT_DETECTOR)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: py -m pip install -r requirements.txt") from exc

    model = YOLO(str(weights))
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    temporary_project: Path | None = None
    if args.keep_artifacts:
        project = args.project.resolve()
    else:
        temporary_project = Path(tempfile.mkdtemp(prefix="tsdr-evaluation-"))
        project = temporary_project

    try:
        print(f"Evaluating {weights} on the {args.split} split...")
        metrics = model.val(
            data=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            project=str(project),
            name=f"{args.name}_{args.split}",
            plots=args.keep_artifacts,
        )
        print_metrics(metrics)
    finally:
        if temporary_project:
            shutil.rmtree(temporary_project, ignore_errors=True)

    if args.source:
        if not args.source.exists():
            raise FileNotFoundError(f"Unseen source not found: {args.source}")
        print(f"Predicting unseen source: {args.source}")
        model.predict(
            source=str(args.source),
            imgsz=args.imgsz,
            conf=args.conf,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(args.project.resolve()),
            name=args.name,
        )


if __name__ == "__main__":
    main()
