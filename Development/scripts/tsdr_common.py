"""Shared helpers for the TSDR scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_latest_best_weights(root: Path = Path("runs")) -> Path | None:
    weights = sorted(root.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return weights[0] if weights else None


def resolve_weights(weights: str | Path | None, default: Path | None = None) -> Path:
    if weights:
        path = Path(weights)
    elif default and default.exists():
        path = default
    else:
        latest = find_latest_best_weights()
        if latest is None:
            raise FileNotFoundError("No detector weights found. Pass --weights path/to/best.pt")
        path = latest
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    return path


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def clamp_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))
    return x1, y1, x2, y2


def read_detection_splits(path: Path) -> dict[str, str]:
    split_by_file: dict[str, str] = {}
    if not path.exists():
        return split_by_file
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_by_file[row["filename"]] = row["split"]
    return split_by_file


def safe_stem(path_or_name: str | Path) -> str:
    return Path(path_or_name).stem.replace(" ", "_").replace("(", "").replace(")", "")


def load_class_labels(path: Path = Path("Configuration/class_labels.csv")) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    if not path.exists():
        return labels
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = row.get("class_name", "").strip()
            if not class_name:
                continue
            labels[class_name] = {
                "display_name": row.get("display_name", class_name).strip() or class_name,
                "description": row.get("description", "").strip(),
                "class_id": row.get("class_id", "").strip(),
            }
    return labels


def human_label(class_name: str, labels: dict[str, dict[str, str]]) -> str:
    return labels.get(class_name, {}).get("display_name", class_name)
