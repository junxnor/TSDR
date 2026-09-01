"""Prepare EMTD/MTSD detection data in Ultralytics YOLO format.

Default mode is binary detection: every annotated sign becomes class 0
(`traffic_sign`). Use `--mode multiclass` later if you want YOLO to detect and
recognise sign classes in one model.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("EMTD"))
    parser.add_argument("--output", type=Path, default=Path("data/mtsd_yolo_detection"))
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="copy")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def read_annotations(csv_path: Path) -> dict[str, list[dict[str, int]]]:
    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "Class ID", "xmin", "ymin", "xmax", "ymax"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            filename = row["filename"].strip()
            grouped[filename].append(
                {
                    "class_id": int(row["Class ID"]),
                    "xmin": int(float(row["xmin"])),
                    "ymin": int(float(row["ymin"])),
                    "xmax": int(float(row["xmax"])),
                    "ymax": int(float(row["ymax"])),
                }
            )
    return dict(grouped)


def find_image(image_dir: Path, filename: str) -> Path:
    candidate = image_dir / filename
    if candidate.exists():
        return candidate
    lower_lookup = {p.name.lower(): p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    try:
        return lower_lookup[filename.lower()]
    except KeyError as exc:
        raise FileNotFoundError(f"Image listed in GT.csv was not found: {filename}") from exc


def split_items(items: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * train_ratio)
    val_end = train_end + round(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def yolo_lines(boxes: list[dict[str, int]], image_size: tuple[int, int], class_map: dict[int, int], mode: str) -> list[str]:
    width, height = image_size
    lines: list[str] = []
    for box in boxes:
        xmin = max(0, min(width, box["xmin"]))
        xmax = max(0, min(width, box["xmax"]))
        ymin = max(0, min(height, box["ymin"]))
        ymax = max(0, min(height, box["ymax"]))
        if xmax <= xmin or ymax <= ymin:
            continue
        class_id = 0 if mode == "binary" else class_map[box["class_id"]]
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return lines


def write_yaml(path: Path, dataset_root: Path, names: list[str], has_test: bool = True) -> None:
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.extend(["", "names:"])
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_manifest(path: Path, splits: dict[str, list[str]], annotations: dict[str, list[dict[str, int]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "filename", "objects", "original_class_ids"])
        for split_name, filenames in splits.items():
            for filename in filenames:
                class_ids = sorted({box["class_id"] for box in annotations[filename]})
                writer.writerow([split_name, filename, len(annotations[filename]), " ".join(map(str, class_ids))])


def write_class_map(path: Path, class_map: dict[int, int], mode: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_class_id", "yolo_class_id", "name"])
        if mode == "binary":
            writer.writerow(["all", 0, "traffic_sign"])
        else:
            for original_id, yolo_id in sorted(class_map.items()):
                writer.writerow([original_id, yolo_id, f"sign_{original_id}"])


def materialise_split(
    output: Path,
    split_name: str,
    filenames: list[str],
    image_paths: dict[str, Path],
    annotations: dict[str, list[dict[str, int]]],
    class_map: dict[int, int],
    mode: str,
    link_mode: str,
) -> Counter:
    counts: Counter = Counter()
    for filename in filenames:
        src = image_paths[filename]
        dst_image = output / "images" / split_name / src.name
        dst_label = output / "labels" / split_name / f"{src.stem}.txt"
        link_or_copy(src, dst_image, link_mode)
        with Image.open(src) as image:
            lines = yolo_lines(annotations[filename], image.size, class_map, mode)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for line in lines:
            counts[int(line.split()[0])] += 1
    return counts


def make_folds(output: Path, filenames: list[str], kfolds: int, seed: int, names: list[str]) -> None:
    if kfolds < 2:
        return
    shuffled = filenames[:]
    random.Random(seed).shuffle(shuffled)
    fold_root = output / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    for fold_idx in range(kfolds):
        val = [name for idx, name in enumerate(shuffled) if idx % kfolds == fold_idx]
        train = [name for idx, name in enumerate(shuffled) if idx % kfolds != fold_idx]
        for split_name, split_files in {"train": train, "val": val}.items():
            list_path = fold_root / f"fold_{fold_idx}_{split_name}.txt"
            image_paths = [(output / "images" / "all" / name).resolve().as_posix() for name in split_files]
            list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        yaml_path = fold_root / f"fold_{fold_idx}.yaml"
        lines = [
            f"path: {output.resolve().as_posix()}",
            f"train: {(fold_root / f'fold_{fold_idx}_train.txt').resolve().as_posix()}",
            f"val: {(fold_root / f'fold_{fold_idx}_val.txt').resolve().as_posix()}",
            "",
            "names:",
        ]
        lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gt_csv = args.source_root / "GT.csv"
    image_dir = args.source_root / "Detection"
    if not gt_csv.exists() or not image_dir.exists():
        raise FileNotFoundError("Expected EMTD/GT.csv and EMTD/Detection.")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} already exists. Use --force to overwrite.")
        shutil.rmtree(args.output)

    annotations = read_annotations(gt_csv)
    image_paths = {filename: find_image(image_dir, filename) for filename in annotations}
    original_class_ids = sorted({box["class_id"] for boxes in annotations.values() for box in boxes})
    class_map = {original_id: idx for idx, original_id in enumerate(original_class_ids)}
    names = ["traffic_sign"] if args.mode == "binary" else [f"sign_{class_id}" for class_id in original_class_ids]

    filenames = sorted(annotations)
    splits = split_items(filenames, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    all_counts = Counter()
    for split_name, split_files in splits.items():
        all_counts.update(
            materialise_split(
                args.output,
                split_name,
                split_files,
                image_paths,
                annotations,
                class_map,
                args.mode,
                args.link_mode,
            )
        )

    for filename in filenames:
        src = image_paths[filename]
        link_or_copy(src, args.output / "images" / "all" / src.name, args.link_mode)
        with Image.open(src) as image:
            lines = yolo_lines(annotations[filename], image.size, class_map, args.mode)
        label_path = args.output / "labels" / "all" / f"{src.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_yaml(args.output / "data.yaml", args.output, names)
    write_split_manifest(args.output / "splits.csv", splits, annotations)
    write_class_map(args.output / "class_map.csv", class_map, args.mode)
    make_folds(args.output, filenames, args.kfolds, args.seed, names)

    print(f"Prepared {len(filenames)} annotated images at {args.output}")
    print(f"Mode: {args.mode}; YOLO classes: {len(names)}")
    print("Images:", {split: len(files) for split, files in splits.items()})
    print("Objects:", dict(sorted(all_counts.items())))
    print(f"Main config: {args.output / 'data.yaml'}")
    if args.kfolds >= 2:
        print(f"K-fold configs: {args.output / 'folds'}")


if __name__ == "__main__":
    main()
