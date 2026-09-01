"""Create a traffic sign recognition dataset from GT.csv crops.

The output follows Ultralytics classification layout:

data/mtsd_recognition/
  train/sign_1/*.jpg
  val/sign_1/*.jpg
  test/sign_1/*.jpg
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from tsdr_common import clamp_box, read_detection_splits, safe_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("EMTD"))
    parser.add_argument("--splits", type=Path, default=Path("data/mtsd_yolo_detection/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/mtsd_recognition"))
    parser.add_argument("--split-strategy", choices=["class-stratified", "detection-split"], default="class-stratified")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--min-size", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_class_items(
    item_indices: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[int, str]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    shuffled = item_indices[:]
    rng.shuffle(shuffled)
    n_items = len(shuffled)
    if n_items == 1:
        split_names = ["train"]
    elif n_items == 2:
        split_names = ["train", "val"]
    else:
        n_train = max(1, round(n_items * train_ratio))
        n_val = max(1, round(n_items * val_ratio))
        if n_train + n_val >= n_items:
            n_train = max(1, n_items - 2)
            n_val = 1
        n_test = n_items - n_train - n_val
        split_names = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return {idx: split_names[pos] for pos, idx in enumerate(shuffled)}


def main() -> None:
    args = parse_args()
    gt_csv = args.source_root / "GT.csv"
    image_dir = args.source_root / "Detection"
    if not gt_csv.exists() or not image_dir.exists():
        raise FileNotFoundError("Expected EMTD/GT.csv and EMTD/Detection.")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} exists. Use --force to overwrite.")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    split_by_file = read_detection_splits(args.splits)
    if args.split_strategy == "detection-split" and not split_by_file:
        raise FileNotFoundError(f"Split manifest not found or empty: {args.splits}. Run prepare_yolo_detection.py first.")

    candidates: list[dict[str, object]] = []
    by_class: dict[int, list[int]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    skipped = 0
    class_ids: set[int] = set()

    with gt_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for crop_idx, row in enumerate(reader):
            filename = row["filename"].strip()
            if args.split_strategy == "detection-split" and filename not in split_by_file:
                skipped += 1
                continue
            class_id = int(row["Class ID"])
            image_path = image_dir / filename
            if not image_path.exists():
                skipped += 1
                continue
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                xyxy = (
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                )
                x1, y1, x2, y2 = clamp_box(xyxy, width, height, args.padding)
                if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                    skipped += 1
                    continue
                crop = rgb.crop((x1, y1, x2, y2))

            class_name = f"sign_{class_id}"
            candidate_idx = len(candidates)
            by_class.setdefault(class_id, []).append(candidate_idx)
            candidates.append(
                {
                    "filename": filename,
                    "crop_idx": crop_idx,
                    "class_id": class_id,
                    "class_name": class_name,
                    "crop": crop,
                    "box": (x1, y1, x2, y2),
                }
            )
            class_ids.add(class_id)

    if args.split_strategy == "class-stratified":
        split_by_candidate: dict[int, str] = {}
        rng = random.Random(args.seed)
        for indices in by_class.values():
            split_by_candidate.update(split_class_items(indices, args.train_ratio, args.val_ratio, args.test_ratio, rng))
    else:
        split_by_candidate = {
            idx: split_by_file[str(candidate["filename"])]
            for idx, candidate in enumerate(candidates)
        }

    rows: list[list[object]] = []
    for candidate_idx, candidate in enumerate(candidates):
        split = split_by_candidate[candidate_idx]
        filename = str(candidate["filename"])
        class_id = int(candidate["class_id"])
        class_name = str(candidate["class_name"])
        crop = candidate["crop"]
        x1, y1, x2, y2 = candidate["box"]
        crop_dir = args.output / split / class_name
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{safe_stem(filename)}_{int(candidate['crop_idx']):05d}.jpg"
        crop_path = crop_dir / crop_name
        crop.save(crop_path, quality=95)
        rows.append([split, filename, crop_path.as_posix(), class_id, class_name, x1, y1, x2, y2])
        counts[(split, class_name)] += 1

    with (args.output / "crops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "source_image", "crop_path", "class_id", "class_name", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    with (args.output / "classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name"])
        for class_id in sorted(class_ids):
            writer.writerow([class_id, f"sign_{class_id}"])
            for split in ["train", "val", "test"]:
                (args.output / split / f"sign_{class_id}").mkdir(parents=True, exist_ok=True)

    print(f"Created {len(rows)} recognition crops at {args.output}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Classes: {len(class_ids)}; skipped boxes: {skipped}")
    for split in ["train", "val", "test"]:
        total = sum(count for (split_name, _), count in counts.items() if split_name == split)
        print(f"{split}: {total} crops")


if __name__ == "__main__":
    main()
