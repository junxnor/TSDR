"""Create a more balanced recognition training set with safe image augmentation.

Validation and test images are copied unchanged. Training classes below the target
count are augmented without horizontal flipping because flipping changes the
meaning of directional traffic signs.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/mtsd_recognition"))
    parser.add_argument("--output", type=Path, default=Path("data/mtsd_recognition_balanced"))
    parser.add_argument("--target", type=int, default=80, help="Minimum training images per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image.convert("RGB")
    result = result.rotate(
        rng.uniform(-8.0, 8.0),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(128, 128, 128),
    )
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.78, 1.22))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.82, 1.20))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.30:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))
    return result


def main() -> None:
    args = parse_args()
    train_source = args.source / "train"
    if not train_source.exists():
        raise FileNotFoundError(f"Training directory not found: {train_source}")
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} exists. Use --force to replace it.")
        shutil.rmtree(args.output)

    rng = random.Random(args.seed)
    for split in ("val", "test"):
        source_split = args.source / split
        if source_split.exists():
            shutil.copytree(source_split, args.output / split)

    total_original = 0
    total_augmented = 0
    for class_dir in sorted(path for path in train_source.iterdir() if path.is_dir()):
        source_images = sorted(
            path for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not source_images:
            (args.output / "train" / class_dir.name).mkdir(parents=True, exist_ok=True)
            continue

        output_class = args.output / "train" / class_dir.name
        output_class.mkdir(parents=True, exist_ok=True)
        for image_path in source_images:
            shutil.copy2(image_path, output_class / image_path.name)
        total_original += len(source_images)

        required = max(0, args.target - len(source_images))
        for index in range(required):
            source_path = source_images[index % len(source_images)]
            with Image.open(source_path) as image:
                generated = augment(image, rng)
                generated.save(
                    output_class / f"aug_{index:04d}_{source_path.stem}.jpg",
                    quality=95,
                )
        total_augmented += required
        print(f"{class_dir.name}: {len(source_images)} original + {required} augmented")

    for metadata_name in ("classes.csv", "crops.csv"):
        metadata = args.source / metadata_name
        if metadata.exists():
            shutil.copy2(metadata, args.output / metadata_name)

    print(f"Balanced dataset written to {args.output}")
    print(f"Original training images: {total_original}")
    print(f"Augmented training images: {total_augmented}")


if __name__ == "__main__":
    main()
