from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_WEIGHTS = PROJECT_ROOT / "Model Weights" / "detection_best.pt"
RECOGNIZER_WEIGHTS = PROJECT_ROOT / "Model Weights" / "recognition_best.pt"
CLASS_LABELS_PATH = PROJECT_ROOT / "Configuration" / "class_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "Application" / "instance"
DATABASE_PATH = OUTPUT_DIR / "records.sqlite3"

OPENCV_PREPROCESSING_MODES = {
    "none": "No enhancement",
    "clahe": "CLAHE contrast enhancement",
    "gamma": "Gamma correction",
    "clahe_gamma": "CLAHE + gamma correction",
}

MODEL_PERFORMANCE = {
    "detector": {
        "precision": 92.25,
        "recall": 85.20,
        "map50": 89.94,
        "map50_95": 75.15,
        "best_epoch": 70,
        "train_images": 983,
        "val_images": 211,
        "test_images": 210,
    },
    "recognizer": {
        "top1": 95.79,
        "top5": 99.51,
        "best_epoch": 45,
        "classes": 66,
        "train_images": 5502,
        "val_images": 404,
        "test_images": 411,
    },
}
