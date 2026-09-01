import sqlite3
from time import time

import cv2
import numpy as np

from .config import DATABASE_PATH, OUTPUT_DIR

def database_connection() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('upload', 'live')),
            source_image TEXT NOT NULL,
            result_image BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
            detection_index INTEGER NOT NULL,
            det_conf REAL NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            cls_conf REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_records_source_time
        ON records(source_type, timestamp DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_detections_record
        ON detections(record_id, detection_index);
        """
    )
    return connection

def save_prediction(
    filename: str,
    result_image: np.ndarray,
    detections: list[dict[str, object]],
    source_type: str = "upload",
) -> int | None:
    if not detections:
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time())
    success, encoded = cv2.imencode(".jpg", result_image)
    if not success:
        raise RuntimeError("Could not encode the record image.")

    with database_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO records (timestamp, source_type, source_image, result_image) VALUES (?, ?, ?, ?)",
            (timestamp, source_type, filename, encoded.tobytes()),
        )
        record_id = int(cursor.lastrowid)
        connection.executemany(
            """
            INSERT INTO detections (
                record_id, detection_index, det_conf, class_id, class_name,
                display_name, cls_conf, x1, y1, x2, y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record_id, row["detection_index"], row["det_conf"], row["class_id"],
                    row["class_name"], row["display_name"], row["cls_conf"],
                    row["x1"], row["y1"], row["x2"], row["y2"],
                )
                for row in detections
            ],
        )
    return record_id

def read_records(source_type: str) -> list[dict[str, object]]:
    with database_connection() as connection:
        record_rows = connection.execute(
            """
            SELECT id, timestamp, source_image
            FROM records
            WHERE source_type = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 80
            """,
            (source_type,),
        ).fetchall()
        records = []
        for row in record_rows:
            detection_rows = connection.execute(
                """
                SELECT display_name, class_name, det_conf, cls_conf
                FROM detections WHERE record_id = ? ORDER BY detection_index
                """,
                (row["id"],),
            ).fetchall()
            detections = [dict(detection) for detection in detection_rows]
            records.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "source_image": row["source_image"],
                    "image_url": f"/record_image/{row['id']}",
                    "detections": detections,
                    "detection_count": len(detections),
                }
            )
    return records