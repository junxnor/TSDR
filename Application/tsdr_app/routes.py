import io
from pathlib import Path
from threading import Lock

from flask import Blueprint, Response, jsonify, render_template, request, send_from_directory
from PIL import Image

from . import camera, config, inference, storage

web = Blueprint("web", __name__)
STATIC_ASSETS_DIR = Path(__file__).resolve().parents[1] / "static"
live_session_signs: dict[str, set[str]] = {}
live_session_lock = Lock()

@web.route("/")
@web.route("/detect")
@web.route("/live")
@web.route("/records-page")
@web.route("/about")
@web.route("/model-performance")
def index():
    page_name = {
        "/": "home",
        "/detect": "detect",
        "/live": "live",
        "/records-page": "records",
        "/about": "about",
        "/model-performance": "performance",
    }.get(request.path, "home")
    return render_template(
        "index.html",
        page_name=page_name,
        detector_name=config.DETECTOR_WEIGHTS.as_posix(),
        recognizer_name=config.RECOGNIZER_WEIGHTS.as_posix(),
        model_performance=config.MODEL_PERFORMANCE,
    )

@web.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded."}), 400
    try:
        conf = max(0.05, min(0.95, float(request.form.get("conf", 0.70))))
        min_cls_conf = max(0.0, min(0.95, float(request.form.get("min_cls_conf", 0.50))))
        preprocessing_mode = inference.parse_preprocessing_mode(request.form.get("opencv_mode", "clahe_gamma"))
        image = Image.open(io.BytesIO(image_file.read()))
        result_image, detections = inference.run_tsdr(
            image,
            conf,
            min_cls_conf=min_cls_conf,
            preprocessing_mode=preprocessing_mode,
        )
        source_type = "live" if request.form.get("source_type") == "live" else "upload"
        if source_type == "upload" and detections:
            storage.save_prediction(image_file.filename or "upload.jpg", result_image, detections, source_type="upload")
        elif detections:
            session_id = str(request.form.get("live_session_id", "default"))[:100]
            with live_session_lock:
                seen_signs = live_session_signs.setdefault(session_id, set())
                new_detections = [
                    row for row in detections
                    if str(row["class_name"]) not in seen_signs
                ]
                seen_signs.update(str(row["class_name"]) for row in new_detections)
                if len(live_session_signs) > 100:
                    oldest_session = next(iter(live_session_signs))
                    if oldest_session != session_id:
                        live_session_signs.pop(oldest_session, None)
            if new_detections:
                storage.save_prediction(
                    image_file.filename or "live_frame.jpg",
                    result_image,
                    new_detections,
                    source_type="live",
                )
        return jsonify({"image": inference.image_to_data_url(result_image), "detections": detections})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@web.route("/video_feed")
def video_feed():
    conf = max(0.05, min(0.95, float(request.args.get("conf", 0.70))))
    min_cls_conf = max(0.0, min(0.95, float(request.args.get("min_cls_conf", 0.50))))
    preprocessing_mode = inference.parse_preprocessing_mode(request.args.get("opencv_mode", "clahe_gamma"))
    return Response(camera.camera_frames(conf, min_cls_conf, preprocessing_mode), mimetype="multipart/x-mixed-replace; boundary=frame")


@web.route("/stop_camera", methods=["POST"])
def stop_camera():
    camera.stop()
    return jsonify({"ok": True})

@web.route("/stop_live_session", methods=["POST"])
def stop_live_session():
    session_id = str(request.form.get("live_session_id", ""))[:100]
    if session_id:
        with live_session_lock:
            live_session_signs.pop(session_id, None)
    return jsonify({"ok": True})

@web.route("/records")
def records():
    return jsonify({"upload": storage.read_records("upload"), "live": storage.read_records("live")})

@web.route("/records/<int:record_id>", methods=["DELETE"])
def delete_record(record_id: int):
    with storage.database_connection() as connection:
        cursor = connection.execute("DELETE FROM records WHERE id = ?", (record_id,))
    if cursor.rowcount == 0:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"ok": True, "record_id": record_id})

@web.route("/record_image/<int:record_id>")
def record_image(record_id: int):
    with storage.database_connection() as connection:
        row = connection.execute(
            "SELECT result_image FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return jsonify({"error": "Record image not found."}), 404
    return Response(row["result_image"], mimetype="image/jpeg")

@web.route("/hero-image")
def hero_image():
    return send_from_directory(STATIC_ASSETS_DIR, "IMG_1244.jpg")

@web.route("/logo-image")
def logo_image():
    return send_from_directory(STATIC_ASSETS_DIR, "logo.jpg")
