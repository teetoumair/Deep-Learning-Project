"""Fashion-MNIST CNN inference API.

A small Flask service that loads the pre-trained Keras model and exposes a
JSON API for classifying 28x28 clothing images.

Endpoints
---------
GET  /                  -> serves the frontend (if present)
GET  /api/health        -> service + model status
GET  /api/model-info    -> layer summary of the loaded CNN
POST /api/predict       -> multipart image upload, returns class + probabilities
POST /predict           -> alias of /api/predict
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DOCUMENT_DIR = BASE_DIR / "Document"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "fashion_cnn_model.h5"))
REPORT_FILENAME = "Fashion-MNIST_CNN_Project_Report.docx"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

IMAGE_SIZE = (28, 28)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "4")) * 1024 * 1024
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fashion-cnn")

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.json.sort_keys = False

cors_origins = (
    "*"
    if ALLOWED_ORIGINS.strip() == "*"
    else [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
)
CORS(app, resources={r"/api/*": {"origins": cors_origins}, r"/predict": {"origins": cors_origins}})

_model = None
_model_error: str | None = None
_model_lock = threading.Lock()
_predict_lock = threading.Lock()


def _load_legacy_h5(path: Path):
    """Rebuild an HDF5 model whose config contains keys this Keras rejects.

    Newer Keras releases serialise an extra ``quantization_config`` entry on
    Dense layers. Older runtimes (e.g. Keras 3.10 on Python 3.9) refuse that
    key, so strip it and rebuild through the legacy deserialiser instead.
    """
    import json

    import h5py
    from keras.src.legacy.saving import saving_utils

    with h5py.File(path, "r") as handle:
        raw = handle.attrs.get("model_config")

    if raw is None:
        raise ValueError("No model_config attribute found in the HDF5 file.")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    config = json.loads(raw)

    def strip_unsupported(node):
        if isinstance(node, dict):
            node.pop("quantization_config", None)
            for value in node.values():
                strip_unsupported(value)
        elif isinstance(node, list):
            for value in node:
                strip_unsupported(value)

    strip_unsupported(config)
    model = saving_utils.model_from_config(config)
    model.load_weights(str(path))
    return model


def load_model(force: bool = False):
    """Load the Keras model once and cache it. Returns the model or None."""
    global _model, _model_error

    if _model is not None and not force:
        return _model

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        logger.info("Loading Keras model from %s", MODEL_PATH)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        import tensorflow as tf

        try:
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception as exc:
            logger.warning("Direct load failed (%s); using legacy HDF5 loader.", exc)
            model = _load_legacy_h5(MODEL_PATH)

        expected = (None, 28, 28, 1)
        input_shape = tuple(model.input_shape) if hasattr(model, "input_shape") else None
        if input_shape and input_shape[1:] != expected[1:]:
            logger.warning("Unexpected model input shape: %s", input_shape)

        warmup = np.zeros((1, 28, 28, 1), dtype=np.float32)
        model.predict(warmup, verbose=0)

        _model = model
        _model_error = None
        logger.info("Model ready (%s parameters)", f"{model.count_params():,}")
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        _model = None
        _model_error = str(exc)
        logger.exception("Unable to load model: %s", exc)

    return _model


def preprocess_image(raw: bytes, invert: str = "auto") -> tuple[np.ndarray, Image.Image, bool]:
    """Turn arbitrary uploaded bytes into a normalized (1, 28, 28, 1) tensor."""
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")

    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32)

    border = np.concatenate([pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]])
    should_invert = invert == "true" or (invert == "auto" and float(border.mean()) > 127.5)

    if should_invert:
        image = ImageOps.invert(image)
        pixels = np.asarray(image, dtype=np.float32)

    tensor = (pixels / 255.0).reshape(1, 28, 28, 1).astype(np.float32)
    return tensor, image, should_invert


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_uploaded_bytes():
    file = request.files.get("image") or request.files.get("file")
    if file is None or file.filename == "":
        return None, "No image uploaded. Send a multipart/form-data field named 'image'."
    raw = file.read()
    if not raw:
        return None, "The uploaded image is empty."
    if len(raw) > MAX_UPLOAD_BYTES:
        return None, f"Image is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
    return raw, None


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify(success=True, message="Fashion-MNIST CNN API is running.", docs="/api/health")


@app.get("/project-report")
def project_report():
    return send_from_directory(
        DOCUMENT_DIR,
        REPORT_FILENAME,
        as_attachment=True,
        download_name=REPORT_FILENAME,
    )


@app.get("/api/health")
def health():
    return jsonify(
        success=True,
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_path=str(MODEL_PATH),
        model_error=_model_error,
        classes=CLASS_NAMES,
        input_shape=list((None, 28, 28, 1)),
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
    )


@app.get("/api/model-info")
def model_info():
    model = load_model()
    if model is None:
        return jsonify(success=False, error=_model_error or "Model is not loaded."), 503

    layers = []
    for layer in model.layers:
        try:
            shape = list(layer.output_shape)
        except Exception:
            shape = None
        layers.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": shape,
                "parameters": int(layer.count_params()),
            }
        )

    return jsonify(
        success=True,
        name=model.name,
        parameters=int(model.count_params()),
        input_shape=list(model.input_shape),
        output_shape=list(model.output_shape),
        layers=layers,
    )


@app.post("/predict")
@app.post("/api/predict")
def predict():
    started = time.perf_counter()

    raw, error = get_uploaded_bytes()
    if error:
        return jsonify(success=False, error=error), 400

    model = load_model()
    if model is None:
        return jsonify(success=False, error=_model_error or "Model is not loaded."), 503

    invert = (request.form.get("invert") or "auto").lower()
    if invert not in {"auto", "true", "false"}:
        invert = "auto"

    try:
        tensor, preprocessed, inverted = preprocess_image(raw, invert)
    except UnidentifiedImageError:
        return jsonify(success=False, error="The uploaded file is not a valid image."), 400
    except Exception as exc:
        logger.warning("Preprocessing failed: %s", exc)
        return jsonify(success=False, error="Could not process the uploaded image."), 400

    try:
        with _predict_lock:
            probabilities = np.asarray(model.predict(tensor, verbose=0)[0], dtype=np.float64)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        logger.exception("Inference failed: %s", exc)
        return jsonify(success=False, error="Model inference failed."), 500

    if probabilities.ndim != 1 or probabilities.size != len(CLASS_NAMES):
        return jsonify(success=False, error="Model returned an unexpected output shape."), 500

    if probabilities.min() < 0 or not np.isfinite(probabilities).all():
        shifted = probabilities - probabilities.max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
    elif not np.isclose(probabilities.sum(), 1.0, atol=1e-3):
        total = probabilities.sum()
        probabilities = probabilities / total if total else np.full_like(probabilities, 0.1)

    ranked = np.argsort(-probabilities)
    top_index = int(ranked[0])
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    all_probabilities = [
        {
            "class_index": int(i),
            "label": CLASS_NAMES[int(i)],
            "probability": round(float(probabilities[i]), 6),
            "percent": round(float(probabilities[i]) * 100, 2),
        }
        for i in range(len(CLASS_NAMES))
    ]

    return jsonify(
        success=True,
        prediction={
            "class_index": top_index,
            "label": CLASS_NAMES[top_index],
            "confidence": round(float(probabilities[top_index]), 6),
            "confidence_percent": round(float(probabilities[top_index]) * 100, 2),
        },
        top_predictions=[all_probabilities[int(i)] for i in ranked[:3]],
        probabilities=all_probabilities,
        preprocessing={
            "inverted": inverted,
            "size": list(IMAGE_SIZE),
            "preview": image_to_data_url(preprocessed),
        },
        elapsed_ms=elapsed_ms,
    )


@app.errorhandler(413)
def payload_too_large(_error):
    return (
        jsonify(success=False, error=f"Image is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."),
        413,
        {"Connection": "close"},
    )


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/") or request.path == "/predict":
        return jsonify(success=False, error="Endpoint not found."), 404
    return error


@app.errorhandler(500)
def server_error(error):
    logger.error("Unhandled error: %s", error)
    if request.path.startswith("/api/") or request.path == "/predict":
        return jsonify(success=False, error="Internal server error."), 500
    return error


if os.environ.get("SKIP_MODEL_LOAD") != "1":
    load_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
