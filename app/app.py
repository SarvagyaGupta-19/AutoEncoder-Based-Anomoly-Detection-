"""
Flask Web Application for Anomalous Sound Detection.

Upload a .wav file → system converts to log-mel spectrogram →
CNN Autoencoder scores it → returns classification.

Usage:
    python -m app.app
    Then open http://localhost:5000
"""

import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.autoencoder_evaluate import AnomalyScorer
from src.preprocessing import audio_to_log_mel, save_spectrogram_image

# ────────────────────────────────────────────────────────────
# App Setup
# ────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Lazy-load the scorer (heavy TF model — only load when first needed)
_scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        print("  Loading anomaly detection models...")
        _scorer = AnomalyScorer()
        print("  ✓ Models loaded")
    return _scorer


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


# ────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Main page with upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle audio file upload and return prediction."""

    # Validate upload
    if "audio_file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("index"))

    file = request.files["audio_file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
              "error")
        return redirect(url_for("index"))

    try:
        # Save uploaded file
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        upload_path = os.path.join(config.UPLOAD_FOLDER, unique_name)
        file.save(upload_path)

        # Generate spectrogram image for display
        log_mel = audio_to_log_mel(upload_path)
        spec_filename = unique_name.rsplit(".", 1)[0] + "_spec.png"
        spec_path = os.path.join(config.UPLOAD_FOLDER, spec_filename)
        save_spectrogram_image(log_mel, spec_path)

        # Score
        scorer = get_scorer()
        result = scorer.score_audio_file(upload_path)

        # Render result
        return render_template("index.html",
                               result=result,
                               spectrogram_url=url_for("static",
                                                       filename=f"uploads/{spec_filename}"),
                               filename=file.filename,
                               timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        flash(f"Error processing file: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic access.

    POST /api/predict with multipart form data:
        audio_file: .wav file

    Returns JSON:
        {
            "classification": "NORMAL" | "NEEDS MAINTENANCE" | "ANOMALY",
            "mahalanobis_score": float,
            "reconstruction_error": float,
            "file": str
        }
    """
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio_file in request"}), 400

    file = request.files["audio_file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save temporarily
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(config.UPLOAD_FOLDER, f"api_{uuid.uuid4().hex}.wav")
    file.save(temp_path)

    try:
        scorer = get_scorer()
        result = scorer.score_audio_file(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "models_loaded": _scorer is not None})


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ANOMALOUS SOUND DETECTION — WEB APP")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
