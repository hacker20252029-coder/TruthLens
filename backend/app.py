from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


YOUTUBE_API_KEY = "your-key-here"


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(BACKEND_DIR))
sys.path.append(str(PROJECT_ROOT))

from scraper import scrape_url  # noqa: E402
from youtube_analyzer import analyze_spread  # noqa: E402
from models.text.predict_fakenews import predict_fakenews  # noqa: E402
from models.text.predict_aidetector import predict_ai_text  # noqa: E402
from models.image.predict_image import (  # noqa: E402
    predict_image,
    predict_image_from_bytes,
)
from models.voice.predict_voice import predict_voice_from_bytes  # noqa: E402


app = Flask(__name__, static_folder=str(PROJECT_ROOT / "frontend"))
CORS(app)


def _safe_probability(result: Dict[str, Any], positive_labels: set[str]) -> float:
    label = str(result.get("label", "")).upper()
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    if label in positive_labels:
        return confidence
    return 1.0 - confidence


def _build_url_analysis(scraped: Dict[str, Any]) -> Dict[str, Any]:
    title = scraped.get("title", "")
    text = scraped.get("text", "")
    image_path = scraped.get("image_path", "")

    results: Dict[str, Any] = {
        "fakenews": {"label": "UNKNOWN", "confidence": 0.0},
        "ai_text": {"label": "UNKNOWN", "confidence": 0.0},
        "ai_image": {"label": "UNKNOWN", "confidence": 0.0},
        "spread": {"score": 0.5, "video_count": 0},
    }

    def run_fakenews() -> None:
        try:
            results["fakenews"] = predict_fakenews(text)
        except Exception:
            results["fakenews"] = {"label": "UNKNOWN", "confidence": 0.0}

    def run_ai_text() -> None:
        try:
            results["ai_text"] = predict_ai_text(text)
        except Exception:
            results["ai_text"] = {"label": "UNKNOWN", "confidence": 0.0}

    def run_ai_image() -> None:
        try:
            if image_path:
                results["ai_image"] = predict_image(image_path)
            else:
                results["ai_image"] = {"label": "REAL", "confidence": 0.5}
        except Exception:
            results["ai_image"] = {"label": "UNKNOWN", "confidence": 0.0}

    def run_spread() -> None:
        try:
            results["spread"] = analyze_spread(title, YOUTUBE_API_KEY)
        except Exception:
            results["spread"] = {"score": 0.5, "video_count": 0}

    threads = [
        threading.Thread(target=run_fakenews),
        threading.Thread(target=run_ai_text),
        threading.Thread(target=run_ai_image),
        threading.Thread(target=run_spread),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    fakenews_score = _safe_probability(results["fakenews"], {"FAKE"})
    ai_text_score = _safe_probability(results["ai_text"], {"AI"})
    ai_image_score = _safe_probability(results["ai_image"], {"AI_GENERATED"})
    spread_score = float(results["spread"].get("score", 0.5))
    spread_score = max(0.0, min(1.0, spread_score))

    weighted_risk = (
        0.35 * fakenews_score
        + 0.20 * ai_text_score
        + 0.20 * ai_image_score
        + 0.25 * spread_score
    )
    overall_credibility_score = max(0.0, min(100.0, 100.0 - (weighted_risk * 100.0)))

    return {
        "scraped": scraped,
        "scores": results,
        "breakdown": {
            "fakenews_risk": round(fakenews_score, 4),
            "ai_text_risk": round(ai_text_score, 4),
            "ai_image_risk": round(ai_image_score, 4),
            "spread_risk": round(spread_score, 4),
            "weights": {
                "fakenews": 0.35,
                "ai_text": 0.20,
                "ai_image": 0.20,
                "spread": 0.25,
            },
            "weighted_risk": round(weighted_risk, 4),
        },
        "overall_credibility_score": round(overall_credibility_score, 2),
    }


@app.get("/")
def serve_index() -> Any:
    return send_from_directory(str(PROJECT_ROOT / "frontend"), "index.html")


@app.post("/analyze/url")
def analyze_url() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url", "")
        if not url:
            return jsonify({"error": "Missing url"}), 400

        scraped = scrape_url(url)
        if "error" in scraped:
            return jsonify(scraped), 400

        response = _build_url_analysis(scraped)
        return jsonify(response)
    except Exception:
        return jsonify({"error": "Could not analyze URL"}), 500


@app.post("/analyze/text")
def analyze_text() -> Any:
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        fakenews = predict_fakenews(text)
        ai_text = predict_ai_text(text)
        return jsonify({"fakenews": fakenews, "ai_text": ai_text})
    except Exception:
        return jsonify({"error": "Could not analyze text"}), 500


@app.post("/analyze/image")
def analyze_image() -> Any:
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Missing file"}), 400

        image_bytes = file.read()
        result = predict_image_from_bytes(image_bytes)
        return jsonify({"ai_image": result})
    except Exception:
        return jsonify({"error": "Could not analyze image"}), 500


@app.post("/analyze/audio")
def analyze_audio() -> Any:
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Missing file"}), 400

        audio_bytes = file.read()
        result = predict_voice_from_bytes(audio_bytes)
        return jsonify({"ai_voice": result})
    except Exception:
        return jsonify({"error": "Could not analyze audio"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
