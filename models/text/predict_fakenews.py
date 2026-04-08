import re
import string
from pathlib import Path
from typing import Dict

import joblib


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split(" ") if word and word not in STOPWORDS]
    return " ".join(tokens)


_MODEL_PATH = Path(__file__).resolve().parent / "fakenews_model.pkl"
_VECTORIZER_PATH = Path(__file__).resolve().parent / "tfidf.pkl"

_MODEL = joblib.load(_MODEL_PATH)
_VECTORIZER = joblib.load(_VECTORIZER_PATH)


def predict_fakenews(text: str) -> Dict[str, float | str]:
    cleaned = clean_text(text)
    features = _VECTORIZER.transform([cleaned])
    pred = int(_MODEL.predict(features)[0])
    proba = _MODEL.predict_proba(features)[0]

    confidence = float(proba[pred])
    label = "FAKE" if pred == 1 else "REAL"

    return {"label": label, "confidence": confidence}
