import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


FEATURE_NAMES = [
    "avg_sentence_length",
    "burstiness",
    "type_token_ratio",
    "repetition_rate",
    "punctuation_density",
]


def _get_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _get_sentence_lengths(text: str) -> List[int]:
    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(_get_words(sentence)) for sentence in raw_sentences]
    return lengths if lengths else [0]


def extract_features(text: str) -> Dict[str, float]:
    text = str(text)
    words = _get_words(text)
    total_words = len(words)
    total_chars = len(text)

    sentence_lengths = _get_sentence_lengths(text)
    avg_sentence_length = float(sum(sentence_lengths) / len(sentence_lengths))

    mean_len = avg_sentence_length
    variance = sum((length - mean_len) ** 2 for length in sentence_lengths) / len(
        sentence_lengths
    )
    burstiness = float(variance**0.5)

    unique_words = len(set(words))
    type_token_ratio = float(unique_words / total_words) if total_words else 0.0

    phrases = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    total_phrases = len(phrases)
    phrase_counts = Counter(phrases)
    repeated_phrases = sum(count - 1 for count in phrase_counts.values() if count > 1)
    repetition_rate = float(repeated_phrases / total_phrases) if total_phrases else 0.0

    punct_count = sum(1 for ch in text if ch in string.punctuation)
    punctuation_density = float(punct_count / total_chars) if total_chars else 0.0

    return {
        "avg_sentence_length": avg_sentence_length,
        "burstiness": burstiness,
        "type_token_ratio": type_token_ratio,
        "repetition_rate": repetition_rate,
        "punctuation_density": punctuation_density,
    }


_MODEL_PATH = Path(__file__).resolve().parent / "aidetector_model.pkl"
_MODEL = joblib.load(_MODEL_PATH)


def predict_ai_text(text: str) -> Dict[str, float | str]:
    features = extract_features(text)
    frame = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    pred = int(_MODEL.predict(frame)[0])
    proba = _MODEL.predict_proba(frame)[0]

    confidence = float(proba[pred])
    label = "AI" if pred == 1 else "HUMAN"
    return {"label": label, "confidence": confidence}
