import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "text" / "ai_vs_human.csv"
    model_path = Path(__file__).resolve().parent / "aidetector_model.pkl"

    df = pd.read_csv(data_path)
    df = df[["text", "label"]].dropna()

    features_df = df["text"].astype(str).apply(extract_features).apply(pd.Series)
    x = features_df[FEATURE_NAMES]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("Feature Importances:")
    for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
