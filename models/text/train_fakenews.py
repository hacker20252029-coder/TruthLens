import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "text" / "news.csv"
    model_path = Path(__file__).resolve().parent / "fakenews_model.pkl"
    vectorizer_path = Path(__file__).resolve().parent / "tfidf.pkl"

    df = pd.read_csv(data_path)
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).apply(clean_text)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=10000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")


if __name__ == "__main__":
    main()
